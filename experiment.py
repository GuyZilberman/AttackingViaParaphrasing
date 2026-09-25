"""
Experiment orchestrator.

Ties together dataset → attacker → victim → evaluator and writes a
self-contained JSON result file whose name encodes the full configuration.

Result file format
------------------
{
  "metadata": { ...config fields... },
  "results": [
    {
      "question_id": "nq_001",
      "original_question": "...",
      "ground_truths": [...],
      "victim_original_answer": "...",
      "victim_original_correct": {                  # per-evaluator
          "exact_match": true,
          "llm_judge": true
      },
      "paraphrases": [                       # every candidate sent to the victim
        {
          "round": 1,
          "paraphrase": "...",
          "status": "queried",
          "equivalent": true,
          "victim_answer": "...",
          "correct": { "exact_match": false, "llm_judge": false },
          "rationale": { "llm_judge": "..." },
          "attack_success": { "exact_match": true, "llm_judge": true }
        },
        ...
      ],
      "attack_success_rate": {               # fraction of paraphrases where
          "exact_match": 0.4,               # original was correct AND
          "llm_judge": 0.3                  # paraphrase was wrong
      },
      "rounds_attempted": 3,
      "rounds": [                            # full iterative-search trace
        {
          "round": 1,
          "n_generated": 5, "n_duplicates": 0, "n_rejected": 1, "n_queried": 4,
          "candidates": [
            {"paraphrase": "...", "status": "rejected", "equivalent": false},
            {"paraphrase": "...", "status": "duplicate"},
            { ...same fields as a "paraphrases" entry (status "queried")... }
          ]
        },
        ...
      ],
      "n_valid_victim_queries": 11,
      "successful_attacks": [
        {"round": 2, "paraphrase": "...", "victim_answer": "...",
         "attack_success": {"exact_match": true, "llm_judge": true}}
      ],
      "n_successful_attacks": {"exact_match": 1, "llm_judge": 1},
      "attack_succeeded": {"exact_match": true, "llm_judge": true}
    },
    ...
  ],
  "summary": {
      "n_questions": 3,
      "n_paraphrases_per_question": 10,     # per attack round
      "max_rounds": 5,
      "stop_on_success": false,
      "evaluators_used": ["exact_match", "llm_judge"],
      "overall_attack_success_rate": {
          "exact_match": 0.35,
          "llm_judge": 0.28
      },
      "question_attack_success_rate": {     # fraction of questions with
          "exact_match": 0.67,              # >= 1 successful paraphrase
          "llm_judge": 0.33
      },
      "total_valid_victim_queries": 33,
      "total_successful_attacks": {"exact_match": 4, "llm_judge": 3},
      "victim_baseline_accuracy": {
          "exact_match": 0.67,
          "llm_judge": 0.67
      }
  }
}
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import ExperimentConfig
from dataset import load_questions, QuestionEntry
from ollama_client import OllamaClient
from victim import VictimModel
from attackers import LLMParaphraser
from evaluators import ExactMatchEvaluator, LLMJudgeEvaluator, BaseEvaluator
from utils.question_equivalence_judge import questions_equivalent

logger = logging.getLogger(__name__)


def build_evaluators(cfg: ExperimentConfig, client: OllamaClient) -> List[BaseEvaluator]:
    evs: List[BaseEvaluator] = []
    if cfg.evaluator in ("exact_match", "both"):
        evs.append(ExactMatchEvaluator())
    if cfg.evaluator in ("llm_judge", "both"):
        evs.append(LLMJudgeEvaluator(
            client=client,
            model=cfg.judge_model,
            temperature=cfg.judge_temperature,
        ))
    return evs


def run_experiment(cfg: ExperimentConfig) -> dict:
    """
    Execute a full experiment run and return the results dict.
    Also writes the dict to a JSON file in cfg.results_dir.
    """
    cfg.validate()

    # --- Infrastructure ---
    client = OllamaClient(base_url=cfg.ollama_base_url)
    client.require_available()

    victim = VictimModel(
        client=client,
        model=cfg.victim_model,
        temperature=cfg.victim_temperature,
    )
    attacker = LLMParaphraser(
        client=client,
        model=cfg.attacker_model,
        strategy=cfg.attacker_strategy,
        temperature=cfg.attacker_temperature,
    )
    evaluators = build_evaluators(cfg, client)
    evaluator_names = [e.name for e in evaluators]

    # --- Dataset ---
    questions = load_questions(
        path=cfg.dataset_path,
        n=cfg.n_questions,
        seed=cfg.random_seed,
    )
    logger.info("Loaded %d questions", len(questions))

    # --- Main loop ---
    results_per_question = []
    for q_idx, entry in enumerate(questions):
        qid = entry["id"]
        question = entry["question"]
        gts = entry["answers"]

        logger.info(
            "[%d/%d] Q%s: %s", q_idx + 1, len(questions), qid, question
        )

        # Victim on original question
        original_answer = victim.answer(question)
        logger.info("  Victim (original): %r", original_answer)

        original_correct: Dict[str, bool] = {}
        for ev in evaluators:
            res = ev.evaluate(original_answer, gts)
            original_correct[ev.name] = res.correct
            logger.info(
                "  [%s] original correct=%s", ev.name, res.correct
            )

        if any(original_correct.values()):
            rounds, paraphrase_records = _iterative_attack(
                cfg, attacker, victim, evaluators,
                question, gts, original_correct,
            )
        else:
            # No paraphrase can count as a successful attack; skip the search.
            logger.info(
                "  Victim already wrong on the original (all evaluators); "
                "skipping attack."
            )
            rounds, paraphrase_records = [], []

        # Per-question attack success rate
        asr: Dict[str, float] = {}
        for ev_name in evaluator_names:
            if original_correct.get(ev_name, True) and paraphrase_records:
                successes = sum(
                    1 for r in paraphrase_records if r["attack_success"].get(ev_name)
                )
                asr[ev_name] = successes / len(paraphrase_records)
            else:
                # If victim was already wrong on the original, ASR is undefined (0)
                asr[ev_name] = 0.0

        successful_attacks = [
            {
                "round": r["round"],
                "paraphrase": r["paraphrase"],
                "victim_answer": r["victim_answer"],
                "attack_success": r["attack_success"],
            }
            for r in paraphrase_records if any(r["attack_success"].values())
        ]
        n_successes = {
            ev_name: sum(1 for r in paraphrase_records if r["attack_success"].get(ev_name))
            for ev_name in evaluator_names
        }
        logger.info(
            "  Attack finished after %d round(s): %d valid victim queries, "
            "successful attacks per evaluator: %s",
            len(rounds), len(paraphrase_records), n_successes,
        )

        results_per_question.append({
            "question_id": qid,
            "original_question": question,
            "ground_truths": gts,
            "victim_original_answer": original_answer,
            "victim_original_correct": original_correct,
            "paraphrases": paraphrase_records,
            "attack_success_rate": asr,
            "rounds_attempted": len(rounds),
            "rounds": rounds,
            "n_valid_victim_queries": len(paraphrase_records),
            "successful_attacks": successful_attacks,
            "n_successful_attacks": n_successes,
            "attack_succeeded": {k: v > 0 for k, v in n_successes.items()},
        })

    # --- Summary ---
    n_q = len(results_per_question)
    overall_asr: Dict[str, float] = {}
    baseline_acc: Dict[str, float] = {}
    for ev_name in evaluator_names:
        overall_asr[ev_name] = (
            sum(r["attack_success_rate"][ev_name] for r in results_per_question) / n_q
            if n_q else 0.0
        )
        baseline_acc[ev_name] = (
            sum(1 for r in results_per_question if r["victim_original_correct"].get(ev_name))
            / n_q
            if n_q else 0.0
        )

    question_asr: Dict[str, float] = {
        ev_name: (
            sum(1 for r in results_per_question if r["attack_succeeded"][ev_name]) / n_q
            if n_q else 0.0
        )
        for ev_name in evaluator_names
    }

    summary = {
        "n_questions": n_q,
        "n_paraphrases_per_question": cfg.n_paraphrases,
        "max_rounds": cfg.max_rounds,
        "stop_on_success": cfg.stop_on_success,
        "evaluators_used": evaluator_names,
        "overall_attack_success_rate": overall_asr,
        "question_attack_success_rate": question_asr,
        "total_valid_victim_queries": sum(
            r["n_valid_victim_queries"] for r in results_per_question
        ),
        "total_successful_attacks": {
            ev_name: sum(r["n_successful_attacks"][ev_name] for r in results_per_question)
            for ev_name in evaluator_names
        },
        "victim_baseline_accuracy": baseline_acc,
    }

    output = {
        "metadata": cfg.to_dict(),
        "results": results_per_question,
        "summary": summary,
    }

    # --- Save ---
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{cfg.run_id()}__{timestamp}.json"
    out_path = results_dir / filename

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    logger.info("Results saved → %s", out_path)
    print(f"\nResults saved → {out_path}")
    _print_summary(summary, out_path)

    return output


def _normalize(text: str) -> str:
    """Key used to detect repeated candidates across rounds."""
    return " ".join(text.lower().split()).rstrip("?.! ")


def _iterative_attack(
    cfg: ExperimentConfig,
    attacker: LLMParaphraser,
    victim: VictimModel,
    evaluators: List[BaseEvaluator],
    question: str,
    gts: List[str],
    original_correct: Dict[str, bool],
) -> Tuple[List[dict], List[dict]]:
    """
    Run up to cfg.max_rounds attack rounds on one question.

    Each round: generate candidates (informed by all previous attempts),
    drop ones already tried in any round, reject non-equivalent ones without
    querying the victim, then query + evaluate the rest.

    Returns (rounds, queried_records).
    """
    seen = {_normalize(question)}
    history: List[dict] = []   # feedback given to the attacker
    rounds: List[dict] = []
    queried: List[dict] = []

    for rnd in range(1, cfg.max_rounds + 1):
        logger.info("  Attack round %d/%d", rnd, cfg.max_rounds)
        candidates = attacker.generate_paraphrases(
            question=question,
            answers=gts,
            n=cfg.n_paraphrases,
            history=history,
        )
        logger.info("    Generated %d candidates", len(candidates))

        round_records: List[dict] = []
        for cand in candidates:
            key = _normalize(cand)
            if key in seen:
                logger.info("    [duplicate, skipped] %r", cand)
                round_records.append({"paraphrase": cand, "status": "duplicate"})
                continue
            seen.add(key)

            if not questions_equivalent(question, cand):
                print(
                    "\n[QUESTION EQUIVALENCE FAILED]"
                    f"\nOriginal:   {question}"
                    f"\nParaphrase: {cand}\n"
                )
                record = {"paraphrase": cand, "status": "rejected", "equivalent": False}
                round_records.append(record)
                history.append(record)
                continue

            para_answer = victim.answer(cand)
            correct_map: Dict[str, bool] = {}
            rationale_map: Dict[str, Optional[str]] = {}
            attack_success_map: Dict[str, bool] = {}
            for ev in evaluators:
                res = ev.evaluate(para_answer, gts)
                correct_map[ev.name] = res.correct
                rationale_map[ev.name] = res.rationale
                # Attack succeeds when the original was correct but the paraphrase is wrong
                attack_success_map[ev.name] = (
                    original_correct.get(ev.name, True) and not res.correct
                )

            logger.info(
                "    Paraphrase: %r  →  Victim: %r  correct=%s",
                cand, para_answer, correct_map,
            )
            record = {
                "round": rnd,
                "paraphrase": cand,
                "status": "queried",
                "equivalent": True,
                "victim_answer": para_answer,
                "correct": correct_map,
                "rationale": rationale_map,
                "attack_success": attack_success_map,
            }
            round_records.append(record)
            queried.append(record)
            history.append({
                "paraphrase": cand,
                "status": "queried",
                "victim_answer": para_answer,
                "victim_correct": all(correct_map.values()),
            })

            if any(attack_success_map.values()):
                logger.info(
                    "    *** ATTACK SUCCESS (round %d) ***\n"
                    "        Paraphrase:    %s\n"
                    "        Victim answer: %s\n"
                    "        Per evaluator: %s",
                    rnd, cand, para_answer, attack_success_map,
                )

        n_dup = sum(1 for r in round_records if r["status"] == "duplicate")
        n_rej = sum(1 for r in round_records if r["status"] == "rejected")
        n_q = sum(1 for r in round_records if r["status"] == "queried")
        n_succ = sum(
            1 for r in round_records
            if r["status"] == "queried" and any(r["attack_success"].values())
        )
        logger.info(
            "    Round %d/%d done: %d generated, %d duplicates, %d rejected, "
            "%d passed equivalence validation (queried), %d successful",
            rnd, cfg.max_rounds, len(candidates), n_dup, n_rej, n_q, n_succ,
        )
        rounds.append({
            "round": rnd,
            "n_generated": len(candidates),
            "n_duplicates": n_dup,
            "n_rejected": n_rej,
            "n_queried": n_q,
            "candidates": round_records,
        })

        if cfg.stop_on_success and n_succ:
            logger.info("    stop_on_success set; ending search after round %d", rnd)
            break

    return rounds, queried


def _print_summary(summary: dict, out_path: Path) -> None:
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"  Questions evaluated : {summary['n_questions']}")
    print(f"  Paraphrases / round : {summary['n_paraphrases_per_question']}")
    print(f"  Max rounds          : {summary['max_rounds']}")
    print(f"  Valid victim queries: {summary['total_valid_victim_queries']}")
    print()
    for ev_name in summary["evaluators_used"]:
        bacc = summary["victim_baseline_accuracy"].get(ev_name, 0)
        asr  = summary["overall_attack_success_rate"].get(ev_name, 0)
        print(f"  [{ev_name}]")
        print(f"    Victim baseline accuracy : {bacc:.1%}")
        print(f"    Overall Attack Success Rate (ASR) : {asr:.1%}")
        print(f"    Questions with >= 1 success       : "
              f"{summary['question_attack_success_rate'].get(ev_name, 0):.1%}")
        print(f"    Total successful paraphrases      : "
              f"{summary['total_successful_attacks'].get(ev_name, 0)}")
    print("=" * 60)
    print(f"  Full results at: {out_path}")
    print("=" * 60)
