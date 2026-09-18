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
      "paraphrases": [
        {
          "paraphrase": "...",
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
      }
    },
    ...
  ],
  "summary": {
      "n_questions": 3,
      "n_paraphrases_per_question": 10,
      "evaluators_used": ["exact_match", "llm_judge"],
      "overall_attack_success_rate": {
          "exact_match": 0.35,
          "llm_judge": 0.28
      },
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
from typing import Dict, List, Optional

from config import ExperimentConfig
from dataset import load_questions, QuestionEntry
from ollama_client import OllamaClient
from victim import VictimModel
from attackers import LLMParaphraser
from evaluators import ExactMatchEvaluator, LLMJudgeEvaluator, BaseEvaluator

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

        # Generate paraphrases
        logger.info(
            "  Generating %d paraphrases (strategy=%s)…",
            cfg.n_paraphrases, cfg.attacker_strategy,
        )
        paraphrases = attacker.generate_paraphrases(
            question=question,
            answers=gts,
            n=cfg.n_paraphrases,
        )
        logger.info("  Got %d paraphrases", len(paraphrases))

        paraphrase_records = []
        for p_idx, para in enumerate(paraphrases):
            para_answer = victim.answer(para)
            logger.info(
                "    [%d] Paraphrase: %r  →  Victim: %r", p_idx + 1, para, para_answer
            )

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

            paraphrase_records.append({
                "paraphrase": para,
                "victim_answer": para_answer,
                "correct": correct_map,
                "rationale": rationale_map,
                "attack_success": attack_success_map,
            })

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

        results_per_question.append({
            "question_id": qid,
            "original_question": question,
            "ground_truths": gts,
            "victim_original_answer": original_answer,
            "victim_original_correct": original_correct,
            "paraphrases": paraphrase_records,
            "attack_success_rate": asr,
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

    summary = {
        "n_questions": n_q,
        "n_paraphrases_per_question": cfg.n_paraphrases,
        "evaluators_used": evaluator_names,
        "overall_attack_success_rate": overall_asr,
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


def _print_summary(summary: dict, out_path: Path) -> None:
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"  Questions evaluated : {summary['n_questions']}")
    print(f"  Paraphrases / Q     : {summary['n_paraphrases_per_question']}")
    print()
    for ev_name in summary["evaluators_used"]:
        bacc = summary["victim_baseline_accuracy"].get(ev_name, 0)
        asr  = summary["overall_attack_success_rate"].get(ev_name, 0)
        print(f"  [{ev_name}]")
        print(f"    Victim baseline accuracy : {bacc:.1%}")
        print(f"    Overall Attack Success Rate (ASR) : {asr:.1%}")
    print("=" * 60)
    print(f"  Full results at: {out_path}")
    print("=" * 60)
