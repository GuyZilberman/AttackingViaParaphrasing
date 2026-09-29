"""
Experiment orchestrator.

Ties together dataset → attacker → victim → evaluator and writes a
self-contained JSON result file whose name encodes the full configuration.

Result file format
------------------
{
  "metadata": { ...config fields... },
  "complete": true,                        # false while running / if it crashed
  "aborted": null,                         # reason, if stopped after repeated failures
  "not_attempted": [],                     # question ids skipped by that stop
  "failed_questions": [                    # questions skipped after an error
    {"question_id": "...", "original_question": "...", "error": "..."}
  ],
  "results": [
    {
      "question_id": "nq_001",
      "original_question": "...",
      "ground_truths": [...],
      "victim_original_answer": "...",
      "victim_original_correct": {"llm_judge": true},   # per-evaluator
      "paraphrases": [                       # every candidate sent to the victim
        {
          "round": 1,
          "paraphrase": "...",
          "status": "queried",
          "equivalent": true,
          "victim_answer": "...",
          "correct": {"llm_judge": false},
          "rationale": { "llm_judge": "..." },
          "attack_success": {"llm_judge": true},
          "sample_answers": ["...", ...],   # extra victim answers at fitness_temperature
          "fitness": 0.4,                   # fraction of victim answers judged wrong
                                            # (by llm_judge if enabled)
          "fitness_gain": 0.2,              # fitness - original_wrong_rate
          "robust_success": false,          # success AND gain >= robust_margin
          "confirmed_success": true,        # re-test verdict (null: not re-tested)
          "retest": {"answers": [...], "n_wrong": 17, "n": 20,   # fresh answers vs
                     "original_n_wrong": 0, "original_n": 20,    # the original's
                     "p_value": 2e-8}                            # confidence samples
        },
        ...
      ],
      "attack_success_rate": {"llm_judge": 0.3},  # fraction of paraphrases where
                                                  # original was correct AND
                                                  # paraphrase was wrong
      "victim_original_samples": ["...", ...],  # original asked at fitness_temperature
      "original_wrong_rate": 0.2,        # same measure as fitness, on the original
      "original_confidence_samples": ["...", ...],  # cfg.confidence_samples fresh
      "original_confidence_wrong_rate": 0.05,       # answers to the original, and
      "confidence_group": "mostly",      # the share wrong: certain (none),
                                         # mostly (< 25%) or unsure
      "n_robust_successes": 1,
      "n_confirmed_successes": 1,        # successes that held up in the re-test
      "rounds_attempted": 3,
      "rounds": [                            # full iterative-search trace
        {
          "round": 1,
          "parents": ["..."],                # paraphrases evolved (evolutionary search)
          "n_generated": 5, "n_duplicates": 0, "n_rejected": 1, "n_queried": 4,
          "n_errors": 0,                    # candidates whose model calls failed
          "candidates": [
            {"paraphrase": "...", "status": "rejected", "equivalent": false,
             "rejected_by": "leaked_instruction" | "too_different" | "equivalence" |
                            "answer_leak" | "answer_preservation"},
            {"paraphrase": "...", "status": "error", "error": "..."},
            {"paraphrase": "...", "status": "duplicate"},
            { ...same fields as a "paraphrases" entry (status "queried")... }
          ]
        },
        ...
      ],
      "n_valid_victim_queries": 11,
      "successful_attacks": [
        {"round": 2, "paraphrase": "...", "victim_answer": "...",
         "attack_success": {"llm_judge": true}}
      ],
      "n_successful_attacks": {"llm_judge": 1},
      "attack_succeeded": {"llm_judge": true}
    },
    ...
  ],
  "summary": {
      "n_questions": 3,
      "n_paraphrases_per_question": 10,     # per attack round
      "max_rounds": 5,
      "stop_on_success": false,
      "evaluators_used": ["llm_judge"],
      "overall_attack_success_rate": {"llm_judge": 0.28},
      "question_attack_success_rate": {"llm_judge": 0.33},  # questions with
                                                          # >= 1 success
      "total_valid_victim_queries": 33,
      "total_successful_attacks": {"llm_judge": 3},
      "robust_margin": 0.5,
      "total_robust_successes": 1,
      "question_robust_success_rate": 0.33,   # questions with >= 1 robust success
      "victim_baseline_accuracy": {"llm_judge": 0.67},
      "retest_samples": 20,
      "retest_alpha": 0.05,
      "total_confirmed_successes": 1,
      "question_confirmed_success_rate": 0.33,
      "confidence_samples": 20,
      "by_confidence_group": {                # attacked questions only
          "certain": {"n_questions": 2, "n_queried": 40, "n_successes": 1,
                      "n_questions_with_success": 1, "n_confirmed_successes": 1,
                      "n_questions_with_confirmed_success": 1},
          ...
      }
  }
}
"""

import json
import logging
import os
from collections import Counter
from datetime import datetime
from math import comb
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import ExperimentConfig
from dataset import load_questions, QuestionEntry
from ollama_client import OllamaClient
from victim import VictimModel
from attackers import LLMParaphraser
from evaluators import LLMJudgeEvaluator, BaseEvaluator
from utils.question_equivalence_judge import questions_equivalent
from utils.answer_preservation_judge import (
    answer_preservation_verdict, leaks_answer, original_overlap,
)

logger = logging.getLogger(__name__)


def build_evaluators(cfg: ExperimentConfig, client: OllamaClient) -> List[BaseEvaluator]:
    evs: List[BaseEvaluator] = []
    if cfg.evaluator == "llm_judge":
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

    # --- Output file: written after every question so a crash loses at most
    # the question in progress ("complete" is false until the run finishes).
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"{cfg.run_id()}__{timestamp}.json"
    logger.info("Results file: %s", out_path)

    # --- Main loop ---
    results_per_question: List[dict] = []
    failed_questions: List[dict] = []
    output: dict = {}
    consecutive_failures = 0
    aborted: Optional[str] = None
    for q_idx, entry in enumerate(questions):
        logger.info(
            "[%d/%d] Q%s: %s", q_idx + 1, len(questions), entry["id"], entry["question"]
        )
        try:
            results_per_question.append(
                _run_question(cfg, entry, client, victim, attacker, evaluators, evaluator_names)
            )
            consecutive_failures = 0
        except Exception as exc:
            consecutive_failures += 1
            # e.g. the Ollama server stops responding; keep the other questions.
            logger.exception("  Question %s failed and is skipped: %s", entry["id"], exc)
            failed_questions.append({
                "question_id": entry["id"],
                "original_question": entry["question"],
                "error": repr(exc),
            })

        if cfg.max_consecutive_failures and consecutive_failures >= cfg.max_consecutive_failures:
            aborted = (
                f"{consecutive_failures} questions in a row failed "
                f"(last error: {failed_questions[-1]['error'][:200]})"
            )

        output = {
            "metadata": cfg.to_dict(),
            "complete": q_idx == len(questions) - 1 and not aborted,
            "aborted": aborted,
            "not_attempted": (
                [e["id"] for e in questions[q_idx + 1:]] if aborted else []
            ),
            "results": results_per_question,
            "failed_questions": failed_questions,
            "summary": _summarize(cfg, results_per_question, evaluator_names),
        }
        _write_json(out_path, output)

        if aborted:
            logger.error(
                "Stopping the run: %s. %d question(s) not attempted. "
                "Check that the Ollama server and GPUs are working, then re-run.",
                aborted, len(output["not_attempted"]),
            )
            break

    logger.info("Results saved → %s", out_path)
    print(f"\nResults saved → {out_path}")
    if failed_questions:
        print(f"  {len(failed_questions)} question(s) failed and were skipped: "
              f"{[q['question_id'] for q in failed_questions]}")
    if aborted:
        print(f"\n  RUN STOPPED EARLY: {aborted}.\n"
              f"  {len(output['not_attempted'])} question(s) not attempted: "
              f"{output['not_attempted']}\n"
              f"  Check that the Ollama server and GPUs work (nvidia-smi), then re-run.")
    _print_summary(output["summary"], out_path)

    return output


def _run_question(
    cfg: ExperimentConfig,
    entry: QuestionEntry,
    client: OllamaClient,
    victim: VictimModel,
    attacker: LLMParaphraser,
    evaluators: List[BaseEvaluator],
    evaluator_names: List[str],
) -> dict:
    """Baseline + iterative attack for one question; returns its result record."""
    qid = entry["id"]
    question = entry["question"]
    gts = entry["answers"]

    # Victim on original question
    original_answer = victim.answer(question)
    logger.info("  Victim (original): %r", original_answer)

    original_correct: Dict[str, bool] = {}
    for ev in evaluators:
        res = ev.evaluate(original_answer, gts, question=question)
        original_correct[ev.name] = res.correct
        logger.info(
            "  [%s] original correct=%s", ev.name, res.correct
        )

    baseline: Optional[dict] = None
    if any(original_correct.values()):
        rounds, paraphrase_records, baseline = _iterative_attack(
            cfg, client, attacker, victim, evaluators,
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
            "fitness": r["fitness"],
            "fitness_gain": r["fitness_gain"],
            "robust_success": r["robust_success"],
            "confirmed_success": r["confirmed_success"],
            "retest_p_value": r.get("retest", {}).get("p_value"),
        }
        for r in paraphrase_records if any(r["attack_success"].values())
    ]
    n_robust = sum(1 for r in paraphrase_records if r["robust_success"])
    n_confirmed = sum(1 for r in paraphrase_records if r["confirmed_success"])
    n_successes = {
        ev_name: sum(1 for r in paraphrase_records if r["attack_success"].get(ev_name))
        for ev_name in evaluator_names
    }
    logger.info(
        "  Attack finished after %d round(s): %d valid victim queries, "
        "successful attacks per evaluator: %s, robust: %d, confirmed by re-test: %d",
        len(rounds), len(paraphrase_records), n_successes, n_robust, n_confirmed,
    )

    return {
        "question_id": qid,
        "original_question": question,
        "ground_truths": gts,
        "victim_original_answer": original_answer,
        "victim_original_correct": original_correct,
        "victim_original_samples": baseline["sample_answers"] if baseline else [],
        "original_wrong_rate": baseline["wrong_rate"] if baseline else None,
        "original_confidence_samples": baseline["confidence_samples"] if baseline else [],
        "original_confidence_wrong_rate": (
            baseline["confidence_wrong_rate"] if baseline else None
        ),
        "confidence_group": (
            confidence_group(baseline["confidence_wrong_rate"]) if baseline else None
        ),
        "n_robust_successes": n_robust,
        "n_confirmed_successes": n_confirmed,
        "paraphrases": paraphrase_records,
        "attack_success_rate": asr,
        "rounds_attempted": len(rounds),
        "rounds": rounds,
        "n_valid_victim_queries": len(paraphrase_records),
        "successful_attacks": successful_attacks,
        "n_successful_attacks": n_successes,
        "attack_succeeded": {k: v > 0 for k, v in n_successes.items()},
    }


def _summarize(
    cfg: ExperimentConfig, results_per_question: List[dict], evaluator_names: List[str]
) -> dict:
    """Aggregate statistics over the questions completed so far."""
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

    return {
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
        "robust_margin": cfg.robust_margin,
        "total_robust_successes": sum(
            r["n_robust_successes"] for r in results_per_question
        ),
        "question_robust_success_rate": (
            sum(1 for r in results_per_question if r["n_robust_successes"]) / n_q
            if n_q else 0.0
        ),
        "retest_samples": cfg.retest_samples,
        "retest_alpha": cfg.retest_alpha,
        "total_confirmed_successes": sum(
            r["n_confirmed_successes"] for r in results_per_question
        ),
        "question_confirmed_success_rate": (
            sum(1 for r in results_per_question if r["n_confirmed_successes"]) / n_q
            if n_q else 0.0
        ),
        "confidence_samples": cfg.confidence_samples,
        "by_confidence_group": _by_confidence_group(results_per_question, evaluator_names),
    }


def _by_confidence_group(results_per_question: List[dict], evaluator_names: List[str]) -> dict:
    """Attacked questions and their successes per confidence group (guide evaluator)."""
    guide = "llm_judge" if "llm_judge" in evaluator_names else evaluator_names[0]
    groups: Dict[str, dict] = {}
    for r in results_per_question:
        if r.get("confidence_group") is None:
            continue
        g = groups.setdefault(r["confidence_group"], {
            "n_questions": 0, "n_queried": 0, "n_successes": 0, "n_questions_with_success": 0,
            "n_confirmed_successes": 0, "n_questions_with_confirmed_success": 0,
        })
        n_succ = r["n_successful_attacks"].get(guide, 0)
        g["n_questions"] += 1
        g["n_queried"] += r["n_valid_victim_queries"]
        g["n_successes"] += n_succ
        g["n_questions_with_success"] += n_succ > 0
        g["n_confirmed_successes"] += r["n_confirmed_successes"]
        g["n_questions_with_confirmed_success"] += r["n_confirmed_successes"] > 0
    return {name: groups[name] for name in CONFIDENCE_GROUPS if name in groups}


def _write_json(path: Path, data: dict) -> None:
    """Write via a temp file + rename, so a crash never leaves a half-written file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _normalize(text: str) -> str:
    """Key used to detect repeated candidates across rounds."""
    return " ".join(text.lower().split()).rstrip("?.! ")


def _iterative_attack(
    cfg: ExperimentConfig,
    client: OllamaClient,
    attacker: LLMParaphraser,
    victim: VictimModel,
    evaluators: List[BaseEvaluator],
    question: str,
    gts: List[str],
    original_correct: Dict[str, bool],
) -> Tuple[List[dict], List[dict], dict]:
    """
    Run up to cfg.max_rounds attack rounds on one question.

    Each round: generate candidates (informed by all previous attempts),
    drop ones already tried in any round, reject ones that fail
    questions_equivalent() or (cfg.answer_check) answer_preserved() without
    querying the victim, then query + evaluate the rest.

    Every queried candidate gets a fitness in [0, 1]: the fraction of victim
    answers judged wrong, over the greedy answer plus cfg.fitness_samples
    sampled ones. Fitness and the correct/wrong verdict shown to the attacker
    use a single "guide" evaluator: the LLM judge when enabled, else the
    first configured evaluator.

    With cfg.search == "evolutionary", each later round asks the attacker to
    mutate / recombine the cfg.n_parents fittest candidates so far; while no
    candidate has fitness > 0 it explores freely instead.

    Before the first round the victim is also sampled on the ORIGINAL
    question (same sample count and temperature). Its wrong rate is the
    baseline: a paraphrase only demonstrates a framing effect if its
    fitness clearly exceeds it, so each record gets fitness_gain and
    robust_success (success with gain >= cfg.robust_margin).

    Returns (rounds, queried_records, baseline).
    """
    seen = {_normalize(question)}
    history: List[dict] = []   # feedback given to the attacker
    rounds: List[dict] = []
    queried: List[dict] = []
    eval_cache: Dict[str, Tuple[Dict[str, bool], Dict[str, Optional[str]]]] = {}
    ev_names = [ev.name for ev in evaluators]
    guide = "llm_judge" if "llm_judge" in ev_names else ev_names[0]

    def evaluate(answer: str) -> Tuple[Dict[str, bool], Dict[str, Optional[str]]]:
        # Sampled answers repeat a lot; don't re-run (LLM) evaluators on them.
        key = _normalize(answer)
        if key not in eval_cache:
            # Always judged against the ORIGINAL question: the ground truths
            # belong to it, and paraphrases are validated as equivalent.
            results = {
                ev.name: ev.evaluate(answer, gts, question=question)
                for ev in evaluators
            }
            eval_cache[key] = (
                {name: r.correct for name, r in results.items()},
                {name: r.rationale for name, r in results.items()},
            )
        return eval_cache[key]

    # Baseline uncertainty on the original question. The greedy original
    # answer was judged correct (else we would not attack), so it counts as
    # one correct answer, mirroring how fitness counts the greedy answer.
    original_samples = [
        victim.answer(question, temperature=cfg.fitness_temperature)
        for _ in range(cfg.fitness_samples)
    ]
    n_orig_wrong = sum(
        1 for a in original_samples
        if not _is_non_answer(a) and not evaluate(a)[0][guide]
    )
    original_wrong_rate = n_orig_wrong / (1 + len(original_samples))
    logger.info(
        "  Original wrong rate: %.2f (%d/%d sampled answers wrong, %s)",
        original_wrong_rate, n_orig_wrong, 1 + len(original_samples), guide,
    )

    # The victim's confidence on the original, from fresh samples only (the
    # greedy answer is right by selection). Flips track how unsure the victim
    # already is, so results are reported per confidence group.
    confidence_samples = [
        victim.answer(question, temperature=cfg.fitness_temperature)
        for _ in range(cfg.confidence_samples)
    ]
    n_conf_wrong = sum(
        1 for a in confidence_samples
        if not _is_non_answer(a) and not evaluate(a)[0][guide]
    )
    confidence_wrong_rate = (
        n_conf_wrong / len(confidence_samples) if confidence_samples else None
    )
    if confidence_samples:
        logger.info(
            "  Confidence on the original: %d/%d sampled answers wrong (%s)",
            n_conf_wrong, len(confidence_samples), confidence_group(confidence_wrong_rate),
        )

    for rnd in range(1, cfg.max_rounds + 1):
        logger.info("  Attack round %d/%d", rnd, cfg.max_rounds)
        parents = _select_parents(cfg, history)
        if parents:
            logger.info("    Evolving %d parent(s):", len(parents))
            for p in parents:
                logger.info("      fitness=%.2f  %r", p["fitness"], p["paraphrase"])
        candidates = attacker.generate_paraphrases(
            question=question,
            answers=gts,
            n=cfg.n_paraphrases,
            history=history,
            parents=parents,
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

            # One failed model call (timeout, server error) must not end the
            # run: record this candidate as an error and move on.
            try:
                # Free check first: attacker notes leaked into the paraphrase.
                leak = _leaked_instruction(question, cand)
                if leak:
                    print(
                        "\n[LEAKED INSTRUCTION]"
                        f"\nOriginal:   {question}"
                        f"\nParaphrase: {cand}"
                        f"\nReason:     {leak}\n"
                    )
                    record = {"paraphrase": cand, "status": "rejected", "equivalent": False,
                              "rejected_by": "leaked_instruction",
                              "rejection_reason": leak}
                    round_records.append(record)
                    history.append(record)
                    continue

                # Free checks before any model call. Too big a change: subtle
                # phrasing keeps most of the original's wording.
                overlap = original_overlap(question, cand)
                if overlap < cfg.min_original_overlap:
                    print(
                        "\n[TOO DIFFERENT]"
                        f"\nOriginal:   {question}"
                        f"\nParaphrase: {cand}"
                        f"\nKeeps {overlap:.0%} of the original's content words\n"
                    )
                    record = {"paraphrase": cand, "status": "rejected", "equivalent": False,
                              "rejected_by": "too_different",
                              "rejection_reason": f"keeps only {overlap:.0%} of the original's content words"}
                    round_records.append(record)
                    history.append(record)
                    continue

                # A paraphrase that names the answer asks
                # a different question ("For what price did Judas ...?").
                if cfg.answer_check and leaks_answer(question, cand, gts):
                    print(
                        "\n[ANSWER LEAK]"
                        f"\nOriginal:   {question}"
                        f"\nAnswer(s):  {gts}"
                        f"\nParaphrase: {cand}\n"
                    )
                    record = {"paraphrase": cand, "status": "rejected", "equivalent": False,
                              "rejected_by": "answer_leak",
                              "rejection_reason": f"names the answer ({' / '.join(gts)})"}
                    round_records.append(record)
                    history.append(record)
                    continue

                if not questions_equivalent(
                    question, cand, model_name=cfg.equivalence_model, client=client
                ):
                    print(
                        "\n[QUESTION EQUIVALENCE FAILED]"
                        f"\nOriginal:   {question}"
                        f"\nParaphrase: {cand}\n"
                    )
                    record = {"paraphrase": cand, "status": "rejected", "equivalent": False,
                              "rejected_by": "equivalence"}
                    round_records.append(record)
                    history.append(record)
                    continue

                preserved, reason = (
                    answer_preservation_verdict(
                        question, cand, gts, model_name=cfg.preservation_judge_model,
                        client=client,
                    )
                    if cfg.answer_check else (1, "")
                )
                if not preserved:
                    print(
                        "\n[ANSWER PRESERVATION FAILED]"
                        f"\nOriginal:   {question}"
                        f"\nAnswer(s):  {gts}"
                        f"\nParaphrase: {cand}\n"
                    )
                    record = {"paraphrase": cand, "status": "rejected", "equivalent": False,
                              "rejected_by": "answer_preservation",
                              "rejection_reason": reason}
                    round_records.append(record)
                    history.append(record)
                    continue

                para_answer = victim.answer(cand)
                correct_map, rationale_map = evaluate(para_answer)
                non_answer = _is_non_answer(para_answer)
                # Attack succeeds when the original was correct but the paraphrase
                # got a wrong ANSWER (a non-answer is not a factual error).
                attack_success_map: Dict[str, bool] = {
                    name: original_correct.get(name, True) and not ok and not non_answer
                    for name, ok in correct_map.items()
                }

                sample_answers = [
                    victim.answer(cand, temperature=cfg.fitness_temperature)
                    for _ in range(cfg.fitness_samples)
                ]
                wrong_answers = [
                    _normalize(a) for a in [para_answer] + sample_answers
                    if not _is_non_answer(a) and not evaluate(a)[0][guide]
                ]
                n_wrong = len(wrong_answers)
                fitness = n_wrong / (1 + len(sample_answers))
                # The wrong answer this paraphrase most often triggers; parents
                # are picked one per such answer (see _select_parents).
                wrong_answer_key = (
                    Counter(wrong_answers).most_common(1)[0][0] if wrong_answers else None
                )

                logger.info(
                    "    Paraphrase: %r  →  Victim: %r%s  correct=%s  fitness=%.2f (%d/%d wrong, %s)",
                    cand, para_answer, " (non-answer)" if non_answer else "",
                    correct_map, fitness, n_wrong, 1 + len(sample_answers), guide,
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
                    "sample_answers": sample_answers,
                    "victim_non_answer": non_answer,
                    "wrong_answer_key": wrong_answer_key,
                    "fitness": fitness,
                    "fitness_gain": fitness - original_wrong_rate,
                    "robust_success": bool(
                        attack_success_map.get(guide)
                        and fitness - original_wrong_rate >= cfg.robust_margin
                    ),
                    # Set by the re-test after the search (None: not re-tested)
                    "confirmed_success": None,
                }
                round_records.append(record)
                queried.append(record)
                history.append({
                    "paraphrase": cand,
                    "status": "queried",
                    "victim_answer": para_answer,
                    "victim_correct": correct_map[guide] and not non_answer,
                    "victim_non_answer": non_answer,
                    "fitness": fitness,
                    "wrong_answer_key": wrong_answer_key,
                })

                if any(attack_success_map.values()):
                    logger.info(
                        "    *** ATTACK SUCCESS (round %d) ***\n"
                        "        Paraphrase:    %s\n"
                        "        Victim answer: %s\n"
                        "        Per evaluator: %s\n"
                        "        Wrong rate:    %.2f vs %.2f on the original (%s)",
                        rnd, cand, para_answer, attack_success_map,
                        fitness, original_wrong_rate,
                        "ROBUST" if record["robust_success"] else "not robust",
                    )
            except Exception as exc:
                logger.warning("    [candidate failed, skipped] %r: %s", cand, exc)
                round_records.append(
                    {"paraphrase": cand, "status": "error", "error": repr(exc)}
                )

        n_dup = sum(1 for r in round_records if r["status"] == "duplicate")
        n_rej = sum(1 for r in round_records if r["status"] == "rejected")
        n_q = sum(1 for r in round_records if r["status"] == "queried")
        n_err = sum(1 for r in round_records if r["status"] == "error")
        n_succ = sum(
            1 for r in round_records
            if r["status"] == "queried" and any(r["attack_success"].values())
        )
        logger.info(
            "    Round %d/%d done: %d generated, %d duplicates, %d rejected, "
            "%d passed validation (queried), %d successful%s",
            rnd, cfg.max_rounds, len(candidates), n_dup, n_rej, n_q, n_succ,
            f", {n_err} failed" if n_err else "",
        )
        rounds.append({
            "round": rnd,
            "parents": [p["paraphrase"] for p in parents],
            "n_generated": len(candidates),
            "n_duplicates": n_dup,
            "n_rejected": n_rej,
            "n_queried": n_q,
            "n_errors": n_err,
            "candidates": round_records,
        })

        # Mostly failed model calls means the server is down or broken, not
        # that this question is hard: fail the question, so the run's stop
        # rule (max_consecutive_failures) can end the run.
        n_err_total = sum(r["n_errors"] for r in rounds)
        n_tried_total = sum(r["n_generated"] - r["n_duplicates"] for r in rounds)
        if n_err_total >= _MIN_ERRORS_TO_FAIL and 2 * n_err_total >= n_tried_total:
            raise RuntimeError(
                f"{n_err_total} of {n_tried_total} candidates failed with model errors"
            )

        if cfg.stop_on_success and n_succ:
            logger.info("    stop_on_success set; ending search after round %d", rnd)
            break

    # Re-test every success on fresh samples: the search selected it on a few
    # noisy answers. Compared with the original's confidence samples (same
    # temperature), it is confirmed only if it is wrong significantly more often.
    if cfg.retest_samples and confidence_samples:
        for r in queried:
            if not r["attack_success"].get(guide):
                continue
            try:
                answers = [
                    victim.answer(r["paraphrase"], temperature=cfg.fitness_temperature)
                    for _ in range(cfg.retest_samples)
                ]
                n_wrong = sum(
                    1 for a in answers if not _is_non_answer(a) and not evaluate(a)[0][guide]
                )
            except Exception as exc:
                logger.warning("    [re-test failed] %r: %s", r["paraphrase"], exc)
                r["retest"] = {"error": repr(exc)}
                continue
            p_value = fisher_greater(n_wrong, len(answers), n_conf_wrong, len(confidence_samples))
            r["retest"] = {
                "answers": answers,
                "n_wrong": n_wrong,
                "n": len(answers),
                "original_n_wrong": n_conf_wrong,
                "original_n": len(confidence_samples),
                "p_value": p_value,
            }
            r["confirmed_success"] = p_value < cfg.retest_alpha
            logger.info(
                "    Re-test: %d/%d wrong vs %d/%d on the original, p=%.3f -> %s: %r",
                n_wrong, len(answers), n_conf_wrong, len(confidence_samples), p_value,
                "CONFIRMED" if r["confirmed_success"] else "not confirmed", r["paraphrase"],
            )

    return rounds, queried, {
        "sample_answers": original_samples,
        "wrong_rate": original_wrong_rate,
        "confidence_samples": confidence_samples,
        "confidence_wrong_rate": confidence_wrong_rate,
    }


def _select_parents(cfg: ExperimentConfig, history: List[dict]) -> List[dict]:
    """
    Pick up to cfg.n_parents fittest queried candidates so far, at most ONE
    per wrong answer they trigger (wrong_answer_key). Paraphrases that make
    the victim give the same wrong answer exploit the same weakness; without
    this, one high-scoring idea (often a drift to a different question)
    fills every parent slot and the search collapses onto it, as GEPA's
    Pareto selection is designed to avoid. Returns [] for reflective search
    or while no candidate has fooled the victim even once.
    """
    if cfg.search != "evolutionary":
        return []
    scored = [h for h in history if h.get("fitness", 0) > 0]
    # Stable sort: ties keep the earlier candidate first.
    scored.sort(key=lambda h: h["fitness"], reverse=True)
    parents: List[dict] = []
    used_keys = set()
    for h in scored:
        key = h.get("wrong_answer_key")
        if key in used_keys:
            continue
        used_keys.add(key)
        parents.append(h)
        if len(parents) == cfg.n_parents:
            break
    return parents


def fisher_greater(a: int, n1: int, b: int, n2: int) -> float:
    """One-sided Fisher exact p-value that the rate a/n1 exceeds the rate b/n2."""
    k, n = a + b, n1 + n2
    return sum(
        comb(n1, i) * comb(n2, k - i) for i in range(a, min(k, n1) + 1)
    ) / comb(n, k)


# Confidence groups by the share of sampled answers to the original that are
# wrong; the cut-offs match the 2026-09-26 analysis (0/20, 1-4/20, >= 5/20).
CONFIDENCE_GROUPS = ("certain", "mostly", "unsure")


def confidence_group(wrong_rate: Optional[float]) -> Optional[str]:
    """certain: no sampled answer wrong; mostly: under 25% wrong; unsure: 25% or more."""
    if wrong_rate is None:
        return None
    if wrong_rate == 0:
        return "certain"
    return "mostly" if wrong_rate < 0.25 else "unsure"


# Victim replies longer than this are unfinished reasoning (qwen3 ran out of
# tokens before its final answer), not answers: the victim is told to give
# only the shortest answer, and real answers are a few words.
_MAX_ANSWER_WORDS = 40


def _is_non_answer(answer: str) -> bool:
    """True for empty replies and truncated reasoning dumps."""
    return not _normalize(answer) or len(answer.split()) > _MAX_ANSWER_WORDS


# A question fails once at least this many of its candidates hit model errors
# (and they are at least half of those tried).
_MIN_ERRORS_TO_FAIL = 5

# Characters and words that show up when the attacker leaks its own notes or
# instructions into a paraphrase instead of returning a clean question.
_LEAK_CHARS = "()[]{}\""
# ("synonym" is left out: real paraphrases use it, e.g. "Another synonym for
# the torso is ...?".)
_LEAK_WORDS = ("avoid", "replaced", "paraphrase", "rephrase", "reworded",
               "rewording", "original question", "note:")


def _leaked_instruction(original: str, cand: str) -> Optional[str]:
    """
    Reason string if the candidate carries attacker notes rather than being a
    clean question (e.g. "... (replaced with correct synonym 'distinguished')
    is not valid (avoid adding new text)"), else None. Only flags characters
    and words the original does not itself contain.
    """
    orig_l, cand_l = original.lower(), cand.lower()
    for ch in _LEAK_CHARS:
        if ch in cand and ch not in original:
            return f"adds {ch!r}, which the original does not contain"
    for w in _LEAK_WORDS:
        if w in cand_l and w not in orig_l:
            return f"contains instruction word {w!r}"
    return None


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
    print(f"  Robust successes (gain >= {summary['robust_margin']:.2f} over the "
          f"original's wrong rate): {summary['total_robust_successes']}, on "
          f"{summary['question_robust_success_rate']:.1%} of questions")
    if summary["retest_samples"]:
        print(f"  Confirmed by re-test ({summary['retest_samples']} fresh answers, one-sided "
              f"Fisher p < {summary['retest_alpha']}): {summary['total_confirmed_successes']}, "
              f"on {summary['question_confirmed_success_rate']:.1%} of questions")
    if summary.get("by_confidence_group"):
        print(f"  By the victim's confidence on the original "
              f"({summary['confidence_samples']} sampled answers):")
        for name, g in summary["by_confidence_group"].items():
            print(f"    {name:8s} {g['n_questions']:3d} question(s): {g['n_successes']} successes "
                  f"({g['n_confirmed_successes']} confirmed) in {g['n_queried']} queried "
                  f"paraphrases, on {g['n_questions_with_success']} question(s)")
    print("=" * 60)
    print(f"  Full results at: {out_path}")
    print("=" * 60)
