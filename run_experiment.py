#!/usr/bin/env python3
"""
Entry point for the adversarial paraphrasing attack experiment.

Usage
-----
  # Quick smoke test (built-in questions; attacker llama3.1:8b, victim qwen3:4b, judge gemma3:12b):
  python3 run_experiment.py --n-questions 3 --n-paraphrases 5

  # Iterative attack: up to 5 rounds of 5 paraphrases each per question:
  python3 run_experiment.py --n-questions 3 --n-paraphrases 5 --max-rounds 5

  # The curated question set (hand-reviewed NQ-Open questions, minimally amended):
  python3 run_experiment.py --dataset-path data/accepted_questions.json --n-questions 10

  # Harder questions (real search queries from NQ-Open; build once with
  # python3 utils/download_nq_open.py):
  python3 run_experiment.py --dataset-path data/nq_open_sample.json --n-questions 10

  # Reflective search only (no parent selection, no victim sampling):
  python3 run_experiment.py --search reflective --fitness-samples 0

  # Change the attack strategy:
  python3 run_experiment.py --attacker-strategy temporal_shift --n-questions 3 --n-paraphrases 5

  # Use different models for attacker vs victim (when more models are available):
  python3 run_experiment.py \
      --attacker-model llama3:8b \
      --victim-model qwen3:4b \
      --judge-model qwen3:4b

  # Load questions from a custom CSV file:
  python3 run_experiment.py --dataset-path my_questions.csv
"""

import logging
import sys

from config import parse_args
from experiment import run_experiment


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = parse_args()

    print("\n" + "=" * 60)
    print("ADVERSARIAL PARAPHRASING EXPERIMENT")
    print("=" * 60)
    print(f"  Attacker model    : {cfg.attacker_model}")
    print(f"  Attack strategy   : {cfg.attacker_strategy}")
    print(f"  Victim model      : {cfg.victim_model}")
    print(f"  Judge model       : {cfg.judge_model}  (evaluator={cfg.evaluator})")
    print(f"  Questions         : {cfg.n_questions}")
    print(f"  Paraphrases/round : {cfg.n_paraphrases}")
    print(f"  Max rounds        : {cfg.max_rounds}  (stop_on_success={cfg.stop_on_success})")
    print(f"  Answer check      : "
          f"{cfg.preservation_judge_model if cfg.answer_check else 'off'}")
    print(f"  Search            : {cfg.search}  (parents={cfg.n_parents}, "
          f"fitness samples={cfg.fitness_samples} @ T={cfg.fitness_temperature})")
    print(f"  Dataset           : {cfg.dataset_path or 'built-in (data/sample_questions.json)'}")
    print(f"  Results dir       : {cfg.results_dir}")
    print("=" * 60 + "\n")

    run_experiment(cfg)


if __name__ == "__main__":
    main()
