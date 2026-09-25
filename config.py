"""
Central configuration for attack experiments.

Every axis that might vary across runs (model choice, attack strategy,
number of samples, …) lives here so that swapping components is a
one-liner on the command line.
"""

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from typing import List, Literal, Optional


# ---------------------------------------------------------------------------
# Available strategies (prompt templates live in attackers/llm_paraphraser.py)
# ---------------------------------------------------------------------------

ATTACKER_STRATEGIES = [
    "misleading_entity",   # swap a key entity to a plausible-but-wrong alternative
    "temporal_shift",      # modify dates / sequences / order
    "scope_change",        # narrow or widen scope / geographic qualifier
    "presupposition",      # embed a false premise inside the question
    "semantic_preserve",   # genuine paraphrase — control / baseline (should NOT attack)
]

EVALUATOR_CHOICES = ["exact_match", "llm_judge", "both"]


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    # ---- Ollama server ----
    ollama_base_url: str = "http://127.0.0.1:11434"

    # ---- Models ----
    attacker_model: str = "qwen3:4b"
    victim_model: str = "qwen3:4b"
    judge_model: str = "qwen3:4b"

    # ---- Attack ----
    attacker_strategy: str = "misleading_entity"
    n_paraphrases: int = 10  # paraphrases generated per attack round
    max_rounds: int = 5      # iterative attack rounds per question
    stop_on_success: bool = False  # stop a question's search at its first success

    # ---- Dataset ----
    # None → use built-in data/sample_questions.json
    dataset_path: Optional[str] = None
    n_questions: int = 3          # how many questions to sample
    random_seed: int = 42

    # ---- Evaluation ----
    evaluator: str = "both"       # "exact_match" | "llm_judge" | "both"

    # ---- Output ----
    results_dir: str = "results"

    # ---- Generation settings ----
    attacker_temperature: float = 0.9   # higher → more diverse paraphrases
    victim_temperature: float = 0.0     # greedy for reproducibility
    judge_temperature: float = 0.0

    def validate(self) -> None:
        if self.attacker_strategy not in ATTACKER_STRATEGIES:
            raise ValueError(
                f"Unknown strategy {self.attacker_strategy!r}. "
                f"Choose from: {ATTACKER_STRATEGIES}"
            )
        if self.evaluator not in EVALUATOR_CHOICES:
            raise ValueError(
                f"Unknown evaluator {self.evaluator!r}. "
                f"Choose from: {EVALUATOR_CHOICES}"
            )
        if self.n_paraphrases < 1:
            raise ValueError("n_paraphrases must be >= 1")
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        if self.n_questions < 1:
            raise ValueError("n_questions must be >= 1")

    def run_id(self) -> str:
        """
        Unique, filesystem-safe identifier for this configuration.
        Used as the output filename stem.
        """
        def slug(s: str) -> str:
            return re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_")

        return (
            f"attacker_{slug(self.attacker_model)}_{slug(self.attacker_strategy)}"
            f"__victim_{slug(self.victim_model)}"
            f"__judge_{slug(self.judge_model)}"
            f"__q{self.n_questions}_p{self.n_paraphrases}_r{self.max_rounds}"
        )

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> ExperimentConfig:
    defaults = ExperimentConfig()

    parser = argparse.ArgumentParser(
        description="Run a subtle-paraphrasing adversarial attack experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Ollama
    parser.add_argument("--ollama-base-url", default=defaults.ollama_base_url)

    # Models
    parser.add_argument("--attacker-model", default=defaults.attacker_model,
                        help="Ollama model tag for the paraphrase generator")
    parser.add_argument("--victim-model", default=defaults.victim_model,
                        help="Ollama model tag for the victim QA model")
    parser.add_argument("--judge-model", default=defaults.judge_model,
                        help="Ollama model tag for the LLM-as-a-judge evaluator")

    # Attack
    parser.add_argument("--attacker-strategy", default=defaults.attacker_strategy,
                        choices=ATTACKER_STRATEGIES,
                        help="Paraphrasing strategy / attack type")
    parser.add_argument("--n-paraphrases", type=int, default=defaults.n_paraphrases,
                        help="Number of adversarial paraphrases per attack round")
    parser.add_argument("--max-rounds", type=int, default=defaults.max_rounds,
                        help="Maximum number of iterative attack rounds per question")
    parser.add_argument("--stop-on-success", action="store_true",
                        help="Stop attacking a question after its first successful paraphrase")

    # Dataset
    parser.add_argument("--dataset-path", default=None,
                        help="Path to a JSON or CSV question file (default: built-in samples)")
    parser.add_argument("--n-questions", type=int, default=defaults.n_questions,
                        help="Number of questions to sample from the dataset")
    parser.add_argument("--random-seed", type=int, default=defaults.random_seed)

    # Evaluation
    parser.add_argument("--evaluator", default=defaults.evaluator,
                        choices=EVALUATOR_CHOICES)

    # Output
    parser.add_argument("--results-dir", default=defaults.results_dir)

    # Generation temperatures
    parser.add_argument("--attacker-temperature", type=float,
                        default=defaults.attacker_temperature)
    parser.add_argument("--victim-temperature", type=float,
                        default=defaults.victim_temperature)
    parser.add_argument("--judge-temperature", type=float,
                        default=defaults.judge_temperature)

    args = parser.parse_args(argv)

    cfg = ExperimentConfig(
        ollama_base_url=args.ollama_base_url,
        attacker_model=args.attacker_model,
        victim_model=args.victim_model,
        judge_model=args.judge_model,
        attacker_strategy=args.attacker_strategy,
        n_paraphrases=args.n_paraphrases,
        max_rounds=args.max_rounds,
        stop_on_success=args.stop_on_success,
        dataset_path=args.dataset_path,
        n_questions=args.n_questions,
        random_seed=args.random_seed,
        evaluator=args.evaluator,
        results_dir=args.results_dir,
        attacker_temperature=args.attacker_temperature,
        victim_temperature=args.victim_temperature,
        judge_temperature=args.judge_temperature,
    )
    cfg.validate()
    return cfg
