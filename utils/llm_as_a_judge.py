"""
LLM-as-a-Judge utility (Ollama-backed).

Rewritten to use the Ollama HTTP API instead of transformers/torch.
The public API is unchanged so callers need no modifications:
  - best_llm_judge(prediction, ground_truths)  → float (0.0 or 1.0)
  - llm_score_pair(prediction, ground_truth)   → int  (0 or 1)
"""

import os
import sys

# Allow running from the utils/ subdirectory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ollama_client import OllamaClient, normalise_text, _parse_json_robust
from evaluators.llm_judge import LLMJudgeEvaluator
from typing import List, Optional

# ---------------------------------------------------------------------------
# Configuration (override with environment variables)
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
MODEL = os.environ.get("JUDGE_MODEL", "qwen3:4b")

_client: Optional[OllamaClient] = None
_evaluator: Optional[LLMJudgeEvaluator] = None


def _get_evaluator() -> LLMJudgeEvaluator:
    global _client, _evaluator
    if _evaluator is None:
        _client = OllamaClient(base_url=OLLAMA_BASE_URL)
        _evaluator = LLMJudgeEvaluator(client=_client, model=MODEL, temperature=0.0)
    return _evaluator


def llm_score_pair(
    prediction: str,
    ground_truth: str,
    use_fast_path: bool = True,
    # tok and model args kept for backwards-compatibility but ignored
    tok=None,
    model=None,
) -> int:
    """
    Returns 1 if prediction is semantically equivalent to ground_truth, else 0.
    """
    ev = _get_evaluator()
    ev.use_fast_path = use_fast_path
    result = ev._score_pair(prediction, ground_truth)
    return 1 if result.correct else 0


def best_llm_judge(prediction: str, ground_truths: List[str]) -> float:
    """
    Returns 1.0 if prediction matches any ground truth, else 0.0.
    API is compatible with the original transformers-based version.
    """
    ev = _get_evaluator()
    result = ev.evaluate(prediction, ground_truths)
    return result.score


# ---------------------------------------------------------------------------
# Standalone test runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_cases = [
        ("Paris", ["the city of Paris", "Paris, France"], 1),
        ("Barack Obama", ["President Barack H. Obama", "Obama"], 1),
        ("Paris, Texas", ["Paris, France"], 0),
        ("USA", ["United States of America", "U.S."], 1),
        ("NYC", ["New York City"], 1),
        ("42", ["forty two", "forty-two"], 1),
        ("Cat", ["Dog"], 0),
        ("New York", ["New Jersey"], 0),
    ]

    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"

    passed = 0
    for pred, gts, expected in test_cases:
        score = best_llm_judge(pred, gts)
        ok = (score == expected)
        if ok:
            passed += 1
            tag = f"{GREEN}PASS{RESET}"
        else:
            tag = f"{RED}FAIL{RESET}"
            print(f"  {tag}  pred={pred!r}  gts={gts}  score={score}  expected={expected}")

    total = len(test_cases)
    print(f"\nSummary: {passed}/{total} passed")
