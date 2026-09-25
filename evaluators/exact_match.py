"""
Exact-match and substring-containment evaluator.

Fast, zero-cost (no model calls), used as a first-pass filter and
as a standalone baseline metric.
"""

from typing import List, Optional

from ollama_client import normalise_text
from evaluators.base import BaseEvaluator, EvalResult


class ExactMatchEvaluator(BaseEvaluator):
    """
    Scores a prediction as correct if, after Unicode normalisation
    (lowercase, strip accents, keep only alphanumeric + spaces), either:
      (a) prediction == ground_truth (exact match), or
      (b) ground_truth ⊆ prediction  (containment), or
      (c) prediction ⊆ ground_truth  (sub-answer containment).

    This is the standard "best_subspan_em" heuristic used in NQ/TriviaQA.
    """

    @property
    def name(self) -> str:
        return "exact_match"

    def evaluate(
        self,
        prediction: str,
        ground_truths: List[str],
        question: Optional[str] = None,  # unused: pure string comparison
    ) -> EvalResult:
        np = normalise_text(prediction)
        for gt in ground_truths:
            ng = normalise_text(gt)
            if np == ng or ng in np or np in ng:
                return EvalResult(
                    correct=True,
                    score=1.0,
                    rationale=f"Normalised match: {np!r} ≈ {ng!r}",
                )
        return EvalResult(
            correct=False,
            score=0.0,
            rationale=f"No match found for prediction {np!r}",
        )
