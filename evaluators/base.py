"""
Abstract base class for answer correctness evaluators.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EvalResult:
    """
    Result from a single (prediction, ground_truth_list) evaluation.

    Attributes:
        correct:   True if the prediction is semantically correct.
        score:     Numeric score in [0, 1] (1.0 = correct).
        rationale: Optional explanation from the judge.
    """
    correct: bool
    score: float
    rationale: Optional[str] = None


class BaseEvaluator(ABC):
    """
    Decides whether a model's prediction matches any of the ground-truth answers.

    Every concrete evaluator must implement `evaluate`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in results JSON."""

    @abstractmethod
    def evaluate(
        self,
        prediction: str,
        ground_truths: List[str],
    ) -> EvalResult:
        """
        Args:
            prediction:    The model's answer string.
            ground_truths: List of acceptable correct answer strings.

        Returns:
            EvalResult with at minimum `correct` and `score` filled in.
        """
