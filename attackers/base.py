"""
Abstract base class for all attacker / paraphrase-generator strategies.

A new attack strategy only needs to implement `generate_paraphrases`.
"""

from abc import ABC, abstractmethod
from typing import List


class BaseAttacker(ABC):
    """
    Generates adversarial paraphrases of a question.

    Every concrete subclass must implement `generate_paraphrases`.
    The contract:
      - Input:  original question text  +  list of known correct answers
      - Output: list of paraphrased question strings

    The paraphrases should look semantically similar to a human reader but
    be crafted to lead an LLM victim model to answer incorrectly.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in result filenames and logs."""

    @abstractmethod
    def generate_paraphrases(
        self,
        question: str,
        answers: List[str],
        n: int = 10,
    ) -> List[str]:
        """
        Args:
            question: The original question string.
            answers:  List of acceptable correct answer strings.
            n:        How many paraphrases to generate.

        Returns:
            List of paraphrase strings (may be shorter than n if the model
            refuses or returns duplicates).
        """
