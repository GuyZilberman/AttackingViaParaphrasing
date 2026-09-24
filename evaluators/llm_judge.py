"""
LLM-as-a-Judge evaluator (Ollama-backed).

Replaces the transformers-based utils/llm_as_a_judge.py with an equivalent
that uses the Ollama HTTP API, so no local model weights are needed beyond
what Ollama manages.

The judge scores a single (prediction, ground_truth) pair as 0 or 1 and
returns a short rationale.  `evaluate()` runs the judge against all
ground truths and returns 1 if any passes.
"""

import logging
from typing import List, Optional

from ollama_client import OllamaClient, normalise_text
from evaluators.base import BaseEvaluator, EvalResult

logger = logging.getLogger(__name__)

_JUDGE_SYSTEM = (
    "You are a strict QA evaluation judge.\n"
    "Decide if a prediction is semantically equivalent to a ground truth answer.\n"
    "\n"
    "Scoring rules:\n"
    "- Score 1 if meanings match, even with extra descriptors, aliases, "
    "reordered words, or containment relationships.\n"
    "- Score 1 for abbreviations or nicknames that clearly refer to the same entity.\n"
    "- Score 0 if they refer to different entities or directly contradict.\n"
    "- If both mention geographic/entity qualifiers that conflict, score 0.\n"
    "- Ignore casing, punctuation, whitespace, and articles (the/a/an).\n"
    "\n"
    'Output ONLY this JSON object: {"score": 0 or 1, "rationale": "one sentence"}'
)

_JUDGE_USER_TMPL = (
    "/no_think\n"
    'Prediction: "{prediction}"\n'
    'Ground truth: "{ground_truth}"\n\n'
    'Output ONLY: {{"score": 0 or 1, "rationale": "..."}}'
)


class LLMJudgeEvaluator(BaseEvaluator):
    """
    Uses an Ollama model to judge whether `prediction` is semantically
    equivalent to any of the `ground_truths`.

    A fast normalisation pre-check (exact / containment) short-circuits
    obvious matches before hitting the LLM.

    Args:
        client:      Shared OllamaClient.
        model:       Ollama model tag for the judge.
        temperature: Should be 0 for determinism.
        use_fast_path: Skip LLM call on obvious exact/containment matches.
    """

    def __init__(
        self,
        client: OllamaClient,
        model: str = "qwen3:4b",
        temperature: float = 0.0,
        use_fast_path: bool = True,
    ):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.use_fast_path = use_fast_path

    @property
    def name(self) -> str:
        return "llm_judge"

    def evaluate(
        self,
        prediction: str,
        ground_truths: List[str],
    ) -> EvalResult:
        """
        Returns EvalResult with score=1.0 if any ground truth matches,
        else score=0.0.
        """
        for gt in ground_truths:
            result = self._score_pair(prediction, gt)
            if result.correct:
                return result
        return EvalResult(
            correct=False,
            score=0.0,
            rationale="No ground truth matched.",
        )

    def _score_pair(self, prediction: str, ground_truth: str) -> EvalResult:
        # Fast path: exact / containment after normalisation
        if self.use_fast_path:
            np_ = normalise_text(prediction)
            ng = normalise_text(ground_truth)
            if np_ == ng or ng in np_ or np_ in ng:
                return EvalResult(
                    correct=True,
                    score=1.0,
                    rationale="Fast-path normalised match.",
                )

        messages = [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {
                "role": "user",
                "content": _JUDGE_USER_TMPL.format(
                    prediction=prediction, ground_truth=ground_truth
                ),
            },
        ]

        try:
            parsed = self.client.chat_json(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=512,
                think=False,
            )
        except Exception as exc:
            logger.warning("[judge] LLM call failed: %s", exc)
            return EvalResult(correct=False, score=0.0,
                              rationale=f"Judge error: {exc}")

        score_val = _extract_score(parsed)
        rationale: str = ""

        if isinstance(parsed, dict):
            rationale = parsed.get("rationale") or parsed.get("reason") or ""

        if score_val is None:
            logger.warning("[judge] Could not extract score from output: %r", parsed)
            return EvalResult(correct=False, score=0.0,
                              rationale="Could not parse judge output.")

        correct = int(round(float(score_val))) >= 1
        return EvalResult(
            correct=correct,
            score=float(correct),
            rationale=rationale or str(parsed)[:120],
        )


def _extract_score(parsed: object) -> Optional[float]:
    """Pull a numeric score out of various JSON shapes."""
    if isinstance(parsed, dict):
        for k, v in parsed.items():
            if k.lower() == "score":
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
    if isinstance(parsed, (int, float)):
        return float(parsed)
    return None

