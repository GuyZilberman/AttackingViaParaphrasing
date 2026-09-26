"""
Victim QA model.

The victim receives a question and returns a short factual answer.
Swapping the underlying model is a one-liner in ExperimentConfig.
"""

from typing import Optional

from ollama_client import OllamaClient

_SYSTEM_PROMPT = (
    "Answer the question with only the shortest direct answer. "
    "Do not explain. Do not repeat the question. "
    "Do not use a full sentence unless necessary. "
    "Return only the answer itself."
)


class VictimModel:
    """
    Wraps an Ollama model acting as the QA system being attacked.

    Args:
        client:      OllamaClient instance (shared across components).
        model:       Ollama model tag, e.g. "qwen3:4b".
        temperature: 0.0 = greedy / deterministic answers.
        max_tokens:  Enough room for reasoning + short final answer. qwen3
                     reasons inside the reply even with think=False; on
                     harder questions 512 tokens often cut it off before
                     </think>, leaving no answer to extract.
    """

    def __init__(
        self,
        client: OllamaClient,
        model: str = "qwen3:4b",
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def answer(self, question: str, temperature: Optional[float] = None) -> str:
        """
        Ask the victim model a question and return its answer verbatim,
        minus any leading <think>…</think> reasoning block.

        `temperature` overrides the default for this call only; used to draw
        sampled answers when estimating how often a paraphrase fools the victim.
        """
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ]
        raw = self.client.chat(
            model=self.model,
            messages=messages,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens,
            think=False,
        )

        if "</think>" in raw:
            raw = raw.rsplit("</think>", 1)[1]

        return raw.strip()
