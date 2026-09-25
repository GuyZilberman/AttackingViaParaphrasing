"""
Victim QA model.

The victim receives a question and returns a short factual answer.
Swapping the underlying model is a one-liner in ExperimentConfig.
"""

import re
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
        Ask the victim model a question and return a cleaned short answer.

        `temperature` overrides the default for this call only; used to draw
        sampled answers when estimating how often a paraphrase fools the victim.
        qwen3 often outputs reasoning before the answer; we extract only the
        final answer using _extract_short_answer().
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


# Patterns that introduce the final answer in reasoning-style outputs
_ANSWER_INTRO = re.compile(
    r"(?i)(?:the answer is|so the answer is|answer:|therefore[,\s]+the answer|"
    r"in summary[,\s]+|the correct answer is|my answer is|"
    r"the answer would be|the answer should be|it is|it's)[:\s]*(.+)",
)
_PREAMBLE = re.compile(
    r"(?i)^(?:the answer is|answer:|so the answer is|therefore[,\s]+"
    r"|simply\s+|just\s+|only\s+)[:\s]*",
)
# Strips "a [descriptor] like/called/named" from candidates
# e.g. "a name like Michelangelo" → "Michelangelo"
_TYPE_LIKE = re.compile(
    r"(?i)^a\s+\w+(?:\s+\w+)?\s+(?:like|called|named|such as)\s+(.+)"
)
# Quotes around an answer, e.g.  simply "Leonardo da Vinci"  or  'Canberra'
_QUOTED = re.compile(r"""['"'""](.+?)['"'""]""")
# Lines that are meta-descriptions of the answer type, not the answer itself
_META_LINE = re.compile(
    r"(?i)^(a\s+(name|place|date|number|year|country|city|person|"
    r"person's name|short phrase|noun|proper noun)|"
    r"the name|the place|the date|the year|unknown|n/?a)\.?$"
)


def _extract_short_answer(text: str) -> str:
    """
    Extract the short factual answer from a potentially verbose model response.

    Strategy:
    1. Look for explicit "the answer is …" patterns and extract their payload.
    2. Look for the last short (≤ 8-word) non-empty line that is not a
       meta-description ("a name", "a place", …).
    3. Fall back to the entire text stripped.

    In each candidate, quoted content (e.g. simply "Canberra") is unwrapped.
    """
    text = text.strip()
    if not text:
        return ""

    # 1. Look for explicit answer-intro pattern anywhere in the text
    for line in text.splitlines():
        m = _ANSWER_INTRO.search(line)
        if m:
            candidate = m.group(1).strip().rstrip(".")
            candidate = re.split(r"[,\-–—]", candidate)[0].strip()
            candidate = _clean_candidate(candidate)
            words = candidate.split()
            if 1 <= len(words) <= 8 and not _META_LINE.match(candidate):
                return candidate

    # 2. Take the last short non-empty line that isn't a meta-description
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in reversed(lines):
        line = _PREAMBLE.sub("", line).strip()
        line = _clean_candidate(line)
        words = line.split()
        if 1 <= len(words) <= 8 and not _META_LINE.match(line):
            return line

    # 3. Fallback: return the whole text (evaluators will handle it)
    return text


def _clean_candidate(s: str) -> str:
    """Strip common preamble words and unwrap quotes from a candidate answer."""
    # Strip "a [descriptor] like/called X" → X
    m = _TYPE_LIKE.match(s)
    if m:
        s = m.group(1).strip().rstrip(".")
    return _unwrap_quotes(s)


def _unwrap_quotes(s: str) -> str:
    """If the entire string is wrapped in matching quotes, strip them."""
    m = _QUOTED.fullmatch(s.strip())
    if m:
        return m.group(1).strip()
    # Also handle: word(s) followed by a quoted value, e.g. 'simply "Paris"'
    # Take the LAST quoted segment if one exists and it's short
    all_quoted = _QUOTED.findall(s)
    if all_quoted:
        last = all_quoted[-1].strip()
        if 1 <= len(last.split()) <= 6:
            return last
    return s
