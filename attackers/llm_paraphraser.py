"""
LLM-based adversarial paraphraser.

Each "strategy" is a named system-prompt template that instructs the attacker
LLM to rewrite questions using semantically equivalent but potentially
challenging phrasing.
Swapping strategy = changing the STRATEGIES key used at construction time.

Adding a new strategy: add an entry to the STRATEGIES dict below.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Tuple

from ollama_client import OllamaClient
from attackers.base import BaseAttacker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy definitions
# Each value is (system_prompt_template, user_message_template).
# Available placeholders: {question}, {answers_str}, {n}
# ---------------------------------------------------------------------------

STRATEGIES: Dict[str, Tuple[str, str]] = {
    # ------------------------------------------------------------------
    # Adversarial strategy
    # ------------------------------------------------------------------
    "misleading_entity": (
        # System
        "You are an adversarial paraphrase generator for AI robustness research.\n"
        "Task: rewrite a question using subtle changes in phrasing while preserving "
        "its EXACT semantic meaning and factual content.\n"
        "The goal is to find alternative phrasings that may cause a black-box LLM "
        "to answer incorrectly even though the correct answer remains unchanged.\n"
        "Rules:\n"
        "- Every paraphrase must have exactly the SAME correct answer as the original question.\n"
        "- Preserve all named entities, dates, quantities, relationships, constraints, "
        "negations, geographic scope, and temporal scope.\n"
        "- Do NOT add, remove, weaken, strengthen, or alter any factual assumption.\n"
        "- Do NOT replace the subject or object with a different entity.\n"
        "- Change only wording, syntax, sentence structure, word order, voice, or "
        "semantically equivalent expressions.\n"
        "- The paraphrase must remain natural, grammatical, and unambiguous.\n"
        "- Do NOT use typos, invisible characters, unusual Unicode tricks, or formatting tricks.\n"
        "- Prefer subtle rephrasings rather than radically different sentence forms.\n"
        "- Every variant must be a complete question.\n"
        'Output ONLY this JSON object: {"paraphrases": ["q1", "q2", ...]}',
        # User
        'Original question: "{question}"\n'
        "Known correct answer(s): {answers_str}\n\n"
        "Generate {n} semantically equivalent adversarial paraphrases.\n"
        "Each paraphrase must preserve the exact meaning and require the SAME answer, "
        "while varying the phrasing in a way that could expose sensitivity to framing.\n"
        'Output ONLY valid JSON: {{"paraphrases": ["q1", "q2", ...]}}',
    ),
    # ------------------------------------------------------------------
    # Control / baseline
    # ------------------------------------------------------------------
    "semantic_preserve": (
        "You are a paraphrase generator.\n"
        "Task: rewrite questions in different words while preserving their "
        "EXACT meaning. The paraphrases must require the SAME answer as the original.\n"
        "Rules: use synonyms, restructure the sentence, change voice or word order. "
        "Do NOT change the factual content or add new constraints.\n"
        'Output ONLY this JSON object: {"paraphrases": ["q1", "q2", ...]}',
        'Original question: "{question}"\n\n'
        "Generate {n} genuine paraphrases that preserve the full meaning.\n"
        'Output ONLY valid JSON: {{"paraphrases": ["q1", "q2", ...]}}',
    ),
}


# ---------------------------------------------------------------------------
# Feedback appended to the user message in later attack rounds (shared by all
# strategies): the history of previous attempts, followed by either the
# EXPLORE instruction (no promising parents yet) or the EVOLVE instruction
# (mutate / recombine the highest-fitness paraphrases so far).
# ---------------------------------------------------------------------------

HISTORY_TEMPLATE = (
    "\n\nThe following paraphrases have ALREADY been tried in previous rounds. "
    "Their validation status and the victim model's results are listed below:\n"
    "{history_str}\n\n"
    "Legend:\n"
    "- REJECTED: judged NOT equivalent to the original, or the known correct "
    "answer no longer answers it; never sent to the victim. Avoid whatever "
    "drift in meaning they introduced.\n"
    "- VICTIM CORRECT: equivalent, but the victim still answered correctly.\n"
    "- VICTIM WRONG: equivalent, and the victim answered incorrectly; the "
    "framing worked.\n"
    "- 'wrong in X% of samples': how often the victim answered incorrectly when "
    "asked repeatedly. Higher means the phrasing is closer to fooling it."
)

# The evolutionary search rewards whatever makes the victim answer "wrongly",
# which pulls candidates toward neighbouring questions; spell out the most
# common drifts explicitly.
SAME_QUESTION_RULE = (
    " Keep the same question type and ask for the same attribute: never turn "
    "an open question into a yes/no question, and never ask for a function, "
    "purpose, role, or reason instead of what the original asks for."
)

EXPLORE_INSTRUCTION = (
    "\n\nNow generate {n} NEW paraphrases. Do NOT repeat or closely imitate any "
    "previous attempt. Use the outcomes above to explore different wording, "
    "syntax, sentence structure, voice, or framing, while preserving EXACTLY "
    "the same meaning, entities, time, scope, factual assumptions, and correct answer."
    + SAME_QUESTION_RULE
)

EVOLVE_INSTRUCTION = (
    "\n\nThe most promising paraphrases so far (highest rate of wrong victim "
    "answers) are the PARENTS for this round:\n"
    "{parents_str}\n\n"
    "Now generate {n} NEW paraphrases by evolving the parents. Each new "
    "paraphrase must be either:\n"
    "- a MUTATION of one parent: a small change to its wording, word order, "
    "voice, or framing that keeps what made it effective, or\n"
    "- a CROSSOVER of two parents: combine phrasing elements from both.\n"
    "Do NOT output any parent or any previous attempt verbatim. Every new "
    "paraphrase must still be exactly equivalent to the ORIGINAL question: same "
    "meaning, entities, time, scope, factual assumptions, and correct answer."
    + SAME_QUESTION_RULE
)


class LLMParaphraser(BaseAttacker):
    """
    Uses an Ollama LLM to generate adversarial paraphrases of questions.

    Strategy selection determines the system prompt used, making it trivial
    to add new attack styles without touching anything else.

    Args:
        client:      Shared OllamaClient instance.
        model:       Ollama model tag for the attacker.
        strategy:    Key into STRATEGIES dict.
        temperature: Higher → more diverse / creative paraphrases.
        max_tokens:  Budget for the JSON list the model returns.
        retries:     How many times to retry on parse failure.
    """

    def __init__(
        self,
        client: OllamaClient,
        model: str = "qwen3:4b",
        strategy: str = "misleading_entity",
        temperature: float = 0.9,
        max_tokens: int = 2048,
        retries: int = 1,
    ):
        if strategy not in STRATEGIES:
            raise ValueError(
                f"Unknown strategy {strategy!r}. Available: {list(STRATEGIES)}"
            )

        self.client = client
        self.model = model
        self.strategy = strategy
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries

        self._sys_tmpl, self._usr_tmpl = STRATEGIES[strategy]

    @property
    def name(self) -> str:
        return f"llm_paraphraser_{self.strategy}"

    def generate_paraphrases(
        self,
        question: str,
        answers: List[str],
        n: int = 10,
        history: Optional[List[dict]] = None,
        parents: Optional[List[dict]] = None,
    ) -> List[str]:
        """
        Generate up to `n` adversarial paraphrases for `question`.

        When `history` is given (later attack rounds), previous attempts and
        their outcomes are appended to the prompt so the attacker produces new
        phrasings informed by what did and did not work. When `parents` is
        also given (evolutionary search), the attacker is asked to mutate /
        recombine those paraphrases instead of exploring freely.

        Candidates are NOT checked for semantic equivalence here; the caller
        must validate them before querying the victim.

        Returns a deduplicated list that may be shorter than `n`.
        """
        answers_str = ", ".join(f'"{a}"' for a in answers)

        system = self._sys_tmpl
        user = self._usr_tmpl.format(
            question=question,
            answers_str=answers_str,
            n=n,
        )
        if history:
            user += HISTORY_TEMPLATE.format(history_str=_format_history(history))
            if parents:
                user += EVOLVE_INSTRUCTION.format(
                    parents_str=_format_history(parents),
                    n=n,
                )
            else:
                user += EXPLORE_INSTRUCTION.format(n=n)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        last_error: Optional[Exception] = None
        for attempt in range(1 + self.retries):
            try:
                parsed = self.client.chat_json(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    think=False,
                )
                paraphrases = _extract_string_list(parsed, question)

                if paraphrases:
                    return paraphrases

                logger.warning(
                    "[attacker] Attempt %d/%d: no JSON list found, retrying…",
                    attempt + 1, 1 + self.retries,
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "[attacker] Attempt %d/%d failed: %s",
                    attempt + 1, 1 + self.retries, exc,
                )

        logger.error(
            "[attacker] All attempts exhausted for question %r. Last error: %s",
            question, last_error,
        )
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_history(history: List[dict]) -> str:
    """Render previous attempts (or parents) as a numbered list for the attacker prompt."""
    lines = []
    for i, h in enumerate(history, start=1):
        if h["status"] == "rejected":
            outcome = {
                "answer_preservation": "REJECTED (known answer no longer fits)",
                "answer_leak": "REJECTED (mentions the answer)",
            }.get(h.get("rejected_by"), "REJECTED (not equivalent)")
            if h.get("rejection_reason"):
                outcome += f" - {h['rejection_reason'][:150]}"
        elif h.get("victim_non_answer"):
            outcome = "VICTIM GAVE NO ANSWER"
        elif h.get("victim_correct"):
            outcome = f"VICTIM CORRECT (answered {h['victim_answer']!r})"
        else:
            outcome = f"VICTIM WRONG (answered {h['victim_answer']!r})"
        if h.get("fitness") is not None:
            outcome += f", wrong in {h['fitness']:.0%} of samples"
        lines.append(f'{i}. "{h["paraphrase"]}" -> {outcome}')
    return "\n".join(lines)


# "q1: ...", "Q2) ...", "1. ...", "- ..." list labels at the start of a paraphrase
_LABEL_PREFIX = re.compile(r"^\s*(?:q\s*\d+\s*[:.)\-]|\d+[.)]\s|[-*•]\s)\s*", re.IGNORECASE)


def _extract_string_list(parsed: object, original_question: str) -> List[str]:
    """
    Tolerate various shapes the model might return:
      - A plain list of strings                        → use directly
      - {"paraphrases": [...]}                          → unwrap
      - {"questions": [...]} or {"variants": [...]}     → unwrap
      - A list of dicts with "question" or "text" key  → extract values
    """
    if isinstance(parsed, list):
        items = parsed
    elif isinstance(parsed, dict):
        # Try common wrapper keys
        for key in ("paraphrases", "questions", "variants", "results", "output"):
            if key in parsed and isinstance(parsed[key], list):
                items = parsed[key]
                break
        else:
            # Take the first list value we find
            lists = [v for v in parsed.values() if isinstance(v, list)]
            items = lists[0] if lists else []
    else:
        return []

    results: List[str] = []
    seen = set()
    for item in items:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            text = (
                item.get("question")
                or item.get("paraphrase")
                or item.get("text")
                or ""
            ).strip()
        else:
            continue

        # Models sometimes copy the "q1", "q2" placeholders from the JSON
        # format (or number the list); such prefixes would reach the victim.
        text = _LABEL_PREFIX.sub("", text)

        # Deduplicate and exclude trivial exact copies
        key = text.lower()
        if text and key not in seen and key != original_question.lower():
            seen.add(key)
            results.append(text)

    return results