"""
Answer-preservation judge: a second validation gate after
questions_equivalent().

questions_equivalent() compares the two questions without knowing the answer,
and in practice it lets through paraphrases that drift to a neighbouring
question (e.g. "Which motor neurons control skeletal muscles?" →
"What is the primary function of the nerves controlling skeletal muscles?",
or "What age do you need to be to buy a bb gun?" → "Can a minor buy a bb gun?").
Once the search rewards paraphrases that make the victim answer "wrongly",
it actively seeks out that drift.

This judge closes the gap with a concrete test: given the known correct
answer(s), would that same answer still be a correct, complete, and natural
answer to the paraphrase? A candidate must pass BOTH judges before it is sent
to the victim.
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ollama_client import normalise_text
from utils.question_equivalence_judge import _extract_score, client

# Scored with tests/eval_judges.py on tests/data/answer_preservation_labels.json.
# On the first 46 paraphrases (14 drifted), direct prompt:
#   gemma3:12b        → 2 drifted accepted,  4 valid rejected
#   mistral-nemo:12b  → 2 drifted accepted,  6 valid rejected
#   qwen3:4b          → 1 drifted accepted, 12 valid rejected
# After adding the drift the evolutionary search actually found, gemma3:12b:
#   direct prompt (82 items, 29 drifted)                → 17 drifted accepted, 4 valid rejected
#   structured prompt + leaks_answer() (86 items, 33)   →  4 drifted accepted, 9 valid rejected
# qwen3:4b is the victim: as the gate deciding which paraphrases are tried,
# it could reject exactly the ones that confuse it, hiding real successes
# (it rejected every paraphrase of the WW1 question). gemma3:12b is from a
# different model family. Its known weakness: dropped time qualifiers
# ("after WW1 started" -> "after WW1") can slip through.
DEFAULT_MODEL = "gemma3:12b"


ANSWER_PRESERVATION_SYSTEM = (
    "You are a strict judge that outputs ONLY valid JSON.\n"
    "You receive an ORIGINAL question, its known correct answer(s), and a "
    "CANDIDATE rephrasing of the question.\n"
    "Decide whether the known correct answer is ALSO a correct, complete, and "
    "natural answer to the CANDIDATE, exactly as it is for the ORIGINAL.\n"
    "\n"
    "Score 0 if ANY of the following holds:\n"
    "- The candidate asks for a different attribute of the same subject, e.g. "
    "its function, purpose, role, reason, cause, or effect instead of its "
    "identity, name, type, date, or amount.\n"
    "- The candidate changes the question type, e.g. an open 'what / who / "
    "when / how many / what age' question becomes a yes/no question.\n"
    "- Answering the candidate with the known answer would sound odd, "
    "incomplete, or off-topic.\n"
    "- The candidate is broader or narrower, so the known answer would no "
    "longer be the expected answer.\n"
    "- The candidate drops or changes a time qualifier, e.g. 'after the war "
    "started' becomes 'after the war', or 'during' becomes 'after'.\n"
    "Score 1 only if a person asked the candidate would expect exactly the "
    "known answer.\n"
    "\n"
    "Examples:\n"
    'ORIGINAL: "Who painted the Mona Lisa?"  ANSWER: "Leonardo da Vinci"\n'
    'CANDIDATE: "Which artist is responsible for the Mona Lisa?"\n'
    'Output: {"score": 1, "rationale": "Same request; the same artist answers it."}\n'
    "\n"
    'ORIGINAL: "What organ pumps blood through the body?"  ANSWER: "the heart"\n'
    'CANDIDATE: "What is the main function of the organ that pumps blood?"\n'
    'Output: {"score": 0, "rationale": "Asks for the function, not the organ; '
    '\'the heart\' does not answer it."}\n'
    "\n"
    'ORIGINAL: "What age do you need to be to vote in the US?"  ANSWER: "18"\n'
    'CANDIDATE: "Can a teenager vote in the US?"\n'
    'Output: {"score": 0, "rationale": "Turns an age question into a yes/no '
    'question; \'18\' is not an answer to it."}\n'
    "\n"
    'ORIGINAL: "What was the US foreign policy after WW2 began?"  ANSWER: "neutrality"\n'
    'CANDIDATE: "What policy guided the US after World War 2?"\n'
    'Output: {"score": 0, "rationale": "\'After World War 2\' is the post-war '
    'period, not the period after the war began."}\n'
    "\n"
    'Output format: {"score": 0 or 1, "rationale": "one short sentence"}\n'
)

# Structured variant: small models miss drift when asked for a verdict
# directly, but catch much more of it when they must first spell out what
# each question asks for. Examples are deliberately unrelated to the
# hand-labelled test data (tests/data/answer_preservation_labels.json).
STRUCTURED_SYSTEM = (
    "You check whether a CANDIDATE is a faithful rephrasing of an ORIGINAL "
    "question. You also get the ORIGINAL's known correct answer. Output ONLY a "
    "JSON object with these keys, in this order:\n"
    '  "original_asks": what the ORIGINAL asks for, as a short noun phrase that '
    "keeps EVERY detail narrowing it down (which entity, which attribute, time, "
    "place, scope),\n"
    '  "candidate_asks": the same for the CANDIDATE,\n'
    '  "differences": a list of the differences that could make a DIFFERENT '
    "answer correct: a detail present in one but not the other that changes "
    "which answer fits, or a different kind of answer asked for (person vs. "
    "price, name vs. reason, open question vs. yes/no); [] if none. Extra "
    "wording that the known answer still satisfies is not a difference,\n"
    '  "answer_fits": true only if the known answer is still the natural, '
    "complete answer to the CANDIDATE,\n"
    '  "score": 1 only if "differences" is empty AND "answer_fits" is true; otherwise 0.\n'
    "\n"
    "Watch for these changes, which all make score 0:\n"
    "- asking about a different attribute (who -> for how much, what -> why);\n"
    "- dropping or changing a detail that identifies the answer ('the tallest "
    "mountain' -> 'the most famous mountain');\n"
    "- replacing the key term with a description that is not exactly the same "
    "thing ('sofa' -> 'a piece of furniture with cushions');\n"
    "- dropping or changing a time qualifier ('after the war began' -> 'after the war');\n"
    "- mentioning the known answer inside the candidate.\n"
    "Pure rewording, synonyms, word order, voice, fill-in-the-blank vs. "
    "question form, or an added detail that is simply true of the known "
    "answer are fine.\n"
    "\n"
    "Example:\n"
    'ORIGINAL: "Who painted the Mona Lisa?"  ANSWER: "Leonardo da Vinci"\n'
    'CANDIDATE: "For how much was Leonardo da Vinci\'s Mona Lisa insured?"\n'
    '{"original_asks": "the painter of the Mona Lisa", "candidate_asks": "the '
    'insured value of the Mona Lisa", "differences": ["asks for a value, not a '
    'person", "mentions the answer"], "answer_fits": false, "score": 0}\n'
    "\n"
    'ORIGINAL: "What is the tallest mountain in Africa?"  ANSWER: "Kilimanjaro"\n'
    'CANDIDATE: "Which dormant African volcano rises higher than any other mountain there?"\n'
    '{"original_asks": "the tallest mountain in Africa", "candidate_asks": "the '
    'tallest mountain in Africa, described as a dormant volcano", "differences": [], '
    '"answer_fits": true, "score": 1}'
)

ANSWER_PRESERVATION_TEMPLATE = (
    'ORIGINAL: "{original}"\n'
    "ANSWER: {answers}\n"
    'CANDIDATE: "{candidate}"'
)


# Words too common to count as "the answer leaking into the paraphrase".
_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or", "by",
    "with", "from", "is", "was", "are", "were", "be", "its", "it", "as",
}


def leaks_answer(original: str, candidate: str, answers: List[str]) -> bool:
    """
    True if the candidate reveals a known answer, so it has become a
    different question ("For what price did Judas Iscariot ...?" for "Who
    sold out Jesus ...?"). Two cases, both ignoring words already in the
    original ("Is Greenland part of Europe or North America?" is fine):
      - every remaining word of an answer appears in the candidate, or
      - a capitalised name from an answer appears capitalised in the
        candidate ("... for Judas' actions ..."). Lower-case answer-type
        words like "article" in "In what article ...?" (gold "Article Two")
        do not count.
    """
    orig_words = set(normalise_text(original).split())
    cand_words = set(normalise_text(candidate).split())
    cand_names = {normalise_text(w) for w in candidate.split()[1:] if w[:1].isupper()}
    for answer in answers:
        new_words = [
            w for w in normalise_text(answer).split()
            if w not in _STOPWORDS and len(w) > 1 and w not in orig_words
        ]
        if new_words and all(w in cand_words for w in new_words):
            return True
        names = {normalise_text(w) for w in answer.split() if w[:1].isupper()}
        if any(n in cand_names and n not in orig_words for n in names):
            return True
    return False


def answer_preserved(
    original: str,
    candidate: str,
    answers: List[str],
    model_name: str = DEFAULT_MODEL,
    think: bool = False,
    structured: bool = True,
) -> int:
    """
    Return:
        1 -> the known answer(s) still correctly answer `candidate`
        0 -> otherwise (including unparseable judge output)

    `structured` selects the prompt that makes the model spell out what each
    question asks for before scoring (STRUCTURED_SYSTEM); False uses the
    original direct prompt (ANSWER_PRESERVATION_SYSTEM).
    """
    return answer_preservation_verdict(
        original, candidate, answers, model_name, think, structured
    )[0]


def answer_preservation_verdict(
    original: str,
    candidate: str,
    answers: List[str],
    model_name: str = DEFAULT_MODEL,
    think: bool = False,
    structured: bool = True,
) -> Tuple[int, str]:
    """Like answer_preserved(), but also returns the judge's short reason."""
    messages = [
        {"role": "system",
         "content": STRUCTURED_SYSTEM if structured else ANSWER_PRESERVATION_SYSTEM},
        {
            "role": "user",
            "content": ANSWER_PRESERVATION_TEMPLATE.format(
                original=original,
                answers=" / ".join(f'"{a}"' for a in answers),
                candidate=candidate,
            ),
        },
    ]
    try:
        raw = json.dumps(client.chat_json(
            model=model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=4096 if think else 768,
            think=think,
        ))
    except Exception as exc:
        print(f"[answer-judge-error] {exc}")
        return 0, "judge call failed"
    score = _extract_score(raw)
    if score is None:
        print(f"[answer-judge-parse-error] RAW_TAIL: {raw[-400:]}")
        return 0, "judge output unreadable"
    try:
        verdict = 1 if int(round(float(score))) >= 1 else 0
    except Exception:
        print(f"[answer-judge-bad-score] {score!r}. RAW_TAIL: {raw[-400:]}")
        return 0, "judge output unreadable"
    return verdict, _reason(json.loads(raw))


def _reason(parsed: object) -> str:
    """Short human-readable reason from either prompt's JSON output."""
    if not isinstance(parsed, dict):
        return ""
    diffs = parsed.get("differences")
    if isinstance(diffs, list) and diffs:
        return "; ".join(str(d) for d in diffs)
    if parsed.get("candidate_asks") and parsed.get("original_asks"):
        return f"asks for {parsed['candidate_asks']}, not {parsed['original_asks']}"
    return str(parsed.get("rationale") or "")
