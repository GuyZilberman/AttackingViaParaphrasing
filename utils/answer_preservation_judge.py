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
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.question_equivalence_judge import _extract_score, client

# Scored with tests/eval_judges.py on tests/data/answer_preservation_labels.json
# (46 paraphrases, 14 drifted):
#   gemma3:12b        → 2 drifted accepted,  4 valid rejected
#   mistral-nemo:12b  → 2 drifted accepted,  6 valid rejected
#   qwen3:4b          → 1 drifted accepted, 12 valid rejected
#   llama3.1:8b       → accepted most drifts on an earlier set
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

ANSWER_PRESERVATION_TEMPLATE = (
    'ORIGINAL: "{original}"\n'
    "ANSWER: {answers}\n"
    'CANDIDATE: "{candidate}"'
)


def answer_preserved(
    original: str,
    candidate: str,
    answers: List[str],
    model_name: str = DEFAULT_MODEL,
    think: bool = False,
) -> int:
    """
    Return:
        1 -> the known answer(s) still correctly answer `candidate`
        0 -> otherwise (including unparseable judge output)
    """
    messages = [
        {"role": "system", "content": ANSWER_PRESERVATION_SYSTEM},
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
            max_tokens=4096 if think else 512,
            think=think,
        ))
    except Exception as exc:
        print(f"[answer-judge-error] {exc}")
        return 0
    score = _extract_score(raw)
    if score is None:
        print(f"[answer-judge-parse-error] RAW_TAIL: {raw[-400:]}")
        return 0
    try:
        return 1 if int(round(float(score))) >= 1 else 0
    except Exception:
        print(f"[answer-judge-bad-score] {score!r}. RAW_TAIL: {raw[-400:]}")
        return 0
