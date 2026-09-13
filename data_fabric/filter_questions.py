import json
import sys
from pathlib import Path

# Allow running as "python data_fabric/filter_questions.py" from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.llm_as_a_judge import _balanced_json_substring, generate_json, load_model

INPUT_PATH = "data_fabric/data/raw_data.jsonl"
OUTPUT_PATH = "data_fabric/data/filtered_data.jsonl"

RULES = ["a", "b", "c", "d", "e", "f"]

CLASSIFIER_SYSTEM = (
    "You decide whether a trivia question fits a benchmark. A model will answer the question from memory "
    "and its answer will be graded as passed or failed. Output ONLY valid JSON.\n"
    "A question passes a rule when:\n"
    "a. No additional context is needed to answer it (no need to read Wikipedia, for example).\n"
    "b. It is unambiguous: there is no room for opinion on whether an answer is right or wrong.\n"
    "c. Its answer is short.\n"
    "d. Its answer is not a list. \"Who were the prime ministers of Israel?\" fails, because an answer "
    "could get some names right and some wrong.\n"
    "e. It is not time-sensitive: its answer could not have changed depending on the date of asking. "
    "\"What were the results of the last soccer game?\" fails. Be conservative: if the answer might have "
    "changed, it fails.\n"
    "f. An answer can be judged reliably as correct or incorrect. \"What's 1+1?\" passes, even though it can "
    "be answered \"two\" or \"2\". \"Who wrote Harry Potter?\" passes, even though it can be answered "
    "\"J.K. Rowling\", \"Joanne Rowling\" or \"Joanne Kathleen Rowling\". \"Where is Harvard located?\" fails, "
    "because \"Cambridge\", \"Massachusetts\", \"USA\" or any combination of these could all be correct.\n"
    "Output format: {\"reason\": \"one short sentence\", \"a\": true or false, \"b\": true or false, "
    "\"c\": true or false, \"d\": true or false, \"e\": true or false, \"f\": true or false}\n"
)

QUESTION_TEMPLATE = (
    'Question: "{question}"\n'
    'Gold answers: {answers}'
)


def classify(tok, model, question, answers):
    """Return (passed, reason). Output that cannot be parsed counts as failed."""
    messages = [
        {"role": "system", "content": CLASSIFIER_SYSTEM},
        {"role": "user", "content": QUESTION_TEMPLATE.format(
            question=question, answers=json.dumps(answers, ensure_ascii=False)
        )},
    ]

    raw = generate_json(tok, model, messages)

    try:
        verdict = json.loads(_balanced_json_substring(raw) or raw)
    except json.JSONDecodeError:
        verdict = None
    if not isinstance(verdict, dict):
        return False, f"could not parse: {raw[-200:]}"

    passed = all(verdict.get(rule) is True for rule in RULES)
    return passed, verdict.get("reason", "")


def main():
    tok, model = load_model()

    with open(INPUT_PATH, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    kept = 0
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for i, row in enumerate(rows, start=1):
            passed, reason = classify(tok, model, row["question"], row["answers"])
            print(f"[{i}/{len(rows)}] {'PASS' if passed else 'FAIL'} {row['question']} - {reason}")

            if passed:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                kept += 1

    print(f"Kept {kept}/{len(rows)} questions. Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
