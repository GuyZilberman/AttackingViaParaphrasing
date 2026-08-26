"""
Question dataset loading and sampling.

Supports:
  - Built-in JSON  (data/sample_questions.json)  — default
  - Custom JSON file with the same schema: list of {question, answers, id?}
  - CSV file with columns: question, answer  (answer may be semicolon-separated)
"""

import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

# Each entry the rest of the code works with
QuestionEntry = Dict  # {id, question, answers: List[str]}

_BUILTIN_PATH = Path(__file__).parent / "data" / "sample_questions.json"


def load_questions(
    path: Optional[str] = None,
    n: Optional[int] = None,
    seed: int = 42,
) -> List[QuestionEntry]:
    """
    Load questions from `path` (or the built-in dataset if None),
    optionally sampling `n` items.

    Args:
        path:  File path to a JSON or CSV question dataset.
               None → use the built-in data/sample_questions.json.
        n:     How many questions to return.  None → all.
        seed:  Random seed used when sampling.

    Returns:
        List of dicts with keys: id, question, answers (list of strings).
    """
    if path is None:
        entries = _load_json(_BUILTIN_PATH)
    else:
        p = Path(path)
        if p.suffix.lower() == ".csv":
            entries = _load_csv(p)
        else:
            entries = _load_json(p)

    if not entries:
        raise ValueError(f"No questions found in dataset (path={path!r}).")

    if n is not None and n < len(entries):
        rng = random.Random(seed)
        entries = rng.sample(entries, n)

    return entries


# ---------------------------------------------------------------------------
# Format loaders
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> List[QuestionEntry]:
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)

    entries = []
    for i, item in enumerate(raw):
        question = item.get("question") or item.get("query") or ""
        answers = item.get("answers") or item.get("answer") or []
        if isinstance(answers, str):
            # Possibly semicolon-separated
            answers = [a.strip() for a in answers.split(";") if a.strip()]
        entry_id = item.get("id") or item.get("question_id") or f"q{i+1:04d}"
        if question:
            entries.append({"id": str(entry_id), "question": question, "answers": answers})
    return entries


def _load_csv(path: Path) -> List[QuestionEntry]:
    entries = []
    with open(path, encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            question = (
                row.get("question") or row.get("query") or ""
            ).strip()
            answer_raw = (
                row.get("answer") or row.get("answers") or ""
            ).strip()
            answers = [a.strip() for a in answer_raw.split(";") if a.strip()]
            qid = (
                row.get("id") or row.get("question_id") or f"q{i+1:04d}"
            )
            if question:
                entries.append(
                    {"id": str(qid), "question": question, "answers": answers}
                )
    return entries
