"""
Build a question file from the NQ-Open dataset (Natural Questions, open-domain).

NQ-Open questions are real Google search queries with short gold answers,
so they are much less "textbook" than data/sample_questions.json and the
victim is less certain about them, which is where framing effects show up.

Many NQ-Open questions are answered wrongly by a small victim model, or its
answer doesn't match the narrow gold answers; the experiment skips those, since
an attack needs a correct baseline. `--victim-model` pre-screens the sample and
keeps only questions that victim answers correctly (judged by the LLM judge),
so every question in the file is a usable attack target.

The output uses the same JSON schema as data/sample_questions.json and can be
passed straight to the experiment:

    python3 utils/download_nq_open.py --n 300
    python3 utils/download_nq_open.py --n 50 --victim-model qwen3:4b
    python3 run_experiment.py --dataset-path data/nq_open_qwen3_4b_correct.json --n-questions 10

Run from the project root. Requires the `datasets` package and internet access.
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluators import LLMJudgeEvaluator
from ollama_client import OllamaClient
from victim import VictimModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=300,
                        help="Number of questions to keep")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--victim-model", default=None,
                        help="Keep only questions this Ollama model answers correctly")
    parser.add_argument("--judge-model", default="qwen3:4b",
                        help="LLM judge used for the --victim-model check")
    parser.add_argument("--output", default=None,
                        help="Default: data/nq_open_sample.json, or "
                             "data/nq_open_<victim>_correct.json with --victim-model")
    args = parser.parse_args()

    victim = judge = None
    if args.victim_model:
        client = OllamaClient()
        client.require_available()
        victim = VictimModel(client=client, model=args.victim_model)
        judge = LLMJudgeEvaluator(client=client, model=args.judge_model)

    ds = load_dataset("google-research-datasets/nq_open", split=args.split)
    # Shuffle all indices so that, when filtering, we keep scanning until
    # `n` questions pass.
    indices = list(range(len(ds)))
    random.Random(args.seed).shuffle(indices)

    entries = []
    scanned = 0
    for idx in indices:
        if len(entries) >= args.n:
            break
        row = ds[idx]
        question = row["question"].strip()
        # NQ-Open questions are lower-case search queries without a "?"
        question = question[0].upper() + question[1:]
        if not question.endswith("?"):
            question += "?"

        if victim is not None:
            scanned += 1
            answer = victim.answer(question)
            ok = judge.evaluate(answer, row["answer"]).correct
            print(f"[{len(entries)}/{args.n} kept, {scanned} scanned] "
                  f"{'KEEP' if ok else 'drop'}  {question}  →  {answer!r}")
            if not ok:
                continue

        entries.append({
            "id": f"nqo_{args.split}_{idx:04d}",
            "question": question,
            "answers": row["answer"],
        })

    entries.sort(key=lambda e: e["id"])
    if args.output:
        out = Path(args.output)
    elif args.victim_model:
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", args.victim_model).strip("_")
        out = PROJECT_ROOT / "data" / f"nq_open_{slug}_correct.json"
    else:
        out = PROJECT_ROOT / "data" / "nq_open_sample.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2, ensure_ascii=False)
    print(f"Saved {len(entries)} questions → {out}")


if __name__ == "__main__":
    main()
