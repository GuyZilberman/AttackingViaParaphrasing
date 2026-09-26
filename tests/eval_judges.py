"""
Score candidate judge models against hand-labelled data.

Two judges decide whether an attack "succeeded", so their error rates bound
how much any reported number can be trusted:

  answer judge          Is the victim's answer correct?   (evaluators/llm_judge.py)
                        data: tests/data/answer_judge_labels.json
  answer preservation   Does the gold answer still answer the paraphrase?
                        (utils/answer_preservation_judge.py)
                        data: tests/data/answer_preservation_labels.json

For each model it prints how many wrong items were accepted (false accepts:
these create fake attack successes or let drifted paraphrases through) and
how many right items were rejected (false rejects), listing every mistake.

Run from the project root, e.g.:
    python3 tests/eval_judges.py --answer-judge gemma3:12b qwen3:4b
    python3 tests/eval_judges.py --preservation-judge qwen3:4b gemma3:12b
    python3 tests/eval_judges.py --preservation-judge gemma3:12b --prompt direct --leak-check

The labels were written by hand from real pipeline outputs; they are small
(tens of items), so differences of one or two errors are not conclusive.
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluators import LLMJudgeEvaluator
from ollama_client import OllamaClient
from utils.answer_preservation_judge import answer_preserved, leaks_answer

DATA = Path(__file__).parent / "data"


def eval_answer_judge(client: OllamaClient, model: str) -> None:
    items = json.load(open(DATA / "answer_judge_labels.json", encoding="utf-8"))
    judge = LLMJudgeEvaluator(client=client, model=model)
    _report(
        f"answer judge {model}", items,
        lambda it: judge.evaluate(it["prediction"], it["gold"], question=it["question"]).correct,
        label=lambda it: it["correct"],
        show=lambda it: f"{it['prediction']!r} vs gold {it['gold'][0]!r}  ({it['question']})",
    )


def eval_preservation_judge(model: str, prompt: str, leak_check: bool) -> None:
    items = json.load(open(DATA / "answer_preservation_labels.json", encoding="utf-8"))

    def predict(it) -> bool:
        if leak_check and leaks_answer(it["original"], it["candidate"], it["gold"]):
            return False
        return bool(answer_preserved(it["original"], it["candidate"], it["gold"],
                                     model_name=model, structured=prompt == "structured"))

    _report(
        f"answer preservation {model} ({prompt} prompt{', + leak check' if leak_check else ''})",
        items, predict,
        label=lambda it: it["preserved"],
        show=lambda it: f"{it['candidate']!r}  (original: {it['original']})",
    )


def _report(title, items, predict, label, show) -> None:
    false_accepts, false_rejects = [], []
    start = time.time()
    for it in items:
        verdict, truth = predict(it), label(it)
        if verdict and not truth:
            false_accepts.append(it)
        elif truth and not verdict:
            false_rejects.append(it)
    n_bad = sum(1 for it in items if not label(it))
    print(f"== {title}: accepted {len(false_accepts)}/{n_bad} that should be rejected, "
          f"rejected {len(false_rejects)}/{len(items) - n_bad} that should be accepted "
          f"({(time.time() - start) / len(items):.1f}s/item)")
    for it in false_accepts:
        print(f"     FALSE ACCEPT: {show(it)}")
    for it in false_rejects:
        print(f"     FALSE REJECT: {show(it)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--answer-judge", nargs="*", default=[], metavar="MODEL")
    parser.add_argument("--preservation-judge", nargs="*", default=[], metavar="MODEL")
    parser.add_argument("--prompt", choices=["structured", "direct"], default="structured",
                        help="answer-preservation prompt variant")
    parser.add_argument("--leak-check", action="store_true",
                        help="also reject paraphrases that contain the answer (as the pipeline does)")
    args = parser.parse_args()
    if not args.answer_judge and not args.preservation_judge:
        parser.error("give --answer-judge and/or --preservation-judge models")

    client = OllamaClient()
    client.require_available()
    for model in args.answer_judge:
        eval_answer_judge(client, model)
    for model in args.preservation_judge:
        eval_preservation_judge(model, args.prompt, args.leak_check)


if __name__ == "__main__":
    main()
