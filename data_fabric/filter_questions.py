import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "models/Llama-3.1-8B-Instruct"
INPUT_PATH = "data_fabric/data/raw_data.jsonl"
OUTPUT_PATH = "data_fabric/data/filtered_data.jsonl"
# Number of the last question the run finished. Delete this file to start over from the first question.
PROGRESS_PATH = "data_fabric/data/filter_progress.txt"

BATCH_SIZE = 8  # Tune on the GPU you run on: too large a batch overflows GPU memory and slows everything down
LIMIT = None  # Set to a number to only process the first N questions

RULES = ["a", "b", "c", "d", "e"]

# nq_open already keeps answers of at most 5 tokens, so this only catches rare outliers
MAX_ANSWER_WORDS = 5

# Conservative patterns that are enough to fail a question without the LLM.
# A question that matches none of them still needs an LLM inspection.
FAST_FAILS = [
    ("c", r"^(list|names of|name all|name the (members|countries|states))\b"
          r"|\b(what are some|which of the following|all of the following)\b"),
    ("e", r"^(why|what happens)\b|\bdifference between\b"),
    ("d", r"\b(last time|most recent)\b"),
    ("d", r"\b(currently|right now|so far|to date|as of|latest|the current|this (year|season|week|month))\b"),
    ("d", r"\bwhen (is|are|does|do|will) the (next|new)\b|\bplay next\b"
          r"|\bthe next (season|episode|series|chapter|book|movie|film|album|game|match|olympics|world cup|election)\b"),
    ("d", r"\b(hold|holds|held|has|have) the (\w+ )?record\b|\brecord for (the )?most\b"),
]

CLASSIFIER_SYSTEM = (
    "You decide whether a trivia question fits a benchmark. A model will answer the question from memory "
    "and its answer will be graded as passed or failed. Output ONLY valid JSON.\n"
    "A question passes a rule when:\n"
    "a. No additional context is needed to answer it. It fails only if it refers to something that is not given, "
    "like \"the following\", \"this article\" or \"the image\". A question that needs obscure or specialist "
    "knowledge still passes.\n"
    "b. It is unambiguous: there is no room for opinion on whether an answer is right or wrong.\n"
    "c. Its answer is not a list. \"Who were the prime ministers of Israel?\" fails, because an answer "
    "could get some names right and some wrong.\n"
    "d. It is not time-sensitive. Time-sensitive means the correct answer depends on the date the question is "
    "asked. A question about something that already happened is NOT time-sensitive, even if it is recent or about "
    "news or product releases, because its answer can no longer change. These pass: \"When was the Immigration "
    "Reform and Control Act passed?\", \"Who hosted the Super Bowl in 2019?\", \"When did the iPod Touch 6th "
    "generation come out?\". Questions about the present or the future, or relative to now (current, latest, "
    "last time, next), fail: \"When do the Red Hot Chili Peppers tour?\", \"What channel is Cartoon Network on "
    "Spectrum?\", \"When are the FA Cup semi finals played?\", \"Is the weather hot?\".\n"
    "e. An answer can be judged reliably as correct or incorrect. Different spellings or phrasings of the same "
    "answer are fine: \"What's 1+1?\" passes, even though it can be answered \"two\" or \"2\". \"Who wrote Harry "
    "Potter?\" passes, even though it can be answered \"J.K. Rowling\", \"Joanne Rowling\" or \"Joanne Kathleen "
    "Rowling\". It fails only when different, non-equivalent answers could all be correct: \"Where is Harvard "
    "located?\" fails, because \"Cambridge\", \"Massachusetts\", \"USA\" or any combination of these could all be "
    "correct.\n"
    "Output format: {\"reason\": \"one short sentence\", \"a\": true or false, \"b\": true or false, "
    "\"c\": true or false, \"d\": true or false, \"e\": true or false}\n"
)

QUESTION_TEMPLATE = (
    'Question: "{question}"\n'
    'Gold answers: {answers}'
)


# load_model and _balanced_json_substring come from utils/llm_as_a_judge.py, which main no longer has
def load_model(model_path=MODEL_PATH):
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    return tok, model


def _balanced_json_substring(text):
    """Return the first balanced JSON object or array in text, or None."""
    opens = {"{": "}", "[": "]"}
    stack = []
    start = None
    for i, ch in enumerate(text):
        if ch in opens:
            if not stack:
                start = i
            stack.append(opens[ch])
        elif stack and ch == stack[-1]:
            stack.pop()
            if not stack and start is not None:
                return text[start:i + 1]
    return None


def fast_fail(row):
    """Return the rule a question clearly fails, or None if it needs an LLM inspection."""
    if min(len(answer.split()) for answer in row["answers"]) > MAX_ANSWER_WORDS:
        return "short"
    question = row["question"].lower()
    for rule, pattern in FAST_FAILS:
        if re.search(pattern, question):
            return rule
    return None


def parse_verdict(raw):
    """Return (passed, reason). Output that cannot be parsed counts as failed."""
    try:
        verdict = json.loads(_balanced_json_substring(raw) or raw)
    except json.JSONDecodeError:
        verdict = None
    if not isinstance(verdict, dict):
        return False, f"could not parse: {raw[-200:]}"

    passed = all(verdict.get(rule) is True for rule in RULES)
    return passed, verdict.get("reason", "")


def classify_batch(tok, model, rows):
    """Return a (passed, reason) pair for every row, using one generate call for the whole batch."""
    prompts = [
        tok.apply_chat_template([
            {"role": "system", "content": CLASSIFIER_SYSTEM},
            {"role": "user", "content": QUESTION_TEMPLATE.format(
                question=row["question"], answers=json.dumps(row["answers"], ensure_ascii=False)
            )},
        ], tokenize=False, add_generation_prompt=True)
        for row in rows
    ]
    # The chat template already starts with the BOS token
    inputs = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, do_sample=False, max_new_tokens=80, pad_token_id=tok.pad_token_id)

    generated = out[:, inputs["input_ids"].shape[1]:]
    return [parse_verdict(tok.decode(g, skip_special_tokens=True).strip()) for g in generated]


def main():
    with open(INPUT_PATH, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f][:LIMIT]

    done = int(Path(PROGRESS_PATH).read_text()) if Path(PROGRESS_PATH).exists() else 0
    if done:
        print(f"Continuing after question {done}/{len(rows)}")

    # Cheap filters first, so only the remaining questions reach the LLM
    to_llm = []
    for i, row in enumerate(rows, start=1):
        if i <= done:
            continue
        rule = fast_fail(row)
        if rule is None:
            to_llm.append((i, row))
        else:
            print(f"[{i}/{len(rows)}] FAIL (fast: {rule}) {row['question']}")

    tok, model = load_model()
    tok.padding_side = "left"  # Batched generation continues from the right end of every prompt
    offloaded = [name for name, device in getattr(model, "hf_device_map", {}).items() if device in ("cpu", "disk")]
    if offloaded:
        print(f"WARNING: {len(offloaded)} model parts don't fit on the GPU and run on the CPU, which is very slow")

    # A fresh run replaces the output, a continued run appends to it
    with open(OUTPUT_PATH, "a" if done else "w", encoding="utf-8") as out:
        try:
            for start in range(0, len(to_llm), BATCH_SIZE):
                batch = to_llm[start:start + BATCH_SIZE]
                began = time.time()
                verdicts = classify_batch(tok, model, [row for _, row in batch])

                for (i, row), (passed, reason) in zip(batch, verdicts):
                    print(f"[{i}/{len(rows)}] {'PASS' if passed else 'FAIL'} {row['question']} - {reason}")
                    if passed:
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                # Saved only after the batch's results are written, so a stopped run never skips questions
                done = batch[-1][0]
                Path(PROGRESS_PATH).write_text(str(done))

                print(f"{(time.time() - began) / len(batch):.2f}s per question "
                      f"({start + len(batch)}/{len(to_llm)} LLM questions done in this run)")
        except KeyboardInterrupt:
            print(f"\nStopped after question {done}/{len(rows)}. Run the script again to continue.")
            return

    Path(PROGRESS_PATH).write_text(str(len(rows)))
    print(f"Finished all {len(rows)} questions. Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
