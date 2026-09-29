import json
import os

from datasets import load_dataset

# nq_open is Google's Natural Questions, already filtered on the dataset side to questions
# with a short answer (at most 5 tokens). No filtering happens here.
DATASET = "google-research-datasets/nq_open"
SPLITS = ["train", "validation"]
OUTPUT_PATH = "data_fabric/data/raw_data.jsonl"


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    count = 0
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for split in SPLITS:
            for i, example in enumerate(load_dataset(DATASET, split=split)):
                row = {
                    # nq_open has no ID column, so the split and row index identify a question
                    "id": f"{split}_{i}",
                    "question": example["question"],
                    "answers": example["answer"],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1

    print(f"Saved {count} questions to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
