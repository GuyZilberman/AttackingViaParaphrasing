import csv
from datasets import load_dataset

dataset = load_dataset(
    "sentence-transformers/natural-questions",
    split="train",
    streaming=True
)

rows = []

for i, example in enumerate(dataset):
    rows.append({
        "question": example["query"],
        "answer": example["answer"]
    })

    if i == 9:
        break

with open("natural_questions_10.csv", "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["question", "answer"])
    writer.writeheader()
    writer.writerows(rows)

print("Saved to natural_questions_10.csv")