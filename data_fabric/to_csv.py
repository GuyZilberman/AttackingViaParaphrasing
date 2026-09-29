import csv
import json

INPUT_PATH = "data_fabric/data/filtered_data.jsonl"
OUTPUT_CSV = "data_fabric/data/filtered_data.csv"


def main():
    with open(INPUT_PATH, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow([
            "question_id",
            "question",
            "gold_answers"
        ])

        for row in rows:
            writer.writerow([
                row["id"],
                row["question"],
                # JSON list, so answers that contain commas stay intact
                json.dumps(row["answers"], ensure_ascii=False)
            ])

    print(f"Saved {len(rows)} questions to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
