"""
Model sanity test — Ollama version.

Asks the victim model a set of factual questions and writes the answers to a CSV.
Replaces the transformers/torch version with an Ollama HTTP API call.

Run from the project root:
    python3 tests/test_model_sanity.py
"""

import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ollama_client import OllamaClient
from victim import VictimModel

MODEL = os.environ.get("VICTIM_MODEL", "qwen3:4b")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OUTPUT_CSV = "llama_answers.csv"

QUESTIONS = [
    "Who wrote the lyrics to 'He Ain't Heavy, He's My Brother'?",
    "Who was the ruler of England in 1616?",
    "What is the capital of Australia?",
    "Who wrote the novel '1984'?",
    "What is the chemical symbol for gold?",
    "Who painted the Mona Lisa?",
    "What is the largest planet in the Solar System?",
    "In which year did World War II end?",
    "Who discovered penicillin?",
    "What is the capital of Japan?",
    "Who wrote 'Pride and Prejudice'?",
    "What is the smallest prime number?",
    "Which element has atomic number 6?",
    "Who was the first person to walk on the Moon?",
    "What is the longest river in South America?",
    "Who composed 'The Four Seasons'?",
    "What is the capital of Canada?",
    "Which ancient civilization built Machu Picchu?",
    "Who wrote 'The Old Man and the Sea'?",
    "What is the square root of 144?",
]


def main() -> None:
    client = OllamaClient(base_url=OLLAMA_BASE_URL)
    client.require_available()

    victim = VictimModel(client=client, model=MODEL, temperature=0.0)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["question_id", "question", "answer"])

        for i, question in enumerate(QUESTIONS, start=1):
            print(f"[{i}/{len(QUESTIONS)}] {question}")
            answer = victim.answer(question)
            print(f"  → {answer}\n")
            writer.writerow([i, question, answer])
            csvfile.flush()

    print(f"Results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
