import csv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "models/Llama-3.1-8B-Instruct"
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


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    return tokenizer, model


def ask_question(tokenizer, model, question):
    messages = [
        {
            "role": "system",
            "content": (
                "Answer the question with only the shortest direct answer. "
                "Do not explain. Do not repeat the question. "
                "Do not use a full sentence unless necessary. "
                "Return only the answer itself."
            )
        },
        {
            "role": "user",
            "content": question
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    return response


def main():
    tokenizer, model = load_model()

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow([
            "question_id",
            "question",
            "answer"
        ])

        for i, question in enumerate(QUESTIONS, start=1):
            print(f"[{i}/{len(QUESTIONS)}] {question}")

            answer = ask_question(
                tokenizer,
                model,
                question
            )

            print(f"Answer: {answer}\n")

            writer.writerow([
                i,
                question,
                answer
            ])

            csvfile.flush()

    print(f"Results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()