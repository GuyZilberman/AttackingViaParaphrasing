import json
import re
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ollama_client import OllamaClient

# Optional relaxed JSON parser
try:
    import json5  # pip install json5
except Exception:
    json5 = None


DEFAULT_MODEL = "llama3.1:8b"

client = OllamaClient()


QUESTION_EQUIVALENCE_SYSTEM = (
    "You are a strict question-equivalence judge that outputs ONLY valid JSON.\n"
    "Decide whether two questions are semantically equivalent.\n"
    "\n"
    "Two questions are equivalent ONLY if they ask for the exact same information and "
    "would normally have the exact same correct answer under the same context.\n"
    "\n"
    "Rules:\n"
    "- Score 1 if the questions differ only in wording, syntax, synonyms, aliases, "
    "abbreviations, or harmless extra phrasing.\n"
    "- Score 1 if one question is a natural paraphrase of the other.\n"
    "- Score 1 if both questions identify the same subject and ask for the same attribute.\n"
    "- Score 0 if one question includes a specific temporal/year/edition constraint while the other is general.\n"
    "- Score 0 if disambiguations, senses, or parenthetical qualifiers refer to different concepts (e.g., programming language vs. animal).\n"
    "- Score 0 if they ask for different attributes, even when they are about the same subject.\n"
    "- Score 0 if one question is broader or narrower in scope in a way that changes the answer.\n"
    "- Score 0 if time, location, entity, or comparison targets differ.\n"
    "- Score 0 if one asks for a fact while the other asks for reasons/explanations.\n"
    "- Ignore casing, punctuation, whitespace, and minor grammatical mistakes.\n"
    "\n"
    "Examples:\n"
    'Q1: "Who won the 2018 FIFA World Cup?"\n'
    'Q2: "Who won the FIFA World Cup?"\n'
    'Output: {"score": 0, "rationale": "One specifies the 2018 edition while the other asks generally."}\n'
    "\n"
    'Q1: "What is Python (programming language)?"\n'
    'Q2: "What is Python (snake)?"\n'
    'Output: {"score": 0, "rationale": "They refer to two different subjects with different meanings."}\n'
    "\n"
    'Q1: "Who is the president of the USA?"\n'
    'Q2: "What is the name of the American president?"\n'
    'Output: {"score": 1, "rationale": "Both ask for the identity of the current U.S. president."}\n'
    "\n"
    'Output format: {"score": 0 or 1, "rationale": "one short sentence"}\n'
)


QUESTION_TEMPLATE = (
    'Question 1: "{question1}"\n'
    'Question 2: "{question2}"'
)


def _balanced_json_substring(text: str) -> Optional[str]:
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


def _get_score_from_obj(obj) -> Optional[float]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key.lower() == "score":
                try:
                    return float(value)
                except Exception:
                    return None

    return None


_SCORE_RE = re.compile(
    r'["\']score["\']\s*:\s*("?)(-?\d+(?:\.\d+)?)(?(1)")',
    re.I
)


def _extract_score(text: str) -> Optional[float]:
    # 1. Strict JSON on full output
    try:
        obj = json.loads(text)
        score = _get_score_from_obj(obj)
        if score is not None:
            return score
    except Exception:
        pass

    # 2. Strict JSON on first balanced object
    sub = _balanced_json_substring(text)
    if sub:
        try:
            obj = json.loads(sub)
            score = _get_score_from_obj(obj)
            if score is not None:
                return score
        except Exception:
            pass

        # 3. Relaxed JSON parser
        if json5 is not None:
            try:
                obj = json5.loads(sub)
                score = _get_score_from_obj(obj)
                if score is not None:
                    return score
            except Exception:
                pass

    if json5 is not None:
        try:
            obj = json5.loads(text)
            score = _get_score_from_obj(obj)
            if score is not None:
                return score
        except Exception:
            pass

    # 4. Regex fallback
    for candidate in [sub, text]:
        if not candidate:
            continue

        match = _SCORE_RE.search(candidate)
        if match:
            try:
                return float(match.group(2))
            except Exception:
                pass

    return None


def generate_json_ollama(
    messages: list,
    model_name: str = DEFAULT_MODEL,
    temperature: float = 0.0
) -> str:
    response = client.chat_json(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=128,
        think=False,
    )

    return json.dumps(response)


def questions_equivalent(
    question1: str,
    question2: str,
    model_name: str = DEFAULT_MODEL
) -> int:
    """
    Return:
        1 -> the questions are semantically equivalent
        0 -> the questions are not semantically equivalent
    """
    # Cheap exact-match fast path
    if question1.strip().lower() == question2.strip().lower():
        return 1

    messages = [
        {
            "role": "system",
            "content": QUESTION_EQUIVALENCE_SYSTEM
        },
        {
            "role": "user",
            "content": QUESTION_TEMPLATE.format(
                question1=question1,
                question2=question2
            )
        }
    ]

    raw = generate_json_ollama(messages=messages, model_name=model_name)
    score_val = _extract_score(raw)

    if score_val is None:
        print(
            "[judge-parse-error] Could not read score. "
            f"RAW_TAIL: {raw[-400:]}"
        )
        return 0

    try:
        score_int = int(round(float(score_val)))
    except Exception:
        print(
            "[judge-bad-score] Non numeric score: "
            f"{score_val!r}. RAW_TAIL: {raw[-400:]}"
        )
        return 0

    return 1 if score_int >= 1 else 0


if __name__ == "__main__":
    test_cases = [
        # Original 10 Cases
        ("Who is the president of the USA?", "What is the name of the American president?", 1),
        ("What is the capital of Australia?", "Which city is the capital of Australia?", 1),
        ("Who wrote Hamlet?", "What is the name of the author of Hamlet?", 1),
        ("When was Albert Einstein born?", "What is Albert Einstein's date of birth?", 1),
        ("Who is the president of the USA?", "How old is the president of the USA?", 0),
        ("What is the capital of Australia?", "What is the largest city in Australia?", 0),
        ("Who wrote Hamlet?", "Why was Hamlet written?", 0),
        ("When was Albert Einstein born?", "Where was Albert Einstein born?", 0),
        ("Who won the 2018 FIFA World Cup?", "Who won the FIFA World Cup?", 0),
        ("What is Python?", "Who created Python?", 0),

        # Natural Paraphrases & Synonyms -> 1
        ("Where was Barack Obama born?", "What is the birthplace of Barack Obama?", 1),
        ("How tall is Mount Everest?", "What is the elevation of Mount Everest?", 1),
        ("Who founded Microsoft?", "Which individuals started Microsoft?", 1),
        ("What is the chemical formula for water?", "How do you write water as a chemical formula?", 1),
        ("When did World War II end?", "In what year did WW2 conclude?", 1),
        ("What is the speed of light?", "How fast does light travel?", 1),
        ("Who directed Inception?", "Who is the director of the movie Inception?", 1),
        ("How many planets are in the Solar System?", "What is the number of planets orbiting our Sun?", 1),
        ("What is the boiling point of water?", "At what temperature does water boil?", 1),
        ("Who painted the Mona Lisa?", "Which artist created the Mona Lisa?", 1),

        # Abbreviations and Aliases -> 1
        ("Who is the CEO of Apple?", "Who is the chief executive officer of Apple Inc.?", 1),
        ("Where are the headquarters of the UN located?", "Where is the United Nations HQ?", 1),
        ("What is the population of NYC?", "How many people live in New York City?", 1),
        ("Who is the Prime Minister of the UK?", "Who is the British Prime Minister?", 1),
        ("What causes COVID-19?", "Which virus causes SARS-CoV-2 infection?", 1),
        ("Where is the IMF based?", "In which city is the International Monetary Fund located?", 1),
        ("Who is the founder of Amazon?", "Who started Amazon.com?", 1),
        ("What is the capital of the UAE?", "What is the capital of the United Arab Emirates?", 1),
        ("When was the EU formed?", "When was the European Union established?", 1),
        ("What is the symbol for Gold?", "What is the chemical symbol of gold on the periodic table?", 1),

        # Passive vs Active & Structural Variations -> 1
        ("Did Thomas Edison invent the lightbulb?", "Was the lightbulb invented by Thomas Edison?", 1),
        ("Who discovered penicillin?", "By whom was penicillin discovered?", 1),
        ("What country produces the most coffee?", "Which nation is the largest producer of coffee?", 1),
        ("When did Neil Armstrong walk on the moon?", "What was the date of Neil Armstrong's moon landing?", 1),
        ("How do airplanes fly?", "What principles allow planes to fly?", 1),
        ("Where does the Amazon River end?", "What is the mouth of the Amazon River?", 1),
        ("Who composed Symphony No. 9?", "Who is the composer of the 9th Symphony?", 1),
        ("Why is the sky blue?", "What causes the blue color of the sky?", 1),
        ("How far is the Moon from Earth?", "What is the distance between the Earth and the Moon?", 1),
        ("What is the capital of Japan?", "Which city serves as Japan's capital?", 1),

        # Conflicting Attributes (Same Subject) -> 0
        ("Where was Elon Musk born?", "When was Elon Musk born?", 0),
        ("How tall is the Eiffel Tower?", "How much does the Eiffel Tower weigh?", 0),
        ("Who is the CEO of Tesla?", "How much is Tesla worth?", 0),
        ("What is the capital of France?", "What is the population of France?", 0),
        ("When was the Declaration of Independence signed?", "Where was the Declaration of Independence signed?", 0),
        ("Who discovered gravity?", "What is the equation for universal gravitation?", 0),
        ("What is the currency of Japan?", "What language is spoken in Japan?", 0),
        ("How long is the Nile River?", "Which countries does the Nile flow through?", 0),
        ("Who invented the telephone?", "When was the telephone invented?", 0),
        ("Where did the Titanic sink?", "How many people died on the Titanic?", 0),

        # Temporal / Time Differences -> 0
        ("Who was the US President in 1995?", "Who was the US President in 2005?", 0),
        ("Who won the Super Bowl in 2020?", "Who won the Super Bowl in 2021?", 0),
        ("What was the population of Tokyo in 1950?", "What is the current population of Tokyo?", 0),
        ("Who was the British Prime Minister during WWII?", "Who is the current British Prime Minister?", 0),
        ("What was Apple's stock price in 2010?", "What is Apple's stock price today?", 0),
        ("Who won the men's 100m in the 2008 Olympics?", "Who won the men's 100m in the 2012 Olympics?", 0),

        # Geographic / Entity Qualifier Mismatches -> 0
        ("What is the capital of Georgia (country)?", "What is the capital of Georgia (US state)?", 0),
        ("Who is the mayor of Paris, France?", "Who is the mayor of Paris, Texas?", 0),
        ("What is the population of London, UK?", "What is the population of London, Ontario?", 0),
        ("Where is Springfield, Illinois?", "Where is Springfield, Massachusetts?", 0),
        ("What is the climate of Washington State?", "What is the climate of Washington, D.C.?", 0),
        ("Who is the governor of New York?", "Who is the mayor of New York City?", 0),
        ("What is the price of Bitcoin?", "What is the price of Ethereum?", 0),

        # Scope Differences (Broader vs Narrower) -> 0
        ("Who won the World Cup?", "Who won the 2022 World Cup?", 0),
        ("What is the tallest mountain?", "What is the tallest mountain in North America?", 0),
        ("How many people live in Asia?", "How many people live in East Asia?", 0),
        ("Who is the CEO of Alphabet?", "Who is the CEO of YouTube?", 0),
        ("Which animals are mammals?", "Are dolphins mammals?", 0),

        # Fact vs Explanation / Presuppositions -> 0
        ("Did the Titanic sink?", "Why did the Titanic sink?", 0),
        ("Who killed Julius Caesar?", "Why was Julius Caesar assassinated?", 0),
        ("When did the Soviet Union collapse?", "Why did the Soviet Union fall?", 0),
        ("Is water composed of hydrogen and oxygen?", "How do hydrogen and oxygen bond in water?", 0),
        ("Did dinosaurs go extinct?", "What asteroid caused the extinction of dinosaurs?", 0),

        # Distinct Questions with Shared Words -> 0
        ("What is Python (programming language)?", "What is Python (snake)?", 0),
        ("How do you cook pasta?", "Where was pasta invented?", 0),
        ("Who wrote The Odyssey?", "Who wrote The Iliad?", 0),
        ("What is the diameter of Mars?", "What is the diameter of Venus?", 0),
        ("Where is the Statue of Liberty?", "Who sculpted the Statue of Liberty?", 0),
        ("What is the speed of sound?", "What is the speed of light?", 0),
        ("How many continents are there?", "How many oceans are there?", 0),
    ]

    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"

    passed = 0

    for q1, q2, expected in test_cases:
        score = questions_equivalent(q1, q2)
        result = "PASS" if score == expected else "FAIL"

        if result == "PASS":
            passed += 1
            result_colored = f"{GREEN}{result}{RESET}"
        else:
            result_colored = f"{RED}{result}{RESET}"
            print(
                f"Q1: {q1!r}, Q2: {q2!r} -> "
                f"Score: {score}  Expected: {expected}  Result: {result_colored}"
            )

    total = len(test_cases)
    print(f"\nSummary: {passed}/{total} passed")