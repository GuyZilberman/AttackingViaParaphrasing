import json
import re
import unicodedata
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Optional relaxed JSON parser - tolerates minor issues like trailing commas or single quotes
try:
    import json5  # pip install json5
except Exception:
    json5 = None

MODEL_PATH = "models/Llama-3.1-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global tokenizer and model so best_llm_judge can match best_subspan_em signature
_TOK = None
_MODEL = None

# Keep your original system prompt to avoid behavior drift
BINARY_JUDGE_SYSTEM = (
    "You are a strict QA judge that outputs ONLY valid JSON.\n"
    "Decide if a prediction is semantically equivalent to a ground truth.\n"
    "Rules:\n"
    "- Score 1 if meanings match even if there are extra descriptors, aliases, reordered words, or containment.\n"
    "- Score 1 for abbreviations or nicknames matching the same entity.\n"
    "- Score 0 if they refer to different entities or contradict.\n"
    "- If both mentions include geographic or entity qualifiers that differ, score 0.\n"
    "- Ignore casing, punctuation, whitespace, and articles.\n"
    "Output format: {\"score\": 0 or 1, \"rationale\": \"one short sentence\"}\n"
)

JUDGE_TEMPLATE = (
    'Prediction: "{prediction}"\n'
    'Ground truth: "{ground_truth}"'
)

def load_model(model_path: str = MODEL_PATH):
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    return tok, model

def set_model(tok, model) -> None:
    """Set globals so callers do not need to pass tok and model."""
    global _TOK, _MODEL
    _TOK, _MODEL = tok, model

def get_model() -> Tuple:
    """Return global tok and model, loading if needed."""
    global _TOK, _MODEL
    if _TOK is None or _MODEL is None:
        _TOK, _MODEL = load_model()
    return _TOK, _MODEL

def _build_inputs(tok, model, messages):
    prompt_str = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return tok(prompt_str, return_tensors="pt", padding=True).to(model.device)

def generate_json(tok, model, messages, max_new_tokens: int = 128) -> str:
    inputs = _build_inputs(tok, model, messages)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            do_sample=False,          # greedy
            top_p=1.0,               # inert when do_sample=False
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id
        )
    generated = out[0, inputs["input_ids"].shape[1]:]
    text = tok.decode(generated, skip_special_tokens=True).strip()
    return text

def _balanced_json_substring(text: str) -> Optional[str]:
    """
    Return the first balanced JSON object or array substring if present.
    Looks for either {...} or [...] using a simple stack scan.
    """
    opens = {'{': '}', '[': ']'}
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
                return text[start:i+1]
    return None

def _normalize(s: str) -> str:
    # Normalize unicode and strip accents so Munchen vs München matches
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return "".join(ch.lower() for ch in s if ch.isalnum() or ch.isspace()).strip()

def _get_score_from_obj(obj) -> Optional[float]:
    if isinstance(obj, dict):
        # try score or case variants
        for k in obj.keys():
            if k.lower() == "score":
                try:
                    return float(obj[k])
                except Exception:
                    return None
    return None

# Regex to pull score without needing valid JSON
# Example matches: "score": 1, 'score': 0, "score": 0.0, "score" : "1"
_SCORE_RE = re.compile(r'["\']score["\']\s*:\s*("?)(-?\d+(?:\.\d+)?)(?(1)")', re.I)

def _extract_score(text: str) -> Optional[float]:
    """
    Robust score extraction:
    1) Try strict JSON on full text.
    2) Try strict JSON on balanced JSON substring.
    3) Try json5 on substring then full text.
    4) Regex fallback to read the numeric score.
    Returns a float if found, else None.
    """
    # 1) Strict JSON full
    try:
        obj = json.loads(text)
        s = _get_score_from_obj(obj)
        if s is not None:
            return s
    except Exception:
        pass

    # 2) Balanced substring
    sub = _balanced_json_substring(text)
    if sub:
        try:
            obj = json.loads(sub)
            s = _get_score_from_obj(obj)
            if s is not None:
                return s
        except Exception:
            pass
        # 3) json5 on substring
        if json5 is not None:
            try:
                obj = json5.loads(sub)
                s = _get_score_from_obj(obj)
                if s is not None:
                    return s
            except Exception:
                pass

    # 3b) json5 on full text
    if json5 is not None:
        try:
            obj = json5.loads(text)
            s = _get_score_from_obj(obj)
            if s is not None:
                return s
        except Exception:
            pass

    # 4) Regex fallback - try substring first to avoid stray matches
    haystacks = [h for h in [sub, text] if h]
    for h in haystacks:
        m = _SCORE_RE.search(h)
        if m:
            try:
                return float(m.group(2))
            except Exception:
                continue

    return None

def llm_score_pair(
    prediction: str,
    ground_truth: str,
    use_fast_path: bool = True,
    tok=None,
    model=None
) -> int:
    """
    Returns 1 for match else 0.
    Keeps your fast path and model decode unchanged - uses robust score extraction that ignores broken rationale quotes.
    """
    # Optional fast path: exact or containment after light normalization
    if use_fast_path:
        npred = _normalize(prediction)
        ngt = _normalize(ground_truth)
        if npred == ngt or ngt in npred or npred in ngt:
            return 1

    # Ensure model is available
    if tok is None or model is None:
        tok, model = get_model()

    # Build chat with system + current case
    messages = [{"role": "system", "content": BINARY_JUDGE_SYSTEM}]
    messages.append({"role": "user", "content": JUDGE_TEMPLATE.format(
        prediction=prediction, ground_truth=ground_truth
    )})

    raw = generate_json(tok, model, messages)

    score_val = _extract_score(raw)
    if score_val is None:
        print(f"[judge-parse-error] Could not read score. RAW_TAIL: {raw[-400:]}")
        return 0

    try:
        score_int = int(round(float(score_val)))
    except Exception:
        print(f"[judge-bad-score] Non numeric score: {score_val!r}. RAW_TAIL: {raw[-400:]}")
        score_int = 0

    return 1 if score_int >= 1 else 0

# API matches best_subspan_em - no tok or model args and returns float
def best_llm_judge(prediction: str, ground_truths: List[str]) -> float:
    """
    Compute a binary equivalence score using an LLM judge.
    Returns 1.0 if any ground truth matches under the rules, else 0.0.
    """
    tok, model = get_model()
    scores = [llm_score_pair(prediction, gt, tok=tok, model=model) for gt in ground_truths]
    return float(max(scores) if scores else 0)

if __name__ == "__main__":
    # Preload model into globals for efficiency
    tok, model = load_model()
    set_model(tok, model)

    # (prediction, ground_truths, expected_score)
    test_cases = [
        # Containment and extra detail -> 1
        ("Paris", ["the city of Paris", "Paris, France"], 1),
        ("Barack Obama", ["President Barack H. Obama", "Obama"], 1),
        ("Mount Everest", ["mount everest!", "Everest"], 1),

        # Conflicting qualifiers -> 0
        ("Paris, Texas", ["Paris, France"], 0),
        ("Springfield, IL", ["Springfield, MA"], 0),
        ("Washington state", ["Washington, D.C."], 0),

        # Acronyms, aliases, and nicknames -> 1
        ("USA", ["United States of America", "U.S."], 1),
        ("NYC", ["New York City"], 1),
        ("UK", ["United Kingdom"], 1),
        ("Munich", ["München"], 1),
        ("SARS-CoV-2", ["COVID-19 virus", "the coronavirus causing COVID-19"], 1),

        # Numeric and formatting normalization -> 1
        ("42", ["forty two", "forty-two"], 1),
        ("10 km", ["ten kilometers"], 1),
        ("Jan 1, 2020", ["1 January 2020"], 1),

        # Clear mismatches -> 0
        ("Cat", ["Dog"], 0),
        ("New York", ["New Jersey"], 0),
        ("Mercury (planet)", ["Mercury (element)"], 0),
        ("Python the language", ["Python the snake"], 0),

        # Tricky near misses -> expected 0
        ("Amazon River", ["Amazon rainforest"], 0),
    ]

    more_test_cases = [
        # Simple aliases and abbreviations -> 1
        ("USA", ["United States", "United States of America"], 1),
        ("UAE", ["United Arab Emirates"], 1),
        ("UN", ["United Nations"], 1),
        ("EU", ["European Union", "E.U."], 1),
        ("USSR", ["Soviet Union"], 1),
        ("IBM", ["International Business Machines"], 1),
        ("IMF", ["International Monetary Fund"], 1),
        ("WHO", ["World Health Organization"], 1),
        ("BBC", ["British Broadcasting Corporation"], 1),
        ("NY", ["New York"], 1),
        ("LA", ["Los Angeles"], 1),
        ("PS5", ["PlayStation 5"], 1),
        ("BTC", ["Bitcoin"], 1),
        ("USD", ["US dollar", "United States dollar"], 1),
        ("GBP", ["British pound", "Pound sterling"], 1),
        ("NaCl", ["sodium chloride"], 1),
        ("H2O", ["water"], 1),
        ("CO2", ["carbon dioxide"], 1),
        ("AI", ["artificial intelligence"], 1),
        ("COVID-19 virus", ["SARS-CoV-2"], 1),

        # Nicknames and common alternative names -> 1
        ("NYC", ["New York City", "the City of New York"], 1),
        ("The Big Apple", ["New York City"], 1),
        ("Sin City", ["Las Vegas"], 1),
        ("Motor City", ["Detroit"], 1),
        ("Emerald City", ["Seattle"], 1),
        ("Bombay", ["Mumbai"], 1),
        ("Peking", ["Beijing"], 1),
        ("Calcutta", ["Kolkata"], 1),
        ("Kyiv", ["Kiev"], 1),

        # Diacritics and transliterations -> 1
        ("Sao Paulo", ["São Paulo"], 1),
        ("Munchen", ["München", "Munich"], 1),
        ("Praha", ["Prague"], 1),
        ("Fujisan", ["Mount Fuji", "Mt. Fuji"], 1),
        ("St Petersburg", ["Saint Petersburg"], 1),

        # Formatting, casing, punctuation, and articles ignored -> 1
        ("the beatles", ["Beatles"], 1),
        ("Spider Man", ["Spider-Man"], 1),
        ("R2D2", ["R2-D2"], 1),
        ("Counter Strike Global Offensive", ["Counter-Strike: Global Offensive", "CS:GO"], 1),
        ("user@example.com", ["User@Example.com"], 1),
        ("123 456 7890", ["(123) 456-7890"], 1),
        ("May 1, 2010", ["2010-05-01", "1 May 2010"], 1),
        ("mac os x", ["OS X"], 1),

        # Company naming variants -> 1
        ("Tesla", ["Tesla, Inc."], 1),
        ("Apple", ["Apple Inc."], 1),
        ("Google", ["Google LLC"], 1),
        ("JPM", ["JPMorgan Chase"], 1),

        # Country and state qualifiers that conflict -> 0
        ("Georgia, United States", ["Georgia, country"], 0),
        ("Congo", ["Democratic Republic of the Congo", "Republic of the Congo"], 0),
        ("DRC", ["Republic of the Congo"], 0),
        ("Republic of the Congo", ["Democratic Republic of the Congo"], 0),
        ("St Petersburg, FL", ["St Petersburg, Russia"], 0),
        ("Paris, France", ["Paris, Texas"], 0),
        ("Springfield, MO", ["Springfield, MA"], 0),
        ("London, Canada", ["London, UK"], 0),

        # One with qualifier vs generic where meanings still match -> 1
        ("Apple Inc.", ["Apple"], 1),
        ("New York City", ["New York"], 1),
        ("United States of America", ["United States"], 1),
        ("Republic of Korea", ["South Korea"], 1),
        ("UK", ["United Kingdom of Great Britain and Northern Ireland"], 1),
        ("Cat", ["Caterpillar Inc."], 1),

        # Rebrand or related but not the same entity -> 0
        ("Alphabet Inc.", ["Google"], 0),

        # Clear mismatches -> 0
        ("Saturn", ["Saturday"], 0),
        ("Bitcoin", ["Ethereum"], 0),
    ]

    test_cases.extend(more_test_cases)

    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"

    passed = 0
    for pred, gts, expected in test_cases:
        score = best_llm_judge(pred, gts)  # returns float 0.0 or 1.0
        result = "PASS" if score == expected else "FAIL"
        if result == "PASS":
            passed += 1
            result_colored = f"{GREEN}{result}{RESET}"
        else:
            result_colored = f"{RED}{result}{RESET}"

            print(
                f"Prediction: {pred!r}, Ground truths: {gts} -> "
                f"Score: {score}  Expected: {expected}  Result: {result_colored}"
            )

    total = len(test_cases)
    print(f"\nSummary: {passed}/{total} passed")
