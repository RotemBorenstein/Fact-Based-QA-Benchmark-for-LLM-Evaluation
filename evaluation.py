import json
import os
import re
import unicodedata
from rapidfuzz import fuzz
from dotenv import load_dotenv
"""
Evaluation script (clean output).
- Reads examples.jsonl and predictions.jsonl (paths below).
- For each example, compares the prediction to the gold answer.
- Writes a clean JSONL with tuples (example, prediction, correct).
- Prints ONLY the general accuracy: correct / total.
"""

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
EXAMPLES_PATH = "data/examples.jsonl"
PREDICTIONS_PATH = "data/predictions.jsonl"
OUT_JSONL = "data/results_clean.jsonl"

THRESHOLD = 90
USE_LLM = True
LLM_MODEL = "gpt-4o-mini"




if API_KEY:
    os.environ["OPENAI_API_KEY"] = API_KEY

_BOILERPLATE_WORDS = {
    "the", "club", "football", "soccer", "fc", "cf", "sc", "ac", "cd", "ud", "de", "afc", "cfc"
}

def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))

def normalize_text(s: str) -> str:
    """Lowercase, remove accents, punctuation -> spaces, collapse spaces."""
    if s is None:
        return ""
    s = strip_accents(s.lower())
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_team_for_match(s: str) -> str:
    """Relaxed normalization for team matching (drops boilerplate tokens like 'fc')."""
    s = normalize_text(s)
    tokens = [t for t in s.split() if t not in _BOILERPLATE_WORDS]
    return " ".join(tokens)


# NUMBER MATCHING
def extract_integers(text: str):
    if text is None:
        return []
    return [int(n) for n in re.findall(r"-?\d+", text)]

def compare_numbers(model_answer: str, gold_value):
    """
    Deterministic numeric comparison:
    - Extract all integers from model_answer
    - Coerce gold_value to int if possible
    - Accept if exactly one extracted integer equals gold
    - If multiple numbers, accept only if the gold appears exactly once
    """
    ints = extract_integers(model_answer)
    try:
        gold_int = int(str(gold_value).strip())
    except Exception:
        return False

    if len(ints) == 1:
        return ints[0] == gold_int
    if gold_int in ints and ints.count(gold_int) == 1:
        return True
    return False


# TEAM MATCHING (FUZZY)
def team_equal(pred: str, gold: str) -> bool:
    """Direct equality with robust normalization for team strings."""
    if not pred or not gold:
        return False
    return (
        normalize_text(pred) == normalize_text(gold)
        or normalize_team_for_match(pred) == normalize_team_for_match(gold)
    )

def fuzzy_match_against_gold(raw_answer: str, gold: str, threshold: int = 90) -> bool:
    """Fuzzy-match predicted team name against the single gold team name."""
    if not raw_answer or not gold:
        return False
    q = normalize_team_for_match(raw_answer)
    g = normalize_team_for_match(gold)
    score = fuzz.token_sort_ratio(q, g)
    return score >= threshold



# LLM FALLBACK: YES/NO CHECK
def llm_yesno_equivalence_check(gold_answer: str, model_answer: str, model: str = "gpt-4o-mini") -> bool:
    """
    Ask an LLM to decide if model_answer and gold_answer refer to the SAME TEAM.
    Returns True/False. (No extra printing here; keep output clean.)
    """
    try:
        from openai import OpenAI
    except Exception:
        return False

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False

    client = OpenAI(api_key=api_key)

    sys_prompt = (
        "You are a strict answer checker.\n"
        "Compare the model's answer to the gold answer.\n"
        "Respond with exactly YES if they mean the same football team (allowing abbreviations, nicknames, or minor spelling differences).\n"
        "Respond with exactly NO if they are different teams.\n"
        "Do not add any other text."
    )
    user_prompt = (
        "Determine if the next answer is essentially the same as this answer.\n"
        f"Gold answer: `{gold_answer or ''}`\n"
        f"Model answer: `{model_answer or ''}`\n"
        "Reply YES or NO only."
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = (resp.choices[0].message.content or "").strip().upper()
        return text == "YES"
    except Exception:
        return False


# ROBUST JSONL LOADER
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# EVALUATION
def get_gold_answer(ex: dict) -> str:
    """
    Support both 'gold_answer' and 'answer' as the gold field.
    """
    val = ex.get("gold_answer", ex.get("answer", ""))
    return str(val).strip()

def is_number_example(ex: dict) -> bool:
    """
    Support both 'answer_type' == 'number' and heuristic for numeric gold.
    """
    at = (ex.get("answer_type") or "").strip().lower()
    if at == "number":
        return True
    gold = get_gold_answer(ex)
    return bool(re.fullmatch(r"-?\d+", gold))

def evaluate_example(ex: dict, pred_answer: str, threshold: int, use_llm: bool, llm_model: str) -> bool:
    gold_str = get_gold_answer(ex)
    if is_number_example(ex):
        return compare_numbers(pred_answer, gold_str)
    if team_equal(pred_answer, gold_str):
        return True
    if fuzzy_match_against_gold(pred_answer, gold_str, threshold):
        return True
    if use_llm and gold_str:
        return llm_yesno_equivalence_check(gold_str, pred_answer, model=llm_model)
    return False


def main():
    gold = {ex.get("id"): ex for ex in load_jsonl(EXAMPLES_PATH) if ex.get("id")}
    preds = {pr.get("id"): pr.get("answer", "") for pr in load_jsonl(PREDICTIONS_PATH) if pr.get("id")}
    total = 0
    correct = 0

    with open(OUT_JSONL, "w", encoding="utf-8") as out_f:
        for ex_id, ex in gold.items():
            total += 1
            pred_answer = (preds.get(ex_id, "") or "").strip().strip('"').strip("'")
            ok = evaluate_example(ex, pred_answer, THRESHOLD, USE_LLM, LLM_MODEL)
            if ok:
                correct += 1

            rec = {
                "id": ex_id,
                "example": ex,
                "prediction": pred_answer,
                "correct": bool(ok),
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total = max(total, 1)
    print(f"{correct}/{total} = {correct/total:.3f}")

if __name__ == "__main__":
    main()
