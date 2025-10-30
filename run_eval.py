import json, time, random
from openai import OpenAI
import os
from dotenv import load_dotenv
# ===== Config =====
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("Missing OPENAI_API_KEY environment variable.")
INPUT_FILE = "data/examples.jsonl"
PREDICTIONS_FILE = "data/predictions.jsonl"
MODEL = "gpt-4o"
MAX_RETRIES = 3
MIN_DELAY = 0.7
MAX_DELAY = 1.5

# ===== Auth =====
if not API_KEY or not API_KEY.startswith("sk-"):
    raise ValueError("Please set API_KEY at the top of the script.")
client = OpenAI(api_key=API_KEY)

def call_model(prompt: str):
    """Return (answer_text or None, error_text or None)."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=50,
            )
            text = (resp.choices[0].message.content or "").strip()
            return text, None
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(random.uniform(MIN_DELAY, MAX_DELAY) + attempt)
            else:
                return None, str(e)

def main():
    total = success = fail = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(PREDICTIONS_FILE, "w", encoding="utf-8") as fpred:

        for line in fin:
            total += 1
            try:
                ex = json.loads(line)
                ex_id = ex.get("id")
                prompt = ex.get("prompt")
            except Exception:
                ex_id, prompt = None, None

            if not ex_id or not prompt:
                fail += 1
                continue

            answer, err = call_model(prompt)
            if answer is None:
                fail += 1
                print(f"[ERROR] Example ID {ex_id} failed: {err}")
                fpred.write(json.dumps({"id": ex_id, "answer": ""}, ensure_ascii=False) + "\n")
                fpred.write(json.dumps({"id": ex_id, "answer": ""}, ensure_ascii=False) + "\n")
            else:
                success += 1
                fpred.write(json.dumps({"id": ex_id, "answer": answer}, ensure_ascii=False) + "\n")

            time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

    print(f"Done. Total: {total} | Success: {success} | Failed: {fail}")
    print(f"Wrote predictions to {PREDICTIONS_FILE}")

if __name__ == "__main__":
    main()
