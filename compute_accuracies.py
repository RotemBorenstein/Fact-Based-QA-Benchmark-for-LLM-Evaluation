
import json
from collections import defaultdict
import csv

data_path = "data/results_clean.jsonl"
out_csv = "data/accuracy_breakdowns.csv"

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def safe_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "t"}
    return bool(v)

def agg_accuracy(records, key_func):
    counts = defaultdict(lambda: [0, 0])  # [correct, total]
    for rec in records:
        k = key_func(rec)
        if k is None:
            continue
        is_correct = safe_bool(rec.get("correct", False))
        counts[k][1] += 1
        if is_correct:
            counts[k][0] += 1
    return {k: (c, n, (c / n) if n else 0.0) for k, (c, n) in counts.items()}

def _clean_league_name(raw_league):
    if not raw_league:
        return None
    parts = raw_league.split()
    if len(parts) > 1 and any(ch.isdigit() for ch in parts[0]):
        return " ".join(parts[1:])
    return raw_league

def _safe_int(x):
    try:
        return int(str(x))
    except Exception:
        return None

# -------- Load data into a normalized records list --------
raw = list(read_jsonl(data_path))
records = []
for r in raw:
    ex = r.get("example", {}) or {}
    qtype = ex.get("question_type")
    pos = ex.get("position")
    pos1 = ex.get("position1")
    pos2 = ex.get("position2")
    i1, i2 = _safe_int(pos1), _safe_int(pos2)
    placement_diff = (abs(i1 - i2) if (i1 is not None and i2 is not None) else None)

    records.append({
        "correct": r.get("correct", False),
        "difficulty": ex.get("difficulty"),
        "question_type": qtype,
        "answer_type": ex.get("answer_type"),
        "league": _clean_league_name(ex.get("league")),
        "season": ex.get("season"),
        # For new breakdowns:
        "stat": ex.get("stat"),                  # for single_season_stat
        "placement": pos,                        # for single_season_placement
        "placement_diff": placement_diff,        # for pairwise_higher
    })

# -------- Compute existing breakdowns --------
overall = agg_accuracy(records, lambda _: "overall")["overall"]
by_difficulty = agg_accuracy(records, lambda rec: rec.get("difficulty"))
by_qtype = agg_accuracy(records, lambda rec: rec.get("question_type"))
by_atype = agg_accuracy(records, lambda rec: rec.get("answer_type"))
by_league = agg_accuracy(records, lambda rec: rec.get("league"))
by_season = agg_accuracy(records, lambda rec: rec.get("season"))

# -------- per-stat (single_season_stat only) --------
sss = [rec for rec in records if rec.get("question_type") == "single_season_stat"]
by_stat_sss = agg_accuracy(sss, lambda rec: rec.get("stat"))

# -------- per-placement (single_season_placement only) --------
ssp = [rec for rec in records if rec.get("question_type") == "single_season_placement"]
by_place_ssp = agg_accuracy(ssp, lambda rec: rec.get("placement"))

# -------- per placement-diff (pairwise_higher only) --------
pwh = [rec for rec in records if rec.get("question_type") == "pairwise_higher"]
by_diff_pwh = agg_accuracy(pwh, lambda rec: rec.get("placement_diff"))


def print_block(title, stats):
    print(f"\n=== {title} ===")
    for k, (c, n, acc) in sorted(stats.items(), key=lambda x: (str(x[0]))):
        print(f"{k}: {c}/{n} = {acc:.3f}")


print("Accuracy Report")
print("================")
print(f"Overall: {overall[0]}/{overall[1]} = {overall[2]:.3f}")
print_block("By difficulty", by_difficulty)
print_block("By question_type", by_qtype)
print_block("By answer_type", by_atype)
print_block("By league", by_league)
print_block("By season", by_season)


print_block("By stat (single_season_stat)", by_stat_sss)
print_block("By placement (single_season_placement)", by_place_ssp)
print_block("By |pos1-pos2| (pairwise_higher)", by_diff_pwh)


with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["category", "key", "correct", "total", "accuracy"])

    def dump_stats(category, stats):
        for k, (c, n, acc) in stats.items():
            writer.writerow([category, k, c, n, f"{acc:.3f}"])

    writer.writerow(["overall", "overall", overall[0], overall[1], f"{overall[2]:.3f}"])
    dump_stats("difficulty", by_difficulty)
    dump_stats("question_type", by_qtype)
    dump_stats("answer_type", by_atype)
    dump_stats("league", by_league)
    dump_stats("season", by_season)
    dump_stats("stat_single_season_stat", by_stat_sss)
    dump_stats("placement_single_season_placement", by_place_ssp)
    dump_stats("placement_diff_pairwise_higher", by_diff_pwh)

print(f"\nResults saved to {out_csv}")
