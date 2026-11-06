# All conditions in one table, each compared with the baseline.
# Usage:
#   uv run python -m scripts.analyze runs/<baseline> name=runs/<run> name=runs/<run> ...
# For each condition:
#   success on all / train / held-out tasks with Wilson 95% CIs
#   paired comparison with the baseline on the same (task, seed): better / worse, and an exact
#   McNemar test (two-sided binomial test on the pairs that differ). p < 0.05 = unlikely to be chance.
#   forgetting: train tasks passed during learning but failed at eval (same seed)
#   cost: mean seconds and mean prompt tokens per eval task run, learning minutes

import json
import math
import os
import sys


def wilson(passed, total):
    if total == 0:
        return 0.0, 0.0
    z = 1.96
    p = passed / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(better, worse):
    # two-sided exact binomial test with p = 0.5 on the n = better + worse discordant pairs
    n = better + worse
    if n == 0:
        return 1.0
    k = min(better, worse)
    tail = 0.0
    for i in range(k + 1):
        tail = tail + math.comb(n, i)
    return min(1.0, 2 * tail / 2 ** n)


def read_rows(path):
    rows = []
    if os.path.exists(path):
        for line in open(path):
            rows.append(json.loads(line))
    return rows


def load(run_dir):
    # returns (eval rows, learn rows); the baseline has only results.jsonl
    if os.path.exists(os.path.join(run_dir, "results.jsonl")):
        return read_rows(os.path.join(run_dir, "results.jsonl")), []
    eval_rows = []
    learn_rows = []
    for name in sorted(os.listdir(run_dir)):
        if name.startswith("seed_"):
            eval_rows = eval_rows + read_rows(os.path.join(run_dir, name, "eval_results.jsonl"))
            learn_rows = learn_rows + read_rows(os.path.join(run_dir, name, "learn_results.jsonl"))
    return eval_rows, learn_rows


def rate(rows):
    passed = sum(1 for r in rows if r["success"])
    low, high = wilson(passed, len(rows))
    return str(passed) + "/" + str(len(rows)) + " = " + str(round(100 * passed / max(1, len(rows)))) + "% (" + str(round(100 * low)) + "–" + str(round(100 * high)) + ")"


baseline_dir = sys.argv[1]
conditions = [("C0 baseline", baseline_dir)]
for arg in sys.argv[2:]:
    name, path = arg.split("=", 1)
    conditions.append((name, path))

for name, path in conditions:
    status_file = os.path.join(path, "STATUS")
    if os.path.exists(status_file) and not open(status_file).read().startswith("finished"):
        print("WARNING: not finished:", path)

base_rows, unused = load(baseline_dir)
base = {}
for r in base_rows:
    base[(r["task_id"], r["seed"])] = r["success"]

print("| Condition | All | Train | Held-out | Better / worse vs C0 | McNemar p | Forgetting | s / task | prompt tokens / task | learning min |")
print("|---|---|---|---|---|---|---|---|---|---|")
for name, path in conditions:
    rows, learn = load(path)
    train = [r for r in rows if r["split"] == "train"]
    heldout = [r for r in rows if r["split"] == "heldout"]
    better = 0
    worse = 0
    for r in rows:
        before = base.get((r["task_id"], r["seed"]))
        if before is None:
            continue
        if r["success"] and not before:
            better = better + 1
        if before and not r["success"]:
            worse = worse + 1
    eval_pass = {}
    for r in rows:
        eval_pass[(r["task_id"], r["seed"])] = r["success"]
    learned = [r for r in learn if r["success"]]
    forgotten = [r for r in learned if not eval_pass.get((r["task_id"], r["seed"]), False)]
    seconds = sum(r["seconds"] for r in rows) / max(1, len(rows))
    tokens = sum(r["prompt_tokens"] for r in rows) / max(1, len(rows))
    learn_minutes = sum(r["seconds"] for r in learn) / 60
    if path == baseline_dir:
        paired = "-"
        p_value = "-"
        forgetting = "-"
    else:
        paired = str(better) + " / " + str(worse)
        p_value = str(round(mcnemar_exact(better, worse), 3))
        forgetting = str(len(forgotten)) + " of " + str(len(learned))
    print("| " + " | ".join([name, rate(rows), rate(train), rate(heldout), paired, p_value, forgetting,
                             str(round(seconds, 1)), str(round(tokens)), str(round(learn_minutes, 1)) if learn else "-"]) + " |")
