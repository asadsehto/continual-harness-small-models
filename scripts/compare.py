# Compares the Phase 0 baseline (C0) with a Phase 1 run (C1).
#   - overall, train, held-out success (95% Wilson interval)
#   - paired by (task, seed): C1 better / same / worse than C0
#   - per task: C0 passes vs C1 passes
#   - forgetting inside the learning run: tasks the agent PASSED while learning, but FAILED
#     at evaluation with the final harness (same seed)
# Run: uv run python -m scripts.compare runs/<baseline folder> runs/<phase1 folder>

import json
import math
import os
import sys


# 95% Wilson interval (same as scripts/summarize.py)
def wilson(passed, total):
    if total == 0:
        return 0.0, 0.0
    z = 1.96
    p = passed / total
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return max(0.0, center - half), min(1.0, center + half)


def read_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def rate(rows):
    passed = sum(1 for r in rows if r["success"])
    low, high = wilson(passed, len(rows))
    if not rows:
        return "-"
    return str(passed) + "/" + str(len(rows)) + " = " + str(round(100 * passed / len(rows))) + "% (95% CI " + str(round(100 * low)) + "–" + str(round(100 * high)) + "%)"


baseline_dir = sys.argv[1]
phase1_dir = sys.argv[2]
c0 = read_rows(os.path.join(baseline_dir, "results.jsonl"))

c1 = []
learn = []
evolutions = []
for name in sorted(os.listdir(phase1_dir)):
    seed_dir = os.path.join(phase1_dir, name)
    if not name.startswith("seed_"):
        continue
    if os.path.exists(os.path.join(seed_dir, "eval_results.jsonl")):
        c1 = c1 + read_rows(os.path.join(seed_dir, "eval_results.jsonl"))
    learn = learn + read_rows(os.path.join(seed_dir, "learn_results.jsonl"))
    if os.path.exists(os.path.join(seed_dir, "evolution_log.jsonl")):
        evolutions = evolutions + read_rows(os.path.join(seed_dir, "evolution_log.jsonl"))

seeds = sorted(set(r["seed"] for r in c1))
c0 = [r for r in c0 if r["seed"] in seeds]

print("seeds:", seeds)
print("                 C0 baseline                      C1 Continual Harness (eval, frozen harness)")
for split in ["all", "train", "heldout"]:
    a = [r for r in c0 if split == "all" or r["split"] == split]
    b = [r for r in c1 if split == "all" or r["split"] == split]
    print(split.ljust(8), rate(a).ljust(40), rate(b))
print("learning phase (C1 while editing):", rate(learn))

c0_by_key = {}
for r in c0:
    c0_by_key[(r["task_id"], r["seed"])] = r["success"]
better = 0
worse = 0
same = 0
for r in c1:
    before = c0_by_key.get((r["task_id"], r["seed"]))
    if before is None:
        continue
    if r["success"] and not before:
        better = better + 1
    elif before and not r["success"]:
        worse = worse + 1
    else:
        same = same + 1
print()
print("paired (task, seed): C1 better", better, "| worse", worse, "| same", same)

print()
print("per task: C0 passes -> C1 passes (out of", len(seeds), "seeds)")
task_ids = []
for r in c0:
    if r["task_id"] not in task_ids:
        task_ids.append(r["task_id"])
for task_id in task_ids:
    a = sum(1 for r in c0 if r["task_id"] == task_id and r["success"])
    b = sum(1 for r in c1 if r["task_id"] == task_id and r["success"])
    split = [r["split"] for r in c0 if r["task_id"] == task_id][0]
    mark = ""
    if b > a:
        mark = "  up"
    if b < a:
        mark = "  DOWN"
    print("  " + task_id.ljust(16) + split.ljust(8) + str(a) + " -> " + str(b) + mark)

print()
learned_pass = {}
for r in learn:
    if r["success"]:
        learned_pass[(r["task_id"], r["seed"])] = True
forgotten = []
for r in c1:
    if learned_pass.get((r["task_id"], r["seed"])) and not r["success"]:
        forgotten.append(r["task_id"] + " (seed " + str(r["seed"]) + ")")
print("forgetting: passed while learning, failed at eval:", len(forgotten), "of", len(learned_pass), forgotten)

print()
total_applied = 0
parse_fail = 0
calls = 0
for e in evolutions:
    for name in ["prompt", "subagents", "skills", "memory"]:
        part = e.get(name, {})
        calls = calls + 1
        total_applied = total_applied + len(part.get("applied", []))
        if "failed_to_parse_response" in part.get("rejected", []) or "error" in part:
            parse_fail = parse_fail + 1
print("refiner: generations", len(evolutions), "| passes", calls, "| edits applied", total_applied, "| passes with unparseable output or error", parse_fail)
used = {"run_skill": 0, "call_subagent": 0}
for name in sorted(os.listdir(phase1_dir)):
    for file_name in ["learn_trajectory.jsonl", "eval_trajectory.jsonl"]:
        path = os.path.join(phase1_dir, name, file_name)
        if os.path.exists(path):
            for line in open(path):
                for call in json.loads(line)["tool_calls"]:
                    if call["name"] in used:
                        used[call["name"]] = used[call["name"]] + 1
print("agent used evolved tools:", used)
