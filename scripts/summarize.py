# Summarizes one or more run folders: success rate with a 95% Wilson interval,
# per split, per category, per task, plus time, steps, tokens and stop reasons.
# Run: uv run python -m scripts.summarize runs/<folder> [runs/<folder2> ...]

import json
import math
import os
import sys

from tasks.pilot import ALL_TASKS

CATEGORY = {}
for task in ALL_TASKS:
    CATEGORY[task["id"]] = task["category"]


def wilson(passed, total):
    if total == 0:
        return 0.0, 0.0
    z = 1.96
    p = passed / total
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return max(0.0, center - half), min(1.0, center + half)


def rate_line(name, rows):
    passed = 0
    for row in rows:
        if row["success"]:
            passed = passed + 1
    low, high = wilson(passed, len(rows))
    percent = 100 * passed / len(rows) if rows else 0
    return name.ljust(16) + str(passed).rjust(3) + "/" + str(len(rows)).ljust(4) + (str(round(percent)) + "%").rjust(5) + "   95% CI " + str(round(100 * low)) + "–" + str(round(100 * high)) + "%"


for run_dir in sys.argv[1:]:
    with open(os.path.join(run_dir, "config.json")) as f:
        config = json.load(f)
    rows = []
    with open(os.path.join(run_dir, "results.jsonl")) as f:
        for line in f:
            row = json.loads(line)
            row["category"] = CATEGORY[row["task_id"]]
            rows.append(row)

    print("=" * 70)
    print(config["model"], "|", len(rows), "task runs |", run_dir)
    print(rate_line("ALL", rows))
    for key in ["split", "category"]:
        values = sorted(set(row.get(key, "") for row in rows if key in row))
        for value in values:
            print(rate_line(key + "=" + value, [r for r in rows if r.get(key) == value]))

    print("-- per task (passes out of seeds)")
    task_ids = []
    for row in rows:
        if row["task_id"] not in task_ids:
            task_ids.append(row["task_id"])
    for task_id in task_ids:
        task_rows = [r for r in rows if r["task_id"] == task_id]
        passes = sum(1 for r in task_rows if r["success"])
        print("  " + task_id.ljust(16) + str(passes) + "/" + str(len(task_rows)))

    seconds = [r["seconds"] for r in rows]
    steps = [r["steps"] for r in rows]
    print("-- cost")
    print("  total minutes:", round(sum(seconds) / 60, 1), "| mean seconds per run:", round(sum(seconds) / len(rows), 1), "| max:", max(seconds))
    print("  mean steps:", round(sum(steps) / len(rows), 1), "| mean output tokens:", round(sum(r["output_tokens"] for r in rows) / len(rows)))
    stop = {}
    for row in rows:
        stop[row["stop_reason"]] = stop.get(row["stop_reason"], 0) + 1
    print("  stop reasons:", stop, "| runs with format errors:", sum(1 for r in rows if r["format_errors"] > 0))
