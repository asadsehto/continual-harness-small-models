# Runs the baseline agent (C0) on tasks and saves everything in runs/<time>/.
#   trajectory.jsonl  one line per agent step (what the model said, which tools, timing)
#   results.jsonl     one line per task run (pass/fail, steps, tokens, seconds)
# Examples:
#   uv run python -m scripts.run_pilot --model gemma4:e2b-it-qat --tasks fix_fib --seeds 1
#   uv run python -m scripts.run_pilot --model qwen3.5:0.8b-mlx --tasks all --seeds 3

import argparse
import json
import os
import time

from harness.agent import run_agent
from tasks.pilot import ALL_TASKS
from tasks.task import make_workdir, remove_workdir

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="gemma4:e2b-it-qat")
parser.add_argument("--tasks", default="fix_fib", help="comma-separated task ids, or 'all'")
parser.add_argument("--seeds", type=int, default=1)
parser.add_argument("--max-steps", type=int, default=15)
parser.add_argument("--temperature", type=float, default=0.7)
args = parser.parse_args()

if args.tasks == "all":
    tasks = ALL_TASKS
else:
    wanted = args.tasks.split(",")
    tasks = []
    for task in ALL_TASKS:
        if task["id"] in wanted:
            tasks.append(task)

run_dir = os.path.join("runs", time.strftime("%Y%m%d-%H%M%S") + "_" + args.model.replace(":", "-"))
os.makedirs(run_dir)
with open(os.path.join(run_dir, "config.json"), "w") as f:
    json.dump(vars(args), f, indent=2)

trajectory_file = open(os.path.join(run_dir, "trajectory.jsonl"), "w")
results_file = open(os.path.join(run_dir, "results.jsonl"), "w")

passed = 0
total = 0
total_start = time.time()
for seed in range(args.seeds):
    for task in tasks:
        workdir = make_workdir(task)
        summary = run_agent(
            args.model, task["instruction"], workdir, args.max_steps,
            seed, args.temperature, trajectory_file, task["id"],
        )
        success = task["check"](workdir, task)
        remove_workdir(workdir)

        total = total + 1
        if success:
            passed = passed + 1
        row = {"task_id": task["id"], "split": task["split"], "seed": seed, "success": success}
        row.update(summary)
        results_file.write(json.dumps(row) + "\n")
        results_file.flush()
        print(task["id"], "seed", seed, "PASS" if success else "FAIL", summary)

trajectory_file.close()
results_file.close()
print()
print("passed", passed, "of", total, "| total seconds:", round(time.time() - total_start, 1), "| saved in", run_dir)
