# Runs the baseline agent (C0) on tasks and saves everything in runs/<time>/.
#   trajectory.jsonl  one line per agent step (what the model said, which tools, timing)
#   results.jsonl     one line per task run (pass/fail, steps, tokens, seconds)
# Examples:
#   uv run python -m scripts.run_pilot --model gemma4:e2b-it-qat --tasks fix_fib --seeds 1
#   uv run python -m scripts.run_pilot --model qwen3.5:0.8b-mlx --tasks all --seeds 3

import argparse
import json
import os
import sys
import time

from harness.agent import run_agent
from harness.model import memory_gb, server_info, set_host, unload

MEMORY_LIMIT_GB = 10
from tasks.pilot import ALL_TASKS
from tasks.task import make_workdir, remove_workdir

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="gemma4:e2b-it-qat")
parser.add_argument("--tasks", default="fix_fib", help="comma-separated task ids, or 'all'")
parser.add_argument("--seeds", type=int, default=1)
parser.add_argument("--max-steps", type=int, default=15)
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--host", default="http://127.0.0.1:11434", help="Ollama server URL")
args = parser.parse_args()

# print progress lines at once (otherwise they sit in a buffer and are lost if the run is killed)
sys.stdout.reconfigure(line_buffering=True)
set_host(args.host)

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
config = dict(vars(args))
config["server"] = server_info(args.model)
with open(os.path.join(run_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

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
        # If the agent deleted its own task folder, the task failed (checkers expect the folder).
        if os.path.isdir(workdir):
            success = task["check"](workdir, task)
        else:
            success = False
            summary["deleted_task_folder"] = True
        remove_workdir(workdir)

        # Memory guard: record how much memory the model used, then unload it.
        peak_gb = memory_gb(args.model)
        summary["model_memory_gb"] = round(peak_gb, 2)
        unload(args.model)
        if peak_gb > MEMORY_LIMIT_GB:
            print("WARNING: model used", round(peak_gb, 1), "GB (limit", MEMORY_LIMIT_GB, "GB) on task", task["id"])

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
