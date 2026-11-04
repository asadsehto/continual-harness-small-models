# Phase 1 (condition C1): Continual Harness on the 15 train tasks, then evaluation.
# For each seed:
#   1. LEARN: run the 15 train tasks in a shuffled order. Between tasks, when the step count
#      passes an evolution point (25, 50, 75, ...), the Refiner edits the harness.
#   2. EVAL: freeze the final harness and run all 20 tasks once (15 train + 5 held-out).
# Saved in runs/<time>_phase1_<model>/seed_<k>/:
#   state_initial/, state_gen_<n>/, state_final/   harness snapshots
#   learn_trajectory.jsonl, learn_results.jsonl    learning phase
#   evolution_log.jsonl                            every Refiner pass: response, applied edits (before/after), rejected edits
#   eval_trajectory.jsonl, eval_results.jsonl      evaluation phase (same format as Phase 0 results.jsonl)
# Safety: run_shell and run_skill run in the sandbox. RAM: the model is unloaded after every
# task and every Refiner call; if Ollama reports more than MEMORY_LIMIT_GB the run stops.
# Run: uv run python -m scripts.run_phase1 --seeds 3

import argparse
import json
import os
import random
import shutil
import sys
import time

from harness.agent import run_agent
from harness.model import ModelError, memory_gb, server_info, set_host, unload
from harness.refiner import evolution_points_crossed, evolve
from harness.state import create_state, load_state, save_state
from tasks.pilot import ALL_TASKS
from tasks.task import make_workdir, remove_workdir

MEMORY_LIMIT_GB = 10

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="qwen3.5:0.8b-mlx")
parser.add_argument("--refiner-model", default=None, help="model that writes harness edits (Fix A); default: same as --model")
parser.add_argument("--seeds", type=int, default=3)
parser.add_argument("--first-seed", type=int, default=0)
parser.add_argument("--max-steps", type=int, default=15)
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--train-tasks", default="all", help="comma-separated ids (for smoke tests), or 'all'")
parser.add_argument("--skip-eval", action="store_true")
parser.add_argument("--host", default="http://127.0.0.1:11434", help="Ollama server URL")
args = parser.parse_args()

# print progress lines at once (otherwise they sit in a buffer and are lost if the run is killed)
sys.stdout.reconfigure(line_buffering=True)
set_host(args.host)
if args.refiner_model is None:
    args.refiner_model = args.model

train_tasks = []
for task in ALL_TASKS:
    if task["split"] == "train":
        if args.train_tasks == "all" or task["id"] in args.train_tasks.split(","):
            train_tasks.append(task)

run_name = time.strftime("%Y%m%d-%H%M%S") + "_phase1_" + args.model.replace(":", "-")
if args.refiner_model != args.model:
    run_name = run_name + "_refiner-" + args.refiner_model.replace(":", "-")
run_dir = os.path.join("runs", run_name)
os.makedirs(run_dir)
config = dict(vars(args))
config["server"] = server_info(args.model)
config["refiner_server"] = server_info(args.refiner_model)
config["started"] = time.strftime("%Y-%m-%d %H:%M:%S")
with open(os.path.join(run_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=2)
print("run folder:", run_dir, "| server:", config["server"])


def memory_check(where):
    # check and unload both models (they are the same model unless --refiner-model is set)
    used = max(memory_gb(args.model), memory_gb(args.refiner_model))
    unload(args.model)
    if args.refiner_model != args.model:
        unload(args.refiner_model)
    if used > MEMORY_LIMIT_GB:
        print("STOPPING: model used", round(used, 1), "GB (limit", MEMORY_LIMIT_GB, "GB) after", where)
        write_status("STOPPED (memory limit, results incomplete): " + str(round(used, 1)) + " GB after " + where)
        sys.exit(1)
    return round(used, 2)


def run_one_task(task, state, seed, trajectory_file, results_file, extra):
    workdir = make_workdir(task)
    try:
        summary = run_agent(
            args.model, task["instruction"], workdir, args.max_steps,
            seed, args.temperature, trajectory_file, task["id"], state,
        )
    except ModelError:
        remove_workdir(workdir)
        raise
    if os.path.isdir(workdir):
        success = task["check"](workdir, task)
    else:
        success = False
        summary["deleted_task_folder"] = True
    remove_workdir(workdir)
    summary["model_memory_gb"] = memory_check("task " + task["id"])

    row = {"task_id": task["id"], "split": task["split"], "seed": seed, "success": success}
    row.update(extra)
    row.update(summary)
    results_file.write(json.dumps(row) + "\n")
    results_file.flush()
    return row


def read_steps(path, start_byte):
    steps = []
    with open(path) as f:
        f.seek(start_byte)
        for line in f:
            steps.append(json.loads(line))
    return steps


def main():
    for seed in range(args.first_seed, args.first_seed + args.seeds):
        seed_dir = os.path.join(run_dir, "seed_" + str(seed))
        state_dir = os.path.join(seed_dir, "state")
        create_state(state_dir)
        shutil.copytree(state_dir, os.path.join(seed_dir, "state_initial"))
        state = load_state(state_dir)

        order = list(train_tasks)
        random.Random(seed).shuffle(order)
        print("=== seed", seed, "order:", [t["id"] for t in order])

        # ---- 1. learning phase ----
        history = []
        global_step = 0
        generation = 0
        trajectory_path = os.path.join(seed_dir, "learn_trajectory.jsonl")
        trajectory_file = open(trajectory_path, "w")
        results_file = open(os.path.join(seed_dir, "learn_results.jsonl"), "w")
        evolution_file = open(os.path.join(seed_dir, "evolution_log.jsonl"), "w")

        for position in range(len(order)):
            task = order[position]
            start_byte = trajectory_file.tell()
            row = run_one_task(task, state, seed, trajectory_file, results_file, {"position": position, "generation": generation})
            print("learn", position, task["id"], "PASS" if row["success"] else "FAIL", "steps", row["steps"], "gen", generation, row["seconds"], "s")

            previous_step = global_step
            for record in read_steps(trajectory_path, start_byte):
                global_step = global_step + 1
                record["global_step"] = global_step
                history.append(record)
            history.append({"task_id": task["id"], "outcome": "PASS" if row["success"] else "FAIL"})

            # The original evolves once per evolution point. If a task passed two points, we still
            # evolve once (the window would be the same both times).
            if evolution_points_crossed(previous_step, global_step) > 0 and position < len(order) - 1:
                generation = generation + 1
                results = evolve(args.refiner_model, state, history, global_step, generation, seed, args.temperature)
                results["model_memory_gb"] = memory_check("refiner generation " + str(generation))
                save_state(state_dir, state)
                shutil.copytree(state_dir, os.path.join(seed_dir, "state_gen_" + str(generation)))
                evolution_file.write(json.dumps(results, default=str) + "\n")
                evolution_file.flush()
                applied = 0
                for name in ["prompt", "subagents", "skills", "memory"]:
                    applied = applied + len(results[name].get("applied", []))
                print("  refiner gen", generation, "at step", global_step, "| edits applied:", applied, "| counts:", results["counts"], "|", results["seconds"], "s")

        trajectory_file.close()
        results_file.close()
        evolution_file.close()
        save_state(state_dir, state)
        shutil.copytree(state_dir, os.path.join(seed_dir, "state_final"))

        # ---- 2. evaluation with the frozen final harness ----
        if args.skip_eval:
            continue
        frozen = load_state(os.path.join(seed_dir, "state_final"))
        eval_trajectory = open(os.path.join(seed_dir, "eval_trajectory.jsonl"), "w")
        eval_results = open(os.path.join(seed_dir, "eval_results.jsonl"), "w")
        passed = 0
        for task in ALL_TASKS:
            row = run_one_task(task, frozen, seed, eval_trajectory, eval_results, {"generation": generation})
            if row["success"]:
                passed = passed + 1
            print("eval", task["id"], task["split"], "PASS" if row["success"] else "FAIL", "steps", row["steps"])
        eval_trajectory.close()
        eval_results.close()
        print("seed", seed, "eval passed", passed, "of", len(ALL_TASKS))



def write_status(status):
    with open(os.path.join(run_dir, "STATUS"), "w") as f:
        f.write(status + "\n")


try:
    main()
    write_status("finished " + time.strftime("%Y-%m-%d %H:%M:%S"))
except ModelError as error:
    write_status("STOPPED (model server problem, results incomplete): " + str(error))
    print("STOPPED: model server problem. Results so far are saved but incomplete.")
    print(error)
    sys.exit(2)

print("saved in", run_dir)
