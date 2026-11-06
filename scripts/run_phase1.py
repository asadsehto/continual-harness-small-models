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
#   gate_trajectory.jsonl, gate_results.jsonl      only with --gate (Fix B): the gate's test runs
# Technique #3 (--memory-db): memory is a database of up to MAX_DB_ENTRIES entries; each task's
# prompt gets only the TOP_K entries most similar to its instruction (harness/memory_db.py).
# Technique #4 (--verified-skills): a new or rewritten skill is run once on a fresh copy of the most
# recent train task's folder; it is kept only if it runs without error and returns a result.
# Technique #5 (--structured-errors): failed tool calls return a typed error with a fixed hint.
# Baseline --reflexion: no Refiner; after each failed train task the model writes a self-reflection,
# and the last 3 are shown as memory (harness/reflexion.py). Saved in reflections.jsonl.
# Fix B (--gate): after the Refiner edits the harness, rerun the train tasks that ran with the
# old harness since the last edit (at most GATE_TASKS of them), same seed, with the new harness.
# Keep the edits only if the new harness passes at least as many; otherwise go back to the old one.
# Safety: run_shell and run_skill run in the sandbox. RAM: the model is unloaded after every
# task and every Refiner call; if Ollama reports more than MEMORY_LIMIT_GB the run stops.
# Run: uv run python -m scripts.run_phase1 --seeds 3

import argparse
import copy
import json
import os
import random
import shutil
import sys
import time

from harness.agent import run_agent
from harness.model import ModelError, memory_gb, server_info, set_host, set_model_host, unload
from harness.memory_db import EMBED_MODEL, MAX_DB_ENTRIES, TOP_K, retrieve
from harness.refiner import evolution_points_crossed, evolve
from harness.reflexion import add_reflection, reflect
from harness.state import MAX_MEMORY_ENTRIES
from harness.tools import run_skill_code
from harness.state import create_state, load_state, save_state
from tasks.pilot import ALL_TASKS
from tasks.task import make_workdir, remove_workdir

MEMORY_LIMIT_GB = 10           # task model (qwen)
REFINER_MEMORY_LIMIT_GB = 16   # refiner model (e.g. gemma4:e4b)
GATE_TASKS = 3                 # Fix B: how many recent train tasks the gate reruns

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="qwen3.5:0.8b-mlx")
parser.add_argument("--refiner-model", default=None, help="model that writes harness edits (Fix A); default: same as --model")
parser.add_argument("--memory-db", action="store_true", help="technique #3: memory database with retrieval of the most relevant entries")
parser.add_argument("--verified-skills", action="store_true", help="technique #4: keep a skill only if its code runs without error")
parser.add_argument("--structured-errors", action="store_true", help="technique #5: failed tool calls return a typed error with a hint")
parser.add_argument("--reflexion", action="store_true", help="baseline: Reflexion-style self-reflections instead of the Refiner")
parser.add_argument("--gate", action="store_true", help="Fix B: keep Refiner edits only if they do not lower the pass count on recent train tasks")
parser.add_argument("--seeds", type=int, default=3)
parser.add_argument("--first-seed", type=int, default=0)
parser.add_argument("--max-steps", type=int, default=15)
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--train-tasks", default="all", help="comma-separated ids (for smoke tests), or 'all'")
parser.add_argument("--skip-eval", action="store_true")
parser.add_argument("--host", default="http://127.0.0.1:11434", help="Ollama server URL")
parser.add_argument("--refiner-host", default=None, help="Ollama server for the refiner model; default: --host")
parser.add_argument("--embed-host", default=None, help="Ollama server for the embedding model; default: --host")
args = parser.parse_args()

# print progress lines at once (otherwise they sit in a buffer and are lost if the run is killed)
sys.stdout.reconfigure(line_buffering=True)
set_host(args.host)
if args.refiner_model is None:
    args.refiner_model = args.model
if args.refiner_host is not None and args.refiner_model != args.model:
    set_model_host(args.refiner_model, args.refiner_host)
if args.embed_host is not None:
    set_model_host(EMBED_MODEL, args.embed_host)

train_tasks = []
for task in ALL_TASKS:
    if task["split"] == "train":
        if args.train_tasks == "all" or task["id"] in args.train_tasks.split(","):
            train_tasks.append(task)

run_name = time.strftime("%Y%m%d-%H%M%S") + "_phase1_" + args.model.replace(":", "-")
if args.refiner_model != args.model:
    run_name = run_name + "_refiner-" + args.refiner_model.replace(":", "-")
if args.gate:
    run_name = run_name + "_gate"
if args.memory_db:
    run_name = run_name + "_memdb"
if args.reflexion:
    run_name = run_name.replace("_phase1_", "_reflexion_")
if args.verified_skills:
    run_name = run_name + "_verified"
if args.structured_errors:
    run_name = run_name + "_errors"
run_dir = os.path.join("runs", run_name)
os.makedirs(run_dir)
config = dict(vars(args))
config["server"] = server_info(args.model)
config["refiner_server"] = server_info(args.refiner_model)
if args.memory_db:
    config["memory_db"] = {"embed_model": EMBED_MODEL, "top_k": TOP_K, "max_entries": MAX_DB_ENTRIES, "embed_server": server_info(EMBED_MODEL)}
config["started"] = time.strftime("%Y-%m-%d %H:%M:%S")
with open(os.path.join(run_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=2)
print("run folder:", run_dir, "| server:", config["server"])


def memory_check(where):
    # check and unload both models (they are the same model unless --refiner-model is set)
    # only one model is loaded at a time: qwen after a task, the refiner after a refiner generation
    task_used = memory_gb(args.model)
    refiner_used = memory_gb(args.refiner_model) if args.refiner_model != args.model else 0.0
    unload(args.model)
    if args.refiner_model != args.model:
        unload(args.refiner_model)
    used = max(task_used, refiner_used)
    if task_used > MEMORY_LIMIT_GB or refiner_used > REFINER_MEMORY_LIMIT_GB:
        print("STOPPING: memory limit. task model", round(task_used, 1), "GB (limit", MEMORY_LIMIT_GB, ") refiner", round(refiner_used, 1), "GB (limit", REFINER_MEMORY_LIMIT_GB, ") after", where)
        write_status("STOPPED (memory limit, results incomplete): " + str(round(used, 1)) + " GB after " + where)
        sys.exit(1)
    return round(used, 2)


def run_one_task(task, state, seed, trajectory_file, results_file, extra):
    memory_used = None
    if args.memory_db:
        # the agent sees only the entries most relevant to this task
        task_state = dict(state)
        task_state["memory"] = retrieve(state["memory"], task["instruction"])
        memory_used = [entry["id"] for entry in task_state["memory"]]
        state = task_state
    workdir = make_workdir(task)
    try:
        summary = run_agent(
            args.model, task["instruction"], workdir, args.max_steps,
            seed, args.temperature, trajectory_file, task["id"], state, args.structured_errors,
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
    if memory_used is not None:
        row["memory_used"] = memory_used
    results_file.write(json.dumps(row) + "\n")
    results_file.flush()
    return row


def make_skill_verifier(task):
    # Technique #4: test skill code on a fresh copy of `task`'s starting folder.
    def verify(code, test_args):
        if not code.strip():
            return False, "no code"
        if not isinstance(test_args, dict):
            test_args = {}
        workdir = make_workdir(task)
        try:
            output = run_skill_code(workdir, code, test_args)
        finally:
            remove_workdir(workdir)
        passed = output.startswith("exit code: 0") and "RESULT:" in output and "RESULT: null" not in output
        return passed, output
    return verify


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
        if args.gate:
            gate_trajectory = open(os.path.join(seed_dir, "gate_trajectory.jsonl"), "w")
            gate_results = open(os.path.join(seed_dir, "gate_results.jsonl"), "w")
        if args.reflexion:
            reflections_file = open(os.path.join(seed_dir, "reflections.jsonl"), "w")
        # tasks that ran with the current harness since the last edit: [(task, passed), ...]
        recent = []

        for position in range(len(order)):
            task = order[position]
            start_byte = trajectory_file.tell()
            row = run_one_task(task, state, seed, trajectory_file, results_file, {"position": position, "generation": generation})
            print("learn", position, task["id"], "PASS" if row["success"] else "FAIL", "steps", row["steps"], "gen", generation, row["seconds"], "s")
            recent.append((task, row["success"]))

            previous_step = global_step
            for record in read_steps(trajectory_path, start_byte):
                global_step = global_step + 1
                record["global_step"] = global_step
                history.append(record)
            history.append({"task_id": task["id"], "outcome": "PASS" if row["success"] else "FAIL"})

            if args.reflexion:
                # Reflexion baseline: reflect on failures only, never run the Refiner
                if not row["success"] and position < len(order) - 1:
                    # this task's steps plus its FAIL line (same format the Refiner sees)
                    task_steps = history[len(history) - len(read_steps(trajectory_path, start_byte)) - 1:]
                    text, stats = reflect(args.model, task["instruction"], task_steps, seed, args.temperature)
                    memory_check("reflection after " + task["id"])
                    add_reflection(state, task["id"], text)
                    save_state(state_dir, state)
                    reflections_file.write(json.dumps({"task_id": task["id"], "position": position, "reflection": text, "stats": stats, "memory_now": [m["id"] for m in state["memory"]]}) + "\n")
                    reflections_file.flush()
                    print("  reflection after", task["id"], "|", len(text), "chars")
                continue

            # The original evolves once per evolution point. If a task passed two points, we still
            # evolve once (the window would be the same both times).
            if evolution_points_crossed(previous_step, global_step) > 0 and position < len(order) - 1:
                generation = generation + 1
                old_state = copy.deepcopy(state)
                memory_cap = MAX_DB_ENTRIES if args.memory_db else MAX_MEMORY_ENTRIES
                verify_skill = make_skill_verifier(task) if args.verified_skills else None
                results = evolve(args.refiner_model, state, history, global_step, generation, seed, args.temperature, memory_cap, verify_skill)
                results["model_memory_gb"] = memory_check("refiner generation " + str(generation))
                applied = 0
                for name in ["prompt", "subagents", "skills", "memory"]:
                    applied = applied + len(results[name].get("applied", []))
                print("  refiner gen", generation, "at step", global_step, "| edits applied:", applied, "| counts:", results["counts"], "|", results["seconds"], "s")

                # Fix B: the gate (skipped when nothing changed)
                if args.gate and applied > 0:
                    gate_tasks = recent[-GATE_TASKS:]
                    old_passes = 0
                    new_passes = 0
                    for gate_task, old_success in gate_tasks:
                        if old_success:
                            old_passes = old_passes + 1
                        gate_row = run_one_task(gate_task, state, seed, gate_trajectory, gate_results, {"generation": generation})
                        if gate_row["success"]:
                            new_passes = new_passes + 1
                    accepted = new_passes >= old_passes
                    results["gate"] = {
                        "tasks": [t["id"] for t, s in gate_tasks],
                        "old_passes": old_passes,
                        "new_passes": new_passes,
                        "accepted": accepted,
                    }
                    if not accepted:
                        state = old_state
                    print("  gate:", results["gate"]["tasks"], "old", old_passes, "new", new_passes, "->", "KEEP edits" if accepted else "UNDO edits")

                save_state(state_dir, state)
                shutil.copytree(state_dir, os.path.join(seed_dir, "state_gen_" + str(generation)))
                evolution_file.write(json.dumps(results, default=str) + "\n")
                evolution_file.flush()
                recent = []

        trajectory_file.close()
        results_file.close()
        evolution_file.close()
        if args.reflexion:
            reflections_file.close()
        if args.gate:
            gate_trajectory.close()
            gate_results.close()
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
