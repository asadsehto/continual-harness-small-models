# Export the runs used in the paper into small, public data files in data/.
# Run it once, on the machine that has the runs/ folder:
#   uv run python -m scripts.export_data
#
# Only numbers and task names are exported: no trajectories, no tool output, no paths and no
# host names. The files in data/ are all that scripts/make_tables.py needs.
#
#   data/task_runs.csv      one row per task run (learning, gate rerun or evaluation)
#   data/edits.csv          one row per refiner run: harness size after the edit, gate decision
#   data/final_harness.csv  one row per seed: size of the final harness, same as the start or not
#   data/tool_calls.csv     skill and sub-agent calls per seed and phase
#   data/servers.csv        Ollama version and model file digest of every model used

import csv
import json
import os

RUNS = "runs"
OUT = "data"

# (model, condition) -> run folder. These are the runs reported in the paper; to export your
# own runs, change the folder names here.
RUNS_USED = [
    ("qwen", "C0", "20261105-001629_qwen3.5-0.8b-mlx"),
    ("qwen", "C1", "20261105-003144_phase1_qwen3.5-0.8b-mlx"),
    ("qwen", "refiner", "20261105-020938_phase1_qwen3.5-0.8b-mlx_refiner-gemma4-e4b"),
    ("qwen", "gate", "20261105-034043_phase1_qwen3.5-0.8b-mlx_gate"),
    ("qwen", "memdb", "20261105-174004_phase1_qwen3.5-0.8b-mlx_memdb"),
    ("qwen", "verified", "20261105-225822_phase1_qwen3.5-0.8b-mlx_verified"),
    ("qwen", "errors", "20261106-010306_phase1_qwen3.5-0.8b-mlx_errors"),
    ("qwen", "C8", "20261106-100238_phase1_qwen3.5-0.8b-mlx_refiner-gemma4-e4b_gate_memdb_verified_errors"),
    ("qwen", "Reflexion", "20261106-215702_reflexion_qwen3.5-0.8b-mlx"),
    ("gemma", "C0", "20261107-021257_gemma4-e2b-it-qat"),
    ("gemma", "C1", "20261107-022733_phase1_gemma4-e2b-it-qat"),
    ("gemma", "C8", "20261107-025921_phase1_gemma4-e2b-it-qat_refiner-gemma4-e4b_gate_memdb_verified_errors"),
]

TASK_FIELDS = ["model", "condition", "seed", "phase", "position", "generation", "task_id", "split",
               "success", "steps", "stop_reason", "format_errors", "prompt_tokens", "output_tokens",
               "seconds"]


def read_jsonl(path):
    rows = []
    if os.path.exists(path):
        for line in open(path):
            rows.append(json.loads(line))
    return rows


def seed_dirs(run_dir):
    names = sorted(name for name in os.listdir(run_dir) if name.startswith("seed_"))
    return [(int(name.split("_")[1]), os.path.join(run_dir, name)) for name in names]


def task_row(model, condition, phase, row):
    out = {"model": model, "condition": condition, "phase": phase}
    for field in TASK_FIELDS:
        if field in row and field not in out:
            out[field] = row[field]
    return out


def load_state(folder):
    strategy = open(os.path.join(folder, "strategy.md")).read()
    memory = json.load(open(os.path.join(folder, "memory.json")))
    skills = json.load(open(os.path.join(folder, "skills.json")))
    subagents = json.load(open(os.path.join(folder, "subagents.json")))
    return strategy, memory, skills, subagents


def count_calls(trajectory_path):
    skill_calls = 0
    subagent_calls = 0
    for row in read_jsonl(trajectory_path):
        for call in row.get("tool_calls") or []:
            if call.get("name") == "run_skill":
                skill_calls = skill_calls + 1
            if call.get("name") == "call_subagent":
                subagent_calls = subagent_calls + 1
    return skill_calls, subagent_calls


def server_rows(model, condition, config):
    rows = []
    for key, role in (("server", "worker"), ("refiner_server", "refiner")):
        if key in config:
            rows.append({"model": model, "condition": condition, "role": role,
                         "ollama_version": config[key].get("ollama_version"),
                         "model_digest": config[key].get("model_digest")})
    embed = config.get("memory_db") or {}
    if isinstance(embed, dict) and "embed_server" in embed:
        rows.append({"model": model, "condition": condition, "role": "embedding",
                     "ollama_version": embed["embed_server"].get("ollama_version"),
                     "model_digest": embed["embed_server"].get("model_digest")})
    return rows


os.makedirs(OUT, exist_ok=True)
task_runs = []
edits = []
finals = []
tool_calls = []
servers = []

for model, condition, folder in RUNS_USED:
    run_dir = os.path.join(RUNS, folder)
    config = json.load(open(os.path.join(run_dir, "config.json")))
    servers = servers + server_rows(model, condition, config)

    # The fixed-harness baseline has one results file for all seeds.
    if os.path.exists(os.path.join(run_dir, "results.jsonl")):
        for row in read_jsonl(os.path.join(run_dir, "results.jsonl")):
            task_runs.append(task_row(model, condition, "eval", row))
        continue

    for seed, seed_dir in seed_dirs(run_dir):
        for phase in ("learn", "gate", "eval"):
            for row in read_jsonl(os.path.join(seed_dir, phase + "_results.jsonl")):
                task_runs.append(task_row(model, condition, phase, row))
            skill_calls, subagent_calls = count_calls(os.path.join(seed_dir, phase + "_trajectory.jsonl"))
            tool_calls.append({"model": model, "condition": condition, "seed": seed, "phase": phase,
                               "run_skill": skill_calls, "call_subagent": subagent_calls})

        for row in read_jsonl(os.path.join(seed_dir, "evolution_log.jsonl")):
            gate = row.get("gate") or {}
            counts = row.get("counts") or {}
            edits.append({"model": model, "condition": condition, "seed": seed,
                          "generation": row.get("generation"), "step": row.get("step"),
                          "refiner_seconds": row.get("seconds"),
                          "strategy_chars": counts.get("strategy_chars"),
                          "memory": counts.get("memory"), "skills": counts.get("skills"),
                          "subagents": counts.get("subagents"),
                          "gate_old_passes": gate.get("old_passes"),
                          "gate_new_passes": gate.get("new_passes"),
                          "gate_accepted": gate.get("accepted")})

        reflection_seconds = 0.0
        for row in read_jsonl(os.path.join(seed_dir, "reflections.jsonl")):
            reflection_seconds = reflection_seconds + row.get("stats", {}).get("seconds", 0.0)

        start = load_state(os.path.join(seed_dir, "state_initial"))
        final = load_state(os.path.join(seed_dir, "state_final"))
        finals.append({"model": model, "condition": condition, "seed": seed,
                       "strategy_chars": len(final[0]), "memory": len(final[1]),
                       "skills": len(final[2]), "subagents": len(final[3]),
                       "same_as_start": final == start,
                       "reflection_seconds": round(reflection_seconds, 2)})


def write_csv(name, rows, fields):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print("wrote", os.path.join(OUT, name), len(rows), "rows")


write_csv("task_runs.csv", task_runs, TASK_FIELDS)
write_csv("edits.csv", edits, ["model", "condition", "seed", "generation", "step", "refiner_seconds",
                               "strategy_chars", "memory", "skills", "subagents",
                               "gate_old_passes", "gate_new_passes", "gate_accepted"])
write_csv("final_harness.csv", finals, ["model", "condition", "seed", "strategy_chars", "memory",
                                        "skills", "subagents", "same_as_start", "reflection_seconds"])
write_csv("tool_calls.csv", tool_calls, ["model", "condition", "seed", "phase", "run_skill",
                                         "call_subagent"])
write_csv("servers.csv", servers, ["model", "condition", "role", "ollama_version", "model_digest"])
