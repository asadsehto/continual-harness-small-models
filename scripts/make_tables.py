# Rebuild every table and chart number in the paper from the released data in data/.
#   uv run python -m scripts.make_tables
# It needs only the files in data/ (made by scripts/export_data.py), not the full runs.
#
# Statistics:
#   success rates with Wilson 95% confidence intervals
#   paired comparisons on the same (task, seed): better / worse, exact two-sided McNemar test
#   Holm correction within each family: comparisons with C0, and comparisons with C1, per model

import csv
import math
import os

DATA = "data"
QWEN = ["C0", "C1", "refiner", "gate", "memdb", "verified", "errors", "C8", "Reflexion"]
GEMMA = ["C0", "C1", "C8"]
CONDITIONS = {"qwen": QWEN, "gemma": GEMMA}
NAMES = {"C0": "C0 fixed harness", "C1": "C1 Continual Harness", "refiner": "+ separate refiner",
         "gate": "+ regression gate", "memdb": "+ memory database", "verified": "+ verified skills",
         "errors": "+ structured errors", "C8": "C8 all five", "Reflexion": "Reflexion"}
SEEDS = [0, 1, 2]


def read_csv(name):
    return list(csv.DictReader(open(os.path.join(DATA, name))))


def wilson(passed, total):
    z = 1.96
    p = passed / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(better, worse):
    n = better + worse
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(better, worse) + 1))
    return min(1.0, 2 * tail / 2 ** n)


def holm(p_values):
    # Holm step-down adjustment; returns adjusted p-values in the original order
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def pct(passed, total, ci=False):
    text = str(round(100 * passed / total)) + "%"
    if ci:
        low, high = wilson(passed, total)
        text = text + " (" + str(round(100 * low)) + "-" + str(round(100 * high)) + ")"
    return text


def fmt_p(p):
    return "<0.001" if p < 0.001 else str(round(p, 3))


def print_table(title, header, rows):
    print()
    print("## " + title)
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    for r in [header] + rows:
        print("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(r)))


runs = read_csv("task_runs.csv")
for r in runs:
    r["success"] = r["success"] == "True"
    r["seed"] = int(r["seed"])
    for field in ("steps", "prompt_tokens", "output_tokens", "format_errors"):
        r[field] = int(r[field])
    r["seconds"] = float(r["seconds"])


def select(model, condition, phase, split=None, seed=None):
    out = []
    for r in runs:
        if r["model"] != model or r["condition"] != condition or r["phase"] != phase:
            continue
        if split is not None and r["split"] != split:
            continue
        if seed is not None and r["seed"] != seed:
            continue
        out.append(r)
    return out


def outcomes(model, condition, split=None):
    return {(r["task_id"], r["seed"]): r["success"] for r in select(model, condition, "eval", split)}


def paired(model, a, b, split=None):
    first = outcomes(model, a, split)
    second = outcomes(model, b, split)
    better = sum(1 for k in first if first[k] and not second[k])
    worse = sum(1 for k in first if second[k] and not first[k])
    return better, worse, mcnemar_exact(better, worse)


# Tables 3 and 5: main results, held-out first, with Holm-adjusted p against C0
for model, conditions in CONDITIONS.items():
    others = [c for c in conditions if c != "C0"]
    tests = [paired(model, c, "C0") for c in others]
    adjusted = dict(zip(others, holm([t[2] for t in tests])))
    tests = dict(zip(others, tests))
    rows = []
    for c in conditions:
        held = select(model, c, "eval", "heldout")
        train = select(model, c, "eval", "train")
        every = select(model, c, "eval")
        row = [NAMES[c], pct(sum(r["success"] for r in held), len(held), True),
               pct(sum(r["success"] for r in train), len(train)),
               pct(sum(r["success"] for r in every), len(every), True)]
        if c == "C0":
            row = row + ["-", "-", "-"]
        else:
            better, worse, p = tests[c]
            row = row + [str(better) + " / " + str(worse), fmt_p(p), str(round(adjusted[c], 3))]
        rows.append(row)
    print_table("Main results, " + model + " (held-out first; Holm over comparisons with C0)",
                ["Condition", "Held-out (95% CI)", "Train", "All (95% CI)", "vs C0 better / worse",
                 "p", "p (Holm)"], rows)

# Table 4: comparisons with C1 (qwen: 6 comparisons, as in the paper; gemma: 1)
for model, others in (("qwen", ["refiner", "gate", "memdb", "verified", "errors", "C8"]), ("gemma", ["C8"])):
    tests = [paired(model, c, "C1") for c in others]
    adjusted = holm([t[2] for t in tests])
    rows = [[NAMES[c], str(t[0]) + " / " + str(t[1]), fmt_p(t[2]), str(round(a, 3))]
            for c, t, a in zip(others, tests, adjusted)]
    print_table("Comparisons with C1, " + model, ["Condition", "better / worse", "p", "p (Holm)"], rows)

# Held-out paired comparisons
rows = []
for a, b in (("C1", "C0"), ("C8", "C0"), ("C8", "C1")):
    row = [a + " vs " + b]
    for model in ("qwen", "gemma"):
        better, worse, p = paired(model, a, b, "heldout")
        row = row + [str(better) + " / " + str(worse), fmt_p(p)]
    rows.append(row)
print_table("Held-out paired comparisons", ["Comparison", "qwen b/w", "qwen p", "gemma b/w", "gemma p"], rows)

# Per-seed passes
rows = []
for model, conditions in CONDITIONS.items():
    for c in conditions:
        every = [sum(r["success"] for r in select(model, c, "eval", seed=s)) for s in SEEDS]
        held = [sum(r["success"] for r in select(model, c, "eval", "heldout", s)) for s in SEEDS]
        rows.append([model, NAMES[c], " / ".join(map(str, every)), " / ".join(map(str, held))])
print_table("Passes per seed", ["Model", "Condition", "All of 20 (seed 0/1/2)", "Held-out of 5"], rows)

# Table 6: edits, gate decisions and final harness
edits = read_csv("edits.csv")
finals = read_csv("final_harness.csv")
rows = []
for model, c in (("qwen", "C1"), ("qwen", "gate"), ("qwen", "C8"), ("gemma", "C1"), ("gemma", "C8")):
    mine = [e for e in edits if e["model"] == model and e["condition"] == c]
    gated = [e for e in mine if e["gate_accepted"] != ""]
    kept = sum(1 for e in gated if e["gate_accepted"] == "True")
    ties = sum(1 for e in gated if e["gate_accepted"] == "True" and e["gate_old_passes"] == e["gate_new_passes"])
    fin = [f for f in finals if f["model"] == model and f["condition"] == c]
    chars = [int(f["strategy_chars"]) for f in fin]
    skills = [int(f["skills"]) for f in fin]
    subs = [int(f["subagents"]) for f in fin]
    same = sum(1 for f in fin if f["same_as_start"] == "True")
    gate_text = (str(kept) + " / " + str(len(gated) - kept) + " (ties kept: " + str(ties) + ")") if gated else "-"
    rows.append([model, NAMES[c], len(mine), gate_text,
                 "strategy " + str(min(chars)) + "-" + str(max(chars)) + " chars, " + str(min(skills)) + "-" +
                 str(max(skills)) + " skills, " + str(min(subs)) + "-" + str(max(subs)) + " sub-agents; " +
                 str(same) + " of 3 seeds same as start"])
print_table("Edits", ["Model", "Condition", "Refiner runs", "Gate kept / undone", "Final harness"], rows)

# Table 7: forgetting
rows = []
for model, conditions in CONDITIONS.items():
    for c in conditions:
        if c == "C0":
            continue
        at_eval = outcomes(model, c)
        learned = [r for r in select(model, c, "learn") if r["success"]]
        forgotten = [r for r in learned if not at_eval[(r["task_id"], r["seed"])]]
        rows.append([model, NAMES[c], str(len(learned)) + " / 45",
                     str(len(forgotten)) + " (" + pct(len(forgotten), len(learned)) + ")"])
print_table("Forgetting", ["Model", "Condition", "Passed while learning", "Of those, failed at eval"], rows)

# Table 8: behaviour and cost at evaluation; learning time per seed includes refiner and gate
rows = []
for model, conditions in CONDITIONS.items():
    for c in conditions:
        ev = select(model, c, "eval")
        learn_seconds = sum(r["seconds"] for r in select(model, c, "learn") + select(model, c, "gate"))
        learn_seconds = learn_seconds + sum(float(e["refiner_seconds"] or 0) for e in edits
                                            if e["model"] == model and e["condition"] == c)
        learn_seconds = learn_seconds + sum(float(f["reflection_seconds"]) for f in finals
                                            if f["model"] == model and f["condition"] == c)
        rows.append([model, NAMES[c], round(sum(r["steps"] for r in ev) / len(ev), 1),
                     str(sum(1 for r in ev if r["stop_reason"] == "max_steps")) + " / " + str(len(ev)),
                     round(sum(r["seconds"] for r in ev) / len(ev), 1),
                     round(sum(r["prompt_tokens"] for r in ev) / len(ev)),
                     "-" if c == "C0" else round(learn_seconds / 60 / 3, 1),
                     sum(r["format_errors"] for r in ev)])
print_table("Behaviour and cost at evaluation",
            ["Model", "Condition", "Steps/task", "Hit 15-step limit", "Seconds/task", "Prompt tokens/task",
             "Learning min/seed", "Unknown-tool calls"], rows)

# Table 9: skill and sub-agent calls, all phases
calls = read_csv("tool_calls.csv")
rows = []
for model, conditions in CONDITIONS.items():
    for c in conditions:
        if c in ("C0", "Reflexion"):
            continue
        mine = [x for x in calls if x["model"] == model and x["condition"] == c]
        rows.append([model, NAMES[c], sum(int(x["run_skill"]) for x in mine),
                     sum(int(x["call_subagent"]) for x in mine)])
print_table("Skill and sub-agent calls (all phases)", ["Model", "Condition", "run_skill", "call_subagent"], rows)

# Table 11: passes per task (of 3 seeds)
tasks = []
for r in select("qwen", "C0", "eval", seed=0):
    tasks.append(r["task_id"])
header = ["Task"] + ["qwen " + c for c in QWEN] + ["gemma " + c for c in GEMMA]
rows = []
for t in tasks + ["Total"]:
    row = [t]
    for model, conditions in CONDITIONS.items():
        for c in conditions:
            ev = select(model, c, "eval")
            row.append(sum(r["success"] for r in ev if t == "Total" or r["task_id"] == t))
    rows.append(row)
print_table("Passes per task (of 3 seeds)", header, rows)

# Determinism check: before the first edit, conditions without retrieval or structured errors use
# the same harness and seed as C0, so their runs can be compared with C0 one to one.
print()
print("## Determinism check (first-generation learning runs against C0)")
for model, same_start in (("qwen", ["C1", "refiner", "gate", "verified", "memdb"]), ("gemma", ["C1"])):
    base = {(r["task_id"], r["seed"]): r for r in select(model, "C0", "eval")}
    identical = 0
    drifted = 0
    outcome_changed = 0
    for c in same_start:
        for r in select(model, c, "learn"):
            if r["generation"] != "0":
                continue
            b = base[(r["task_id"], r["seed"])]
            if all(r[f] == b[f] for f in ("success", "steps", "prompt_tokens", "output_tokens")):
                identical = identical + 1
            else:
                drifted = drifted + 1
                if r["success"] != b["success"]:
                    outcome_changed = outcome_changed + 1
    print(" ", model, "identical:", identical, "drifted:", drifted, "of which pass/fail changed:", outcome_changed)
