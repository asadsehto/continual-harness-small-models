# The Refiner (condition C1): a port of Continual Harness's HarnessEvolver
# (agents/utils/harness_evolver.py + agents/utils/prompt_optimizer.py, MIT license,
# github.com/sethkarten/continual-harness) to our tasks and Ollama.
#
# Same as the original:
#   - schedule: no evolution before step 25; then every 25 steps up to step 200, then every 100
#   - window: the last 25 steps for prompt/sub-agents/skills, the last 20 for memory
#   - four passes in order: prompt, sub-agents, skills, memory; each in its own try/except
#   - the SAME model that does the tasks also does the refining
#   - prompt pass returns the whole new strategy as markdown; the other passes return JSON
#   - JSON is read with the same method (strip ``` fences, json.loads, else the first {...} block)
#   - every edit is kept: there is no test of whether an edit helps (that is a Phase 2 technique)
# Changes for our setting (written down in docs/continual_harness_notes.md):
#   - game words replaced by task words; the trajectory shows task results (PASS/FAIL), which
#     play the role of game progress in the original
#   - skill code runs as a Python file in the sandbox, not with tools['...'] game calls
#   - a skill's code is kept when the skill is added (the original only saves code on update)
#   - sub-agent fields reduced to name, description, instructions, max_turns, tools
#   - size caps on everything (RAM and time), and every edit is logged with before and after

import json
import time

from harness.model import ModelError, chat
from harness.state import (
    FIXED_PROMPT, MAX_DESCRIPTION_CHARS, MAX_MEMORY_CHARS, MAX_MEMORY_ENTRIES, MAX_NAME_CHARS,
    MAX_SKILL_CODE_CHARS, MAX_SKILLS, MAX_STRATEGY_CHARS, MAX_SUBAGENT_INSTRUCTION_CHARS,
    MAX_SUBAGENT_TURNS, MAX_SUBAGENTS, next_id,
)
from harness.tools import SUBAGENT_ALLOWED_TOOLS

MIN_WARMUP_STEPS = 25
WINDOW_STEPS = 25
MEMORY_WINDOW_STEPS = 20
MAX_RESPONSE_LOG_CHARS = 6000


def evolution_points_crossed(previous_step, current_step):
    # The original checks `step % 25 == 0` after every step. We only edit between tasks,
    # so we count how many of those points were passed during the last task.
    points = 0
    for step in range(previous_step + 1, current_step + 1):
        if step < MIN_WARMUP_STEPS:
            continue
        if step <= 200 and step % 25 == 0:
            points = points + 1
        elif step > 200 and step % 100 == 0:
            points = points + 1
    return points


# ---------------- formatting the recent steps (port of _format_trajectories_for_analysis) ----

def format_steps(steps):
    text = ""
    for record in steps:
        if "outcome" in record:
            text = text + "\n>>> TASK " + record["task_id"] + " FINISHED: " + record["outcome"] + "\n"
            continue
        text = text + "\n### Step " + str(record["global_step"]) + " (task " + record["task_id"] + ")\n"
        text = text + "**Reasoning:** " + (record.get("text") or "N/A")[:300] + "\n"
        if record["tool_calls"]:
            names = [call["name"] for call in record["tool_calls"]]
            text = text + "**Action:** " + ", ".join(names) + "\n"
            for call in record["tool_calls"]:
                text = text + "  - `" + call["name"] + "` args: " + json.dumps(call["args"], default=str)[:300] + "\n"
                text = text + "  - result: " + str(call["result"])[:300] + "\n"
        else:
            text = text + "**Action:** final answer (no tool call)\n"
    return text


def is_failure(result):
    if result.startswith("ERROR"):
        return True
    if result.startswith("exit code: ") and not result.startswith("exit code: 0"):
        return True
    return False


def extract_tool_failures(steps):
    lines = []
    for record in steps:
        if "outcome" in record:
            continue
        for call in record["tool_calls"]:
            if is_failure(str(call["result"])):
                lines.append("- Step " + str(record["global_step"]) + ": `" + call["name"] + "` args=" + json.dumps(call["args"], default=str)[:200] + " => " + str(call["result"])[:300])
    if not lines:
        return ""
    return "## Tool Failures Detected\n" + "\n".join(lines)


def parse_json_response(response):
    text = response.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except ValueError:
        pass
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            value = json.loads(text[start:end])
            if isinstance(value, dict):
                return value
        except ValueError:
            pass
    return None


def as_list(value):
    if isinstance(value, list):
        return value
    return []


def short_text(value, limit):
    if not isinstance(value, str):
        return None
    value = value.strip()
    if value == "" or len(value) > limit:
        return None
    return value


def ask(model, prompt, seed, temperature):
    message, stats = chat(model, [{"role": "user", "content": prompt}], None, seed, temperature)
    return message.content or "", stats


# ---------------- pass 1: prompt (port of PromptOptimizer.optimize_prompt) ----------------

def evolve_prompt(model, state, steps, seed, temperature):
    prompt = """You are a prompt optimization expert. Your task is to improve an AI agent's strategic guidance prompt based on its recent performance on computer tasks (coding, files, shell).

## Main Agent's System Prompt (FIXED - Cannot Be Changed):
The main agent receives this every step. Do not repeat it in the strategy. Strategy and working habits belong in the strategy you are editing.

""" + FIXED_PROMPT + """

---
## Current Strategy (the part you can improve):

""" + state["strategy"] + """

## Recent Agent Trajectories (Last """ + str(len(steps)) + """ records):
""" + format_steps(steps) + """

## Your Task:
Analyze the agent's recent performance and create an IMPROVED strategy that:

1. **Addresses observed failures** - If the agent made mistakes, add specific guidance to prevent them
2. **Reinforces successful patterns** - If certain strategies worked well, emphasize them
3. **Adds learned lessons** - Include insights derived from trajectory analysis

Look for: repeated failures (loops, wrong tool usage, working outside the task folder), successful strategies, tasks that PASSED vs FAILED, tool usage patterns, not adapting when stuck.

## Output Format:
Provide the complete improved strategy as markdown (at most """ + str(MAX_STRATEGY_CHARS) + """ characters). Make targeted improvements, keeping the elements that are working.

IMPROVED STRATEGY:
"""
    response, stats = ask(model, prompt, seed, temperature)
    new_strategy = response.strip()
    if new_strategy.startswith("```markdown"):
        new_strategy = new_strategy[11:]
    if new_strategy.startswith("```"):
        new_strategy = new_strategy[3:]
    if new_strategy.endswith("```"):
        new_strategy = new_strategy[:-3]
    new_strategy = new_strategy.strip()

    log = {"response": response[:MAX_RESPONSE_LOG_CHARS], "stats": stats, "applied": [], "rejected": []}
    if new_strategy == "":
        log["rejected"].append("empty strategy")
    elif len(new_strategy) > MAX_STRATEGY_CHARS:
        log["rejected"].append("strategy too long: " + str(len(new_strategy)) + " chars")
    else:
        log["applied"].append({"field": "strategy", "before": state["strategy"], "after": new_strategy})
        state["strategy"] = new_strategy
    return log


# ---------------- pass 2: sub-agents (port of _evolve_subagents) ----------------

def evolve_subagents(model, state, steps, seed, temperature):
    overview = json.dumps(state["subagents"], indent=1) if state["subagents"] else "(none yet)"
    prompt = """You are a harness evolution system for an AI agent that does computer tasks (coding, files, shell) inside a task folder.

Your job: analyze recent trajectories and recommend changes to the agent's sub-agent library.

## Current Sub-agent Registry
""" + overview + """

## Recent Trajectories (last """ + str(len(steps)) + """ records)
""" + format_steps(steps) + """

""" + extract_tool_failures(steps) + """

## Analysis Tasks

1. **Identify missing sub-agents**: Look for repeated multi-step patterns the agent does manually that would benefit from a dedicated sub-agent (for example: a test-runner/debugger, a file organizer, a checker that verifies the result before finishing).
2. **Evaluate existing sub-agents**: If any were used, check if they failed or could be improved.
3. **Retire ineffective sub-agents**: If a sub-agent was used with poor results, recommend removal.

## Output Format

Respond with ONLY a JSON object (no markdown fences):
{
  "analysis": "Brief summary of what you observed",
  "create": [
    {
      "name": "string",
      "description": "string",
      "max_turns": 6,
      "available_tools": ["run_shell", "read_file", "write_file"],
      "system_instructions": "Detailed instructions for the sub-agent"
    }
  ],
  "update": [
    {"id": "sa_X", "system_instructions": "improved instructions", "description": "optional"}
  ],
  "retire": ["sa_X"]
}

Only include sections with actual recommendations. Empty arrays are fine.
Available tools the sub-agent can use: """ + str(SUBAGENT_ALLOWED_TOOLS) + "\n"

    response, stats = ask(model, prompt, seed, temperature)
    log = {"response": response[:MAX_RESPONSE_LOG_CHARS], "stats": stats, "applied": [], "rejected": []}
    data = parse_json_response(response)
    if data is None:
        log["rejected"].append("failed_to_parse_response")
        return log
    log["analysis"] = str(data.get("analysis", ""))[:1000]

    for spec in as_list(data.get("create")):
        if not isinstance(spec, dict):
            log["rejected"].append({"create": str(spec)[:200], "reason": "not an object"})
            continue
        name = short_text(spec.get("name"), MAX_NAME_CHARS)
        instructions = short_text(spec.get("system_instructions"), MAX_SUBAGENT_INSTRUCTION_CHARS)
        description = short_text(spec.get("description"), MAX_DESCRIPTION_CHARS) or ""
        if name is None or instructions is None:
            log["rejected"].append({"create": str(spec)[:300], "reason": "missing or too long name/system_instructions"})
            continue
        if len(state["subagents"]) >= MAX_SUBAGENTS:
            log["rejected"].append({"create": name, "reason": "cap of " + str(MAX_SUBAGENTS) + " sub-agents reached"})
            continue
        if name in [sub["name"] for sub in state["subagents"]]:
            log["rejected"].append({"create": name, "reason": "name already exists"})
            continue
        tools = [t for t in as_list(spec.get("available_tools")) if t in SUBAGENT_ALLOWED_TOOLS]
        if not tools:
            tools = ["run_shell", "read_file", "write_file"]
        turns = spec.get("max_turns", MAX_SUBAGENT_TURNS)
        if not isinstance(turns, int) or turns < 1:
            turns = MAX_SUBAGENT_TURNS
        entry = {
            "id": next_id(state["subagents"], "sa_"),
            "name": name,
            "description": description,
            "system_instructions": instructions,
            "max_turns": min(turns, MAX_SUBAGENT_TURNS),
            "available_tools": tools,
        }
        state["subagents"].append(entry)
        log["applied"].append({"field": "subagents", "op": "create", "before": None, "after": entry})

    for upd in as_list(data.get("update")):
        if not isinstance(upd, dict):
            continue
        target = None
        for sub in state["subagents"]:
            if sub["id"] == upd.get("id"):
                target = sub
        if target is None:
            log["rejected"].append({"update": str(upd)[:200], "reason": "unknown id"})
            continue
        before = dict(target)
        instructions = short_text(upd.get("system_instructions"), MAX_SUBAGENT_INSTRUCTION_CHARS)
        description = short_text(upd.get("description"), MAX_DESCRIPTION_CHARS)
        if instructions:
            target["system_instructions"] = instructions
        if description:
            target["description"] = description
        if before != target:
            log["applied"].append({"field": "subagents", "op": "update", "before": before, "after": dict(target)})

    for sid in as_list(data.get("retire")):
        for sub in list(state["subagents"]):
            if sub["id"] == sid:
                state["subagents"].remove(sub)
                log["applied"].append({"field": "subagents", "op": "retire", "before": sub, "after": None})
    return log


# ---------------- pass 3: skills (port of _evolve_skills) ----------------

def evolve_skills(model, state, steps, seed, temperature):
    overview = json.dumps(state["skills"], indent=1) if state["skills"] else "(none yet)"
    inline_code_calls = 0
    run_skill_calls = 0
    for record in steps:
        for call in record.get("tool_calls", []):
            if call["name"] == "run_shell" and ("python -c" in str(call["args"]) or "python3 -c" in str(call["args"])):
                inline_code_calls = inline_code_calls + 1
            if call["name"] == "run_skill":
                run_skill_calls = run_skill_calls + 1
    warning = ""
    if inline_code_calls >= 3 and run_skill_calls == 0:
        warning = "\n## CRITICAL ANTIPATTERN DETECTED\nThe agent ran inline Python " + str(inline_code_calls) + " times but run_skill 0 times. It is writing disposable scripts instead of reusable skills. You MUST create executable skills from these patterns.\n"

    prompt = """You are a harness evolution system analyzing an AI agent's recent work on computer tasks (coding, files, shell) inside a task folder.

Your job: identify reusable behavioral patterns (skills) from successful actions and evaluate existing skills.

## Current Skill Library
""" + overview + """

## Recent Trajectories (last """ + str(len(steps)) + """ records)
""" + format_steps(steps) + """

""" + extract_tool_failures(steps) + warning + """

## Skill Code API (MUST follow this exactly)

Skill code runs as inline Python (NOT a function definition), with the task folder as the current directory. It has:
- `args` — dict of arguments passed by the caller (e.g. `args['path']`)
- `result` — set this variable to return data
- Modules already imported: json, os, sys, re, math, csv, shutil, glob, subprocess
Do NOT write `def skill_name():` — write inline code that reads `args` and sets `result`.

## Analysis Tasks

1. **Identify underperforming skills**: find skills that failed or errored; recommend a rewrite with full replacement `code`.
2. **Fix broken skills**: if skills error, fix the code.
3. **Record new skills**: look for successful action sequences. Include `code` for skills that automate multi-step actions.
4. **Update effectiveness ratings** based on outcomes.

## Output Format

Respond with ONLY a JSON object (no markdown fences):
{
  "analysis": "Brief summary",
  "add": [
    {"name": "string", "description": "What the skill does and which args it needs", "code": "Python code", "effectiveness": "low|medium|high"}
  ],
  "update": [
    {"id": "skill_X", "effectiveness": "low|medium|high", "description": "optional", "code": "optional FULL replacement code"}
  ]
}
"""
    response, stats = ask(model, prompt, seed, temperature)
    log = {"response": response[:MAX_RESPONSE_LOG_CHARS], "stats": stats, "applied": [], "rejected": []}
    data = parse_json_response(response)
    if data is None:
        log["rejected"].append("failed_to_parse_response")
        return log
    log["analysis"] = str(data.get("analysis", ""))[:1000]

    for spec in as_list(data.get("add")):
        if not isinstance(spec, dict):
            log["rejected"].append({"add": str(spec)[:200], "reason": "not an object"})
            continue
        name = short_text(spec.get("name"), MAX_NAME_CHARS)
        description = short_text(spec.get("description"), MAX_DESCRIPTION_CHARS)
        code = spec.get("code")
        if code is not None and short_text(code, MAX_SKILL_CODE_CHARS) is None:
            log["rejected"].append({"add": str(name), "reason": "code is not text or longer than " + str(MAX_SKILL_CODE_CHARS) + " chars"})
            continue
        if name is None or description is None:
            log["rejected"].append({"add": str(spec)[:300], "reason": "missing or too long name/description"})
            continue
        if len(state["skills"]) >= MAX_SKILLS:
            log["rejected"].append({"add": name, "reason": "cap of " + str(MAX_SKILLS) + " skills reached"})
            continue
        if name in [skill["name"] for skill in state["skills"]]:
            log["rejected"].append({"add": name, "reason": "name already exists"})
            continue
        entry = {
            "id": next_id(state["skills"], "skill_"),
            "name": name,
            "description": description,
            "code": code or "",
            "effectiveness": spec.get("effectiveness") if spec.get("effectiveness") in ["low", "medium", "high"] else "medium",
        }
        state["skills"].append(entry)
        log["applied"].append({"field": "skills", "op": "add", "before": None, "after": entry})

    for upd in as_list(data.get("update")):
        if not isinstance(upd, dict):
            continue
        target = None
        for skill in state["skills"]:
            if skill["id"] == upd.get("id"):
                target = skill
        if target is None:
            log["rejected"].append({"update": str(upd)[:200], "reason": "unknown id"})
            continue
        before = dict(target)
        if upd.get("effectiveness") in ["low", "medium", "high"]:
            target["effectiveness"] = upd["effectiveness"]
        description = short_text(upd.get("description"), MAX_DESCRIPTION_CHARS)
        if description:
            target["description"] = description
        code = short_text(upd.get("code"), MAX_SKILL_CODE_CHARS)
        if code:
            target["code"] = code
        if before != target:
            log["applied"].append({"field": "skills", "op": "update", "before": before, "after": dict(target)})
    return log


# ---------------- pass 4: memory (port of _evolve_memory) ----------------

def evolve_memory(model, state, steps, seed, temperature):
    overview = json.dumps(state["memory"], indent=1) if state["memory"] else "(empty)"
    prompt = """You are a memory curator for an AI agent that does computer tasks (coding, files, shell) inside a task folder.

Your job: review the agent's memory store and recent trajectories, then recommend targeted improvements. Memory entries are shown to the agent at the start of every task, so they should be general lessons, not facts about one task.

## Current Memory
""" + overview + """

## Recent Trajectories (last """ + str(len(steps)) + """ records)
""" + format_steps(steps) + """

## Curation Tasks

1. **Fill knowledge gaps**: If the agent made a mistake or found something important and there is no memory entry for it, add one.
2. **Update stale entries**: If a memory entry is wrong according to the trajectories, update it.
3. **Rebalance importance**: 1 (minor) to 5 (critical).

Keep recommendations minimal (max 3-5 changes). Do NOT duplicate existing entries.

## Output Format

Respond with ONLY a JSON object (no markdown fences):
{
  "analysis": "Brief summary of memory state",
  "add": [{"title": "string", "content": "string", "importance": 3}],
  "update": [{"id": "mem_X", "content": "optional updated content", "importance": 2}]
}
"""
    response, stats = ask(model, prompt, seed, temperature)
    log = {"response": response[:MAX_RESPONSE_LOG_CHARS], "stats": stats, "applied": [], "rejected": []}
    data = parse_json_response(response)
    if data is None:
        log["rejected"].append("failed_to_parse_response")
        return log
    log["analysis"] = str(data.get("analysis", ""))[:1000]

    for spec in as_list(data.get("add")):
        if not isinstance(spec, dict):
            log["rejected"].append({"add": str(spec)[:200], "reason": "not an object"})
            continue
        title = short_text(spec.get("title"), MAX_NAME_CHARS)
        content = short_text(spec.get("content"), MAX_MEMORY_CHARS)
        if title is None or content is None:
            log["rejected"].append({"add": str(spec)[:300], "reason": "missing or too long title/content"})
            continue
        if len(state["memory"]) >= MAX_MEMORY_ENTRIES:
            log["rejected"].append({"add": title, "reason": "cap of " + str(MAX_MEMORY_ENTRIES) + " entries reached"})
            continue
        importance = spec.get("importance", 3)
        if not isinstance(importance, int) or importance < 1 or importance > 5:
            importance = 3
        entry = {"id": next_id(state["memory"], "mem_"), "title": title, "content": content, "importance": importance}
        state["memory"].append(entry)
        log["applied"].append({"field": "memory", "op": "add", "before": None, "after": entry})

    for upd in as_list(data.get("update")):
        if not isinstance(upd, dict):
            continue
        target = None
        for entry in state["memory"]:
            if entry["id"] == upd.get("id"):
                target = entry
        if target is None:
            log["rejected"].append({"update": str(upd)[:200], "reason": "unknown id"})
            continue
        before = dict(target)
        content = short_text(upd.get("content"), MAX_MEMORY_CHARS)
        if content:
            target["content"] = content
        importance = upd.get("importance")
        if isinstance(importance, int) and 1 <= importance <= 5:
            target["importance"] = importance
        if before != target:
            log["applied"].append({"field": "memory", "op": "update", "before": before, "after": dict(target)})
    return log


# ---------------- evolve: the four passes in order (port of HarnessEvolver.evolve) ----------------

def last_steps(history, count):
    # The last `count` agent steps, plus the task-result lines that fall between them.
    chosen = []
    steps = 0
    for record in reversed(history):
        if "outcome" not in record:
            if steps == count:
                break
            steps = steps + 1
        chosen.append(record)
    chosen.reverse()
    return chosen


def evolve(model, state, history, current_step, generation, seed, temperature):
    window = last_steps(history, WINDOW_STEPS)
    memory_window = last_steps(history, MEMORY_WINDOW_STEPS)
    start = time.time()
    results = {"generation": generation, "step": current_step, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    passes = [
        ("prompt", evolve_prompt, window),
        ("subagents", evolve_subagents, window),
        ("skills", evolve_skills, window),
        ("memory", evolve_memory, memory_window),
    ]
    for name, function, steps in passes:
        try:
            results[name] = function(model, state, steps, seed, temperature)
        except ModelError:
            # server or connection problem: stop the run, do not log it as a bad Refiner pass
            raise
        except Exception as error:
            results[name] = {"error": repr(error)[:500]}
    results["seconds"] = round(time.time() - start, 1)
    results["counts"] = {
        "strategy_chars": len(state["strategy"]),
        "memory": len(state["memory"]),
        "skills": len(state["skills"]),
        "subagents": len(state["subagents"]),
    }
    return results
