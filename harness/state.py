# The editable harness (Phase 1, Continual Harness). It lives in a folder:
#   strategy.md      the editable "base prompt" (strategy). The Refiner rewrites it.
#   memory.json      list of {id, title, content, importance}
#   skills.json      list of {id, name, description, code, effectiveness}
#   subagents.json   list of {id, name, description, system_instructions, max_turns, available_tools}
# The FIXED part of the system prompt (tools and hard rules) is in this file and never edited,
# same split as the original (fixed system prompt + optimizable base prompt).
#
# Caps keep the prompt (and so RAM and time) from growing without limit.

import json
import os

FIXED_PROMPT = """You are an agent working inside a task folder on a computer.
You can use these tools: run_shell, read_file, write_file.
When the task is complete, reply with a short message and no tool call."""

INITIAL_STRATEGY = """Do the task step by step. Check your work (for example run the tests) before you finish."""

MAX_STRATEGY_CHARS = 4000
MAX_MEMORY_ENTRIES = 20
MAX_MEMORY_CHARS = 500
MAX_SKILLS = 10
MAX_SKILL_CODE_CHARS = 3000
MAX_SUBAGENTS = 5
MAX_SUBAGENT_INSTRUCTION_CHARS = 2000
MAX_SUBAGENT_TURNS = 8
MAX_NAME_CHARS = 60
MAX_DESCRIPTION_CHARS = 300


def create_state(folder):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "strategy.md"), "w") as f:
        f.write(INITIAL_STRATEGY)
    for name in ["memory.json", "skills.json", "subagents.json"]:
        with open(os.path.join(folder, name), "w") as f:
            json.dump([], f)


def load_state(folder):
    state = {}
    with open(os.path.join(folder, "strategy.md")) as f:
        state["strategy"] = f.read()
    with open(os.path.join(folder, "memory.json")) as f:
        state["memory"] = json.load(f)
    with open(os.path.join(folder, "skills.json")) as f:
        state["skills"] = json.load(f)
    with open(os.path.join(folder, "subagents.json")) as f:
        state["subagents"] = json.load(f)
    return state


def save_state(folder, state):
    with open(os.path.join(folder, "strategy.md"), "w") as f:
        f.write(state["strategy"])
    with open(os.path.join(folder, "memory.json"), "w") as f:
        json.dump(state["memory"], f, indent=2)
    with open(os.path.join(folder, "skills.json"), "w") as f:
        json.dump(state["skills"], f, indent=2)
    with open(os.path.join(folder, "subagents.json"), "w") as f:
        json.dump(state["subagents"], f, indent=2)


# The system prompt the agent sees. With an empty harness this is the same text as the
# Phase 0 baseline prompt, so before the first edit, C1 behaves exactly like C0.
def build_system_prompt(state):
    lines = FIXED_PROMPT.split("\n")
    text = lines[0] + "\n" + lines[1] + "\n" + state["strategy"].strip() + "\n" + lines[2]

    if len(state["memory"]) > 0:
        text = text + "\n\n## Memory (lessons from earlier tasks)\n"
        entries = sorted(state["memory"], key=lambda m: -m.get("importance", 3))
        for entry in entries:
            text = text + "- " + entry["title"] + ": " + entry["content"] + "\n"

    if len(state["skills"]) > 0:
        text = text + "\n## Skills (run with run_skill)\n"
        for skill in state["skills"]:
            kind = "runnable" if skill.get("code") else "description only"
            text = text + "- " + skill["name"] + " (" + kind + "): " + skill["description"] + "\n"

    if len(state["subagents"]) > 0:
        text = text + "\n## Sub-agents (use with call_subagent)\n"
        for sub in state["subagents"]:
            text = text + "- " + sub["name"] + ": " + sub["description"] + "\n"

    return text


def next_id(items, prefix):
    number = 1
    for item in items:
        if item["id"].startswith(prefix):
            value = item["id"][len(prefix):]
            if value.isdigit() and int(value) >= number:
                number = int(value) + 1
    return prefix + str(number)
