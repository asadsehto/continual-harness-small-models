# Reflexion-style memory baseline (Shinn et al., 2023, "Reflexion: Language Agents with
# Verbal Reinforcement Learning"), adapted to a sequence of different tasks:
#   - after a FAILED task, the same model writes a short self-reflection on its trajectory
#   - only the last MAX_REFLECTIONS reflections are kept (Reflexion keeps 1-3)
#   - they are shown in the system prompt (as memory) for the next tasks
#   - no Refiner: strategy, skills and sub-agents never change

from harness.model import chat
from harness.refiner import format_steps
from harness.state import MAX_MEMORY_CHARS

MAX_REFLECTIONS = 3

PROMPT = """You are an AI agent that does computer tasks (coding, files, shell) inside a task folder.
You just FAILED a task. Below are the task and your steps.

## Task
{instruction}

## Your steps
{steps}

In a few sentences, diagnose a possible reason for the failure and write a new, concise, high-level
plan that avoids the same mistake. Write it so it also helps on other, similar tasks.
Reply with the reflection only."""


def reflect(model, instruction, steps, seed, temperature):
    prompt = PROMPT.format(instruction=instruction, steps=format_steps(steps))
    message, stats = chat(model, [{"role": "user", "content": prompt}], None, seed, temperature)
    text = (message.content or "").strip()[:MAX_MEMORY_CHARS]
    return text, stats


def add_reflection(state, task_id, text):
    # keep only the last MAX_REFLECTIONS; titles say which task they came from
    if not text:
        return
    number = 1
    for entry in state["memory"]:
        number = max(number, int(entry["id"].split("_")[1]) + 1)
    state["memory"].append({"id": "refl_" + str(number), "title": "Reflection after failing " + task_id, "content": text, "importance": 3})
    while len(state["memory"]) > MAX_REFLECTIONS:
        state["memory"].pop(0)
