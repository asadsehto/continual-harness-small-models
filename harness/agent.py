# The agent loop.
# Each step: send messages to the model -> if it asks for tools, run them and send the
# results back -> stop when the model answers without a tool call or max_steps is reached.
# Every step is written to the trajectory log, one JSON line per step.
#
# state=None  -> Phase 0 baseline (C0): fixed prompt, 3 tools.
# state=dict  -> Phase 1 (C1): prompt built from the editable harness, plus run_skill and
#                call_subagent when the harness has skills / sub-agents.

import json
import time

from harness.errors import structure
from harness.model import chat
from harness.state import build_system_prompt
from harness.tools import SUBAGENT_ALLOWED_TOOLS, call_tool, schemas_for

SYSTEM_PROMPT = """You are an agent working inside a task folder on a computer.
You can use these tools: run_shell, read_file, write_file.
Do the task step by step. Check your work (for example run the tests) before you finish.
When the task is complete, reply with a short message and no tool call."""


def run_subagent_loop(model, sub, task_text, workdir, seed, temperature, state, structured_errors=False):
    # A sub-agent is a smaller agent loop with its own instructions and an allowed tool list.
    messages = [
        {"role": "system", "content": sub["system_instructions"] + "\nWhen done, reply with a short answer and no tool call."},
        {"role": "user", "content": task_text},
    ]
    schemas = []
    for schema in schemas_for(state):
        name = schema["function"]["name"]
        if name in sub["available_tools"] and name in SUBAGENT_ALLOWED_TOOLS:
            schemas.append(schema)
    calls = []
    for turn in range(sub["max_turns"]):
        message, stats = chat(model, messages, schemas, seed, temperature)
        messages.append(message)
        if not message.tool_calls:
            return "sub-agent " + sub["name"] + " answered: " + (message.content or "")[:1000] + "\ntool calls it made: " + str(calls)[:1000]
        for call in message.tool_calls:
            name = call.function.name
            if name not in sub["available_tools"] or name not in SUBAGENT_ALLOWED_TOOLS:
                result = "ERROR: tool not allowed for this sub-agent: " + str(name)
            else:
                result = call_tool(workdir, name, call.function.arguments, state, None)
            if structured_errors:
                result = structure(result)
            messages.append({"role": "tool", "tool_name": name, "content": result})
            calls.append(name)
    return "sub-agent " + sub["name"] + " stopped after " + str(sub["max_turns"]) + " turns without answering. tool calls: " + str(calls)[:1000]


# structured_errors=True -> technique #5: failed tool calls return a typed error with a hint (harness/errors.py)
def run_agent(model, instruction, workdir, max_steps, seed, temperature, log_file, task_id, state=None, structured_errors=False):
    if state is None:
        system_prompt = SYSTEM_PROMPT
    else:
        system_prompt = build_system_prompt(state)
    tools = schemas_for(state)

    def run_subagent(name, task_text):
        for sub in state["subagents"]:
            if sub["name"] == name:
                return run_subagent_loop(model, sub, task_text, workdir, seed, temperature, state, structured_errors)
        names = [sub["name"] for sub in state["subagents"]]
        return "ERROR: no sub-agent named " + name + ". Sub-agents: " + str(names)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction},
    ]

    steps_used = 0
    format_errors = 0
    cut_replies = 0
    total_prompt_tokens = 0
    total_output_tokens = 0
    start = time.time()
    stop_reason = "max_steps"

    for step in range(max_steps):
        steps_used = step + 1
        message, stats = chat(model, messages, tools, seed, temperature)
        total_prompt_tokens = total_prompt_tokens + (stats["prompt_tokens"] or 0)
        total_output_tokens = total_output_tokens + (stats["output_tokens"] or 0)
        if stats["cut_at_max_tokens"]:
            cut_replies = cut_replies + 1

        messages.append(message)

        tool_calls = []
        if message.tool_calls:
            for call in message.tool_calls:
                name = call.function.name
                args = call.function.arguments
                if state is None:
                    result = call_tool(workdir, name, args)
                else:
                    result = call_tool(workdir, name, args, state, run_subagent)
                if result.startswith("ERROR: unknown tool"):
                    format_errors = format_errors + 1
                if structured_errors:
                    result = structure(result)
                messages.append({"role": "tool", "tool_name": name, "content": result})
                tool_calls.append({"name": name, "args": args, "result": result})

        record = {
            "task_id": task_id,
            "step": steps_used,
            "model": model,
            "text": message.content,
            "tool_calls": tool_calls,
            "stats": stats,
        }
        log_file.write(json.dumps(record, default=str) + "\n")
        log_file.flush()

        if not message.tool_calls:
            stop_reason = "answered"
            break

    summary = {
        "steps": steps_used,
        "stop_reason": stop_reason,
        "format_errors": format_errors,
        "cut_replies": cut_replies,
        "prompt_tokens": total_prompt_tokens,
        "output_tokens": total_output_tokens,
        "seconds": round(time.time() - start, 2),
    }
    return summary
