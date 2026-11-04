# The baseline agent loop (condition C0: static harness, nothing is learned).
# Each step: send messages to the model -> if it asks for tools, run them and send the
# results back -> stop when the model answers without a tool call or max_steps is reached.
# Every step is written to the trajectory log, one JSON line per step.

import json
import time

from harness.model import chat
from harness.tools import TOOL_SCHEMAS, call_tool

SYSTEM_PROMPT = """You are an agent working inside a task folder on a computer.
You can use these tools: run_shell, read_file, write_file.
Do the task step by step. Check your work (for example run the tests) before you finish.
When the task is complete, reply with a short message and no tool call."""


def run_agent(model, instruction, workdir, max_steps, seed, temperature, log_file, task_id):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]

    steps_used = 0
    format_errors = 0
    total_prompt_tokens = 0
    total_output_tokens = 0
    start = time.time()
    stop_reason = "max_steps"

    for step in range(max_steps):
        steps_used = step + 1
        message, stats = chat(model, messages, TOOL_SCHEMAS, seed, temperature)
        total_prompt_tokens = total_prompt_tokens + (stats["prompt_tokens"] or 0)
        total_output_tokens = total_output_tokens + (stats["output_tokens"] or 0)
        messages.append(message)

        tool_calls = []
        if message.tool_calls:
            for call in message.tool_calls:
                name = call.function.name
                args = call.function.arguments
                result = call_tool(workdir, name, args)
                if result.startswith("ERROR: unknown tool"):
                    format_errors = format_errors + 1
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
        log_file.write(json.dumps(record) + "\n")
        log_file.flush()

        if not message.tool_calls:
            stop_reason = "answered"
            break

    summary = {
        "steps": steps_used,
        "stop_reason": stop_reason,
        "format_errors": format_errors,
        "prompt_tokens": total_prompt_tokens,
        "output_tokens": total_output_tokens,
        "seconds": round(time.time() - start, 2),
    }
    return summary
