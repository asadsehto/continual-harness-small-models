# Technique #5: structured error feedback.
# In C1 a failed tool call returns raw text (exit code + stdout + stderr, or "ERROR: ...").
# With structured errors, a failed call returns:
#   ERROR
#   type: <one fixed error type>
#   message: <the key line, e.g. the last line of a Python traceback>
#   hint: <a fixed, general hint for this error type (never task-specific)>
#   --- full output ---
#   <the original output, unchanged>
# Successful calls are returned unchanged.

import re

HINTS = {
    "outside_task_folder": "Use paths relative to the task folder (for example 'src/app.py'). The task folder is the current directory; do not use absolute paths like /task or /tmp.",
    "file_not_found": "Check the name with run_shell 'ls' (or 'ls -R') first, then use the exact relative path.",
    "blocked_by_sandbox": "You can only read and write inside the task folder, and there is no network. Work with the files in the task folder.",
    "timeout": "The command ran too long. Avoid commands that wait for input or loop forever; run smaller steps.",
    "task_folder_deleted": "The task folder was deleted. Never delete the task folder itself.",
    "bad_tool_call": "Call one of the listed tools with all required arguments, as a proper tool call.",
    "unknown_skill_or_subagent": "Use only the skill / sub-agent names listed in the system prompt.",
    "module_not_found": "This module is not installed and cannot be installed (no network). Use the Python standard library or plain python3 instead.",
    "command_not_found": "This program is not available. Use standard shell commands or python3.",
    "syntax_error": "The Python code has a syntax error. Read the file again and fix the line shown.",
    "test_failed": "A test failed. Read the assertion message, fix the code, then run the tests again.",
    "python_exception": "The Python code raised an exception. Read the message and the line it points to, then fix it.",
    "write_failed": "The path is a folder or its parent is a file. Choose a file path inside the task folder.",
    "command_failed": "The command exited with an error. Read stderr, fix the cause, then try again.",
}


def exit_code(output):
    match = re.match(r"exit code: (-?\d+)", output)
    if match:
        return int(match.group(1))
    return None


def last_line(text):
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    if lines:
        return lines[-1][:300]
    return ""


def classify(result):
    # Returns (error_type, message), or None if the call did not fail.
    if result.startswith("ERROR: path is outside"):
        return "outside_task_folder", result[7:]
    if result.startswith("ERROR: file not found"):
        return "file_not_found", result[7:]
    if result.startswith("ERROR: command took longer"):
        return "timeout", result[7:]
    if result.startswith("ERROR: the task folder no longer exists"):
        return "task_folder_deleted", result[7:]
    if result.startswith("ERROR: unknown tool"):
        return "bad_tool_call", result[7:300]
    if result.startswith("ERROR: no skill named") or result.startswith("ERROR: no sub-agent named") or result.startswith("ERROR: tool not allowed"):
        return "unknown_skill_or_subagent", result[7:300]
    if result.startswith("ERROR: could not write"):
        return "write_failed", result[7:]
    if result.startswith("ERROR:"):
        return "command_failed", result[7:300]

    code = exit_code(result)
    if code is None or code == 0:
        return None
    stderr = result.split("stderr:", 1)[1] if "stderr:" in result else result
    if "Operation not permitted" in result:
        return "blocked_by_sandbox", last_line(stderr)
    if "ModuleNotFoundError" in stderr or "No module named" in result:
        return "module_not_found", last_line(stderr)
    if "command not found" in result:
        return "command_not_found", last_line(stderr)
    if "SyntaxError" in stderr or "IndentationError" in stderr:
        return "syntax_error", last_line(stderr)
    if "AssertionError" in result or re.search(r"\bFAILED\b", result) or "failed" in last_line(result):
        return "test_failed", last_line(stderr) or last_line(result)
    if "Traceback" in stderr:
        return "python_exception", last_line(stderr)
    if "No such file or directory" in result:
        return "file_not_found", last_line(stderr)
    return "command_failed", last_line(stderr) or last_line(result)


def structure(result):
    found = classify(result)
    if found is None:
        return result
    error_type, message = found
    return ("ERROR\ntype: " + error_type + "\nmessage: " + message + "\nhint: " + HINTS[error_type]
            + "\n--- full output ---\n" + result)
