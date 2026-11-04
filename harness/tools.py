# The tools the agent can use. Every tool works inside one task folder (workdir).
# read_file and write_file refuse paths outside the task folder.
# run_shell starts in the task folder. It is NOT a full sandbox (a command like "cd /" can
# still leave), so only run the agent on pilot tasks, never on folders with real data.

import os
import subprocess

MAX_OUTPUT_CHARS = 2000
SHELL_TIMEOUT_SECONDS = 30


def inside_workdir(workdir, path):
    full_path = os.path.realpath(os.path.join(workdir, path))
    root = os.path.realpath(workdir)
    if full_path == root or full_path.startswith(root + os.sep):
        return full_path
    return None


def cut(text):
    if len(text) > MAX_OUTPUT_CHARS:
        return text[:MAX_OUTPUT_CHARS] + "\n...[output cut]"
    return text


def run_shell(workdir, cmd):
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=SHELL_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return "ERROR: command took longer than " + str(SHELL_TIMEOUT_SECONDS) + " seconds"
    text = "exit code: " + str(result.returncode) + "\n"
    text = text + "stdout:\n" + result.stdout + "\n"
    text = text + "stderr:\n" + result.stderr
    return cut(text)


def read_file(workdir, path):
    full_path = inside_workdir(workdir, path)
    if full_path is None:
        return "ERROR: path is outside the task folder"
    if not os.path.isfile(full_path):
        return "ERROR: file not found: " + path
    with open(full_path) as f:
        return cut(f.read())


def write_file(workdir, path, content):
    full_path = inside_workdir(workdir, path)
    if full_path is None:
        return "ERROR: path is outside the task folder"
    folder = os.path.dirname(full_path)
    os.makedirs(folder, exist_ok=True)
    with open(full_path, "w") as f:
        f.write(content)
    return "wrote " + str(len(content)) + " characters to " + path


# Tool descriptions sent to the model (JSON schema format that Ollama expects).
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a shell command in the task folder. Returns exit code, stdout and stderr.",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string", "description": "The shell command"}},
                "required": ["cmd"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file in the task folder.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "File path relative to the task folder"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write (or overwrite) a text file in the task folder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to the task folder"},
                    "content": {"type": "string", "description": "The full file content"},
                },
                "required": ["path", "content"],
            },
        },
    },
]


def call_tool(workdir, name, args):
    if name == "run_shell" and "cmd" in args:
        return run_shell(workdir, args["cmd"])
    if name == "read_file" and "path" in args:
        return read_file(workdir, args["path"])
    if name == "write_file" and "path" in args and "content" in args:
        return write_file(workdir, args["path"], args["content"])
    return "ERROR: unknown tool or missing arguments: " + name + " " + str(args)
