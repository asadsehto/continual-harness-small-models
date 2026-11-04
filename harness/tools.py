# The tools the agent can use. Every tool works inside one task folder (workdir).
# read_file and write_file refuse paths outside the task folder.
# run_shell runs inside a macOS sandbox (sandbox-exec). The sandbox:
#   - blocks writing anywhere except the task folder
#   - blocks reading file contents under /Users (your home), except the Python install
#     and this project's .venv, which commands like `python test.py` need
#   - blocks all network access
# On timeout the whole process group is killed (so a long `find /` does not keep running).

import os
import signal
import subprocess

MAX_OUTPUT_CHARS = 2000
SHELL_TIMEOUT_SECONDS = 30

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
PYTHON_HOME = os.path.realpath(os.path.expanduser("~/.local/share/uv/python"))

SANDBOX_PROFILE = """(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write* (subpath "WORKDIR") (literal "/dev/null") (literal "/dev/tty") (subpath "/dev/fd"))
(deny file-read-data (subpath "/Users"))
(allow file-read-data (subpath "WORKDIR") (subpath "PYTHON_HOME") (subpath "PROJECT_VENV"))
"""


def sandbox_profile(workdir):
    profile = SANDBOX_PROFILE.replace("WORKDIR", os.path.realpath(workdir))
    profile = profile.replace("PYTHON_HOME", PYTHON_HOME)
    profile = profile.replace("PROJECT_VENV", os.path.join(PROJECT_DIR, ".venv"))
    return profile


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
    # The agent can delete its own task folder (qwen did: `rm -rf <task folder>`).
    # Then there is nowhere to run commands, so report it instead of crashing the run.
    if not os.path.isdir(workdir):
        return "ERROR: the task folder no longer exists (it was deleted)"
    process = subprocess.Popen(
        ["sandbox-exec", "-p", sandbox_profile(workdir), "/bin/sh", "-c", cmd],
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=SHELL_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        return "ERROR: command took longer than " + str(SHELL_TIMEOUT_SECONDS) + " seconds"
    text = "exit code: " + str(process.returncode) + "\n"
    text = text + "stdout:\n" + stdout + "\n"
    text = text + "stderr:\n" + stderr
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
