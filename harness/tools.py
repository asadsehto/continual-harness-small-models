# The tools the agent can use. Every tool works inside one task folder (workdir).
#   run_shell, run_skill  run inside the macOS sandbox (see harness/sandbox.py)
#   read_file, write_file run in our Python process, but refuse paths outside the task folder
# run_skill and call_subagent exist only in Phase 1 (Continual Harness), and are shown to the
# model only when the harness has at least one skill / sub-agent.

import json
import os
import sys
import tempfile

from harness.sandbox import run_sandboxed, cut


def inside_workdir(workdir, path):
    full_path = os.path.realpath(os.path.join(workdir, path))
    root = os.path.realpath(workdir)
    if full_path == root or full_path.startswith(root + os.sep):
        return full_path
    return None


def run_shell(workdir, cmd):
    return run_sandboxed(workdir, ["/bin/sh", "-c", cmd], [])


def read_file(workdir, path):
    full_path = inside_workdir(workdir, path)
    if full_path is None:
        return "ERROR: path is outside the task folder"
    if not os.path.isfile(full_path):
        return "ERROR: file not found: " + path
    with open(full_path, errors="replace") as f:
        return cut(f.read())


def write_file(workdir, path, content):
    full_path = inside_workdir(workdir, path)
    if full_path is None:
        return "ERROR: path is outside the task folder"
    # A bad path (for example a folder name, or a file where a folder should be) is the
    # model's mistake: tell it, do not crash the run.
    try:
        folder = os.path.dirname(full_path)
        os.makedirs(folder, exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)
    except OSError as error:
        return "ERROR: could not write " + path + ": " + error.strerror
    return "wrote " + str(len(content)) + " characters to " + path


# Skill code is written by the Refiner (the model). It is inline Python that reads `args`
# and may set `result`. It is NEVER run inside our process: we write it to a file in a
# separate temp folder and run that file with Python inside the sandbox, in the task folder.
SKILL_WRAPPER_TOP = "import json, os, sys, re, math, csv, shutil, glob, subprocess\nargs = json.loads(sys.argv[1])\nresult = None\n"
SKILL_WRAPPER_BOTTOM = "\nprint('RESULT:', json.dumps(result, default=str))\n"


def run_skill(workdir, state, name, skill_args):
    skill = None
    for item in state["skills"]:
        if item["name"] == name:
            skill = item
    if skill is None:
        names = [item["name"] for item in state["skills"]]
        return "ERROR: no skill named " + name + ". Skills: " + str(names)
    if not skill.get("code"):
        return "ERROR: skill " + name + " has no code (it is a description only): " + skill.get("description", "")
    skill_dir = tempfile.mkdtemp(prefix="skill_")
    skill_file = os.path.join(skill_dir, "skill.py")
    with open(skill_file, "w") as f:
        f.write(SKILL_WRAPPER_TOP + skill["code"] + SKILL_WRAPPER_BOTTOM)
    try:
        output = run_sandboxed(workdir, [sys.executable, skill_file, json.dumps(skill_args)], [skill_dir])
    finally:
        os.remove(skill_file)
        os.rmdir(skill_dir)
    return output


def tool_schema(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


# Tool descriptions sent to the model (JSON schema format that Ollama expects).
RUN_SHELL = tool_schema(
    "run_shell", "Run a shell command in the task folder. Returns exit code, stdout and stderr.",
    {"cmd": {"type": "string", "description": "The shell command"}}, ["cmd"],
)
READ_FILE = tool_schema(
    "read_file", "Read a text file in the task folder.",
    {"path": {"type": "string", "description": "File path relative to the task folder"}}, ["path"],
)
WRITE_FILE = tool_schema(
    "write_file", "Write (or overwrite) a text file in the task folder.",
    {
        "path": {"type": "string", "description": "File path relative to the task folder"},
        "content": {"type": "string", "description": "The full file content"},
    },
    ["path", "content"],
)
RUN_SKILL = tool_schema(
    "run_skill", "Run a saved skill (listed in the system prompt) in the task folder.",
    {
        "name": {"type": "string", "description": "Skill name"},
        "args": {"type": "object", "description": "Arguments for the skill (a JSON object)"},
    },
    ["name"],
)
CALL_SUBAGENT = tool_schema(
    "call_subagent", "Give a sub-task to a saved sub-agent (listed in the system prompt). Returns its final answer.",
    {
        "name": {"type": "string", "description": "Sub-agent name"},
        "task": {"type": "string", "description": "What the sub-agent should do"},
    },
    ["name", "task"],
)

TOOL_SCHEMAS = [RUN_SHELL, READ_FILE, WRITE_FILE]

# Tools a sub-agent is allowed to use (it can never call another sub-agent).
SUBAGENT_ALLOWED_TOOLS = ["run_shell", "read_file", "write_file", "run_skill"]


def schemas_for(state):
    schemas = [RUN_SHELL, READ_FILE, WRITE_FILE]
    if state is not None and len(state["skills"]) > 0:
        schemas.append(RUN_SKILL)
    if state is not None and len(state["subagents"]) > 0:
        schemas.append(CALL_SUBAGENT)
    return schemas


def is_text(value):
    return isinstance(value, str)


# run_subagent is passed in by agent.py (it needs the agent loop, which lives there).
def call_tool(workdir, name, args, state=None, run_subagent=None):
    if not isinstance(args, dict):
        return "ERROR: unknown tool or missing arguments: " + str(name) + " " + str(args)
    if name == "run_shell" and is_text(args.get("cmd")):
        return run_shell(workdir, args["cmd"])
    if name == "read_file" and is_text(args.get("path")):
        return read_file(workdir, args["path"])
    if name == "write_file" and is_text(args.get("path")) and is_text(args.get("content")):
        return write_file(workdir, args["path"], args["content"])
    if name == "run_skill" and state is not None and is_text(args.get("name")):
        skill_args = args.get("args", {})
        if not isinstance(skill_args, dict):
            skill_args = {"value": skill_args}
        return run_skill(workdir, state, args["name"], skill_args)
    if name == "call_subagent" and run_subagent is not None and is_text(args.get("name")) and is_text(args.get("task")):
        return run_subagent(args["name"], args["task"])
    return "ERROR: unknown tool or missing arguments: " + str(name) + " " + str(args)
