# Runs a command inside a macOS sandbox (sandbox-exec). Used by run_shell and run_skill.
# The sandbox:
#   - blocks writing anywhere except the task folder
#   - blocks reading file contents under /Users (your home), except the Python install,
#     this project's .venv, and any extra folders passed in (e.g. the skill code folder)
#   - blocks all network access
# On timeout the whole process group is killed (so a long `find /` does not keep running).

import os
import signal
import subprocess

MAX_OUTPUT_CHARS = 2000
TIMEOUT_SECONDS = 30

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
PYTHON_HOME = os.path.realpath(os.path.expanduser("~/.local/share/uv/python"))
PROJECT_VENV = os.path.join(PROJECT_DIR, ".venv")


def cut(text):
    if len(text) > MAX_OUTPUT_CHARS:
        return text[:MAX_OUTPUT_CHARS] + "\n...[output cut]"
    return text


def make_profile(workdir, extra_read_dirs):
    workdir = os.path.realpath(workdir)
    read_paths = '(subpath "' + workdir + '") (subpath "' + PYTHON_HOME + '") (subpath "' + PROJECT_VENV + '")'
    for folder in extra_read_dirs:
        read_paths = read_paths + ' (subpath "' + os.path.realpath(folder) + '")'
    profile = "(version 1)\n"
    profile = profile + "(allow default)\n"
    profile = profile + "(deny network*)\n"
    profile = profile + "(deny file-write*)\n"
    profile = profile + '(allow file-write* (subpath "' + workdir + '") (literal "/dev/null") (literal "/dev/tty") (subpath "/dev/fd"))\n'
    profile = profile + '(deny file-read-data (subpath "/Users"))\n'
    profile = profile + "(allow file-read-data " + read_paths + ")\n"
    return profile


# argv is the command as a list, e.g. ["/bin/sh", "-c", "ls"] or ["python", "skill.py", "{}"]
def run_sandboxed(workdir, argv, extra_read_dirs):
    # The agent can delete its own task folder (qwen did: `rm -rf <task folder>`).
    # Then there is nowhere to run commands, so report it instead of crashing the run.
    if not os.path.isdir(workdir):
        return "ERROR: the task folder no longer exists (it was deleted)"
    process = subprocess.Popen(
        ["sandbox-exec", "-p", make_profile(workdir, extra_read_dirs)] + argv,
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        return "ERROR: command took longer than " + str(TIMEOUT_SECONDS) + " seconds"
    text = "exit code: " + str(process.returncode) + "\n"
    text = text + "stdout:\n" + stdout + "\n"
    text = text + "stderr:\n" + stderr
    return cut(text)
