# Task format. Every task is a dict with:
#   id          short unique name
#   category    "coding" or "pc"
#   split       "train" (the harness may learn from it) or "heldout" (only evaluated)
#   instruction the text the agent receives
#   files       {path: content} written into a fresh task folder before the agent starts
#   check       function(workdir) -> True/False, looks only at the final folder state
#   reference   function(workdir) that solves the task (used only to test the checker)

import os
import shutil
import subprocess
import sys
import tempfile


def make_workdir(task):
    workdir = tempfile.mkdtemp(prefix="task_" + task["id"] + "_")
    for path in task["files"]:
        full_path = os.path.join(workdir, path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(task["files"][path])
    return workdir


def remove_workdir(workdir):
    shutil.rmtree(workdir, ignore_errors=True)


# Small helpers used by the checkers and reference solutions.

def read(workdir, path):
    full_path = os.path.join(workdir, path)
    if not os.path.isfile(full_path):
        return None
    with open(full_path) as f:
        return f.read()


def write(workdir, path, content):
    full_path = os.path.join(workdir, path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w") as f:
        f.write(content)


def run_python(workdir, args):
    result = subprocess.run(
        [sys.executable] + args,
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result


def tests_pass(workdir, test_file):
    result = run_python(workdir, ["-m", "unittest", test_file])
    return result.returncode == 0


def unchanged(workdir, task, path):
    return read(workdir, path) == task["files"][path]
