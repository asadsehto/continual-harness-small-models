# Tests every checker without any model:
#   1. on the untouched task folder the checker must say FAIL (task is not already solved)
#   2. after the reference solution the checker must say PASS
# Run: uv run python -m scripts.check_tasks

from tasks.pilot import ALL_TASKS
from tasks.task import make_workdir, remove_workdir

problems = 0
for task in ALL_TASKS:
    workdir = make_workdir(task)
    before = task["check"](workdir, task)
    task["reference"](workdir, task)
    after = task["check"](workdir, task)
    remove_workdir(workdir)

    ok = (before is False) and (after is True)
    if not ok:
        problems = problems + 1
    print(task["id"].ljust(16), task["split"].ljust(8), "before:", before, " after:", after, " OK" if ok else " PROBLEM")

print()
print(str(len(ALL_TASKS)) + " tasks, " + str(problems) + " problems")
