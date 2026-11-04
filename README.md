# continual-harness-small-models

Making Continual Harness (Karten et al., 2026, arXiv:2605.09998) self-improvement work for
small local models running on Ollama. Plan: [docs/research_plan.md](docs/research_plan.md).
Notes on the original code: [docs/continual_harness_notes.md](docs/continual_harness_notes.md).

## Layout

```
harness/   model.py (Ollama client), tools.py (run_shell, read_file, write_file), agent.py (baseline loop)
tasks/     task.py (task format + helpers), pilot.py (20 pilot tasks: 15 train, 5 held-out)
scripts/   check_tasks.py (tests every checker), run_pilot.py (runs the agent, saves logs)
docs/      research plan, notes, results
runs/      run logs (not in git)
results/   summarized results kept in git
```

## Run

```bash
uv sync
uv run python -m scripts.check_tasks
uv run python -m scripts.run_pilot --model gemma4:e2b-it-qat --tasks fix_fib --seeds 1
```

Safety: `run_shell` starts in a temporary task folder but is not a full sandbox.
Only run it on the pilot tasks.

## Citation of the original work

Karten, S., Zhang, J., Upaa Jr, T., Feng, R., Li, W., Shi, C., Jin, C., Vodrahalli, K. (2026).
Continual Harness. arXiv:2605.09998. Code: https://github.com/sethkarten/continual-harness (MIT).
