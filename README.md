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

Safety: `run_shell` and `run_skill` run inside a macOS sandbox (`harness/sandbox.py`): they can write
only in the temporary task folder, cannot read files under /Users, and have no network.

## Citation of the original work

Karten, S., Zhang, J., Upaa Jr, T., Feng, R., Li, W., Shi, C., Jin, C., Vodrahalli, K. (2026).
Continual Harness. arXiv:2605.09998. Code: https://github.com/sethkarten/continual-harness (MIT).

## Using another Ollama server

By default the model runs on the local Ollama (http://127.0.0.1:11434). Pass `--host <url>`
to use another Ollama server; the agent loop and the sandbox always run locally.

Each run's `config.json` records the server's Ollama version and the model digest. Compare
conditions only when they ran on the same server: the same seed can sample differently on a
different server.
