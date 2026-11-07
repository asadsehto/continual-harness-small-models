# Self-Improving Harnesses Below the Capability Floor

**Continual Harness with Small Local Language Models**
Asad Saleem, Independent Researcher

Code, tasks and data for the paper. The LaTeX source is in [paper/](paper/).

## What this is

[Continual Harness](https://arxiv.org/abs/2605.09998) (Karten et al., 2026) lets an AI agent
rewrite its own prompt, memory, skills and helper agents while it works. It was tested with large
cloud models. We test it with two small models that run on a laptop through
[Ollama](https://ollama.com): `qwen3.5:0.8b` and `gemma4:e2b`, on 20 short computer tasks (coding,
files, shell, data) that are checked automatically.

## Main results

- **Continual Harness as-is made both small models worse** than a fixed harness
  (qwen 32% → 15%, gemma 92% → 77%), in 5 of 6 seeds.
- **No single added technique fixed this.** All five together (a larger refiner, a regression
  gate, a memory database, verified skills and structured errors) brought qwen back to the
  fixed-harness level (35%). On held-out tasks it passed 60% against 27%, which is suggestive
  but not significant with our sample size.
- **A Reflexion-style memory made qwen much worse** (5%).

The paper reports every comparison with exact paired tests and Holm correction, results per seed,
and a determinism check.

## Rebuild every table in the paper

```bash
uv sync
uv run python -m scripts.make_tables
```

This reads only [data/](data/): the outcome of every task run used in the paper (1,256 runs), every
harness edit and gate decision, and the server version and model digest of every model. The data
contain no trajectories or model output.

## Run the experiments

You need [uv](https://docs.astral.sh/uv/) and Ollama with the models pulled.

```bash
uv run python -m scripts.check_tasks                                   # test every checker
uv run python -m scripts.run_pilot --model qwen3.5:0.8b-mlx --tasks all --seeds 3    # C0 fixed harness
uv run python -m scripts.run_phase1 --model qwen3.5:0.8b-mlx --seeds 3               # C1 Continual Harness
uv run python -m scripts.run_phase1 --model qwen3.5:0.8b-mlx --seeds 3 \
    --refiner-model gemma4:e4b --gate --memory-db --verified-skills --structured-errors   # C8 all five
uv run python -m scripts.run_phase1 --model qwen3.5:0.8b-mlx --seeds 3 --reflexion   # Reflexion
uv run python -m scripts.export_data                                   # runs/ to data/
```

Runs are saved in `runs/` (not in git). Each run records its settings, the Ollama version and model
digest, every step, every harness edit (before and after) and every result. The same seed can give
different outputs on a different server or Ollama version, so compare conditions only when they
ran on the same server.

By default the model runs on the local Ollama (http://127.0.0.1:11434). Pass `--host <url>` to use
another Ollama server; the agent loop and the sandbox always run locally.

**Safety:** `run_shell` and `run_skill` run inside a macOS sandbox (`harness/sandbox.py`). They can
write only in the temporary task folder, cannot read personal files and have no network.

## Layout

```
harness/   agent loop, tools, sandbox, Continual Harness refiner, memory database, Reflexion
tasks/     the 20 tasks (15 train, 5 held-out) and their checkers
scripts/   run the conditions, export data, rebuild tables
data/      per-run outcomes used in the paper
results/   result notes written during the study
docs/      research plan and notes on the original code
paper/     LaTeX source of the paper
```

## Citation

```bibtex
@misc{saleem2026selfimproving,
  title  = {Self-Improving Harnesses Below the Capability Floor:
            Continual Harness with Small Local Language Models},
  author = {Saleem, Asad},
  year   = {2026},
  url    = {https://github.com/asadsehto/continual-harness-small-models}
}
```

This work builds on Continual Harness: Karten, S., Zhang, J., Upaa Jr, T., Feng, R., Li, W., Shi, C.,
Jin, C., Vodrahalli, K. (2026). arXiv:2605.09998. Code: https://github.com/sethkarten/continual-harness
(MIT).

## License

MIT. See [LICENSE](LICENSE).
