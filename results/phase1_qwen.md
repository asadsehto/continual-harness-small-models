# Phase 1 result: qwen3.5:0.8b-mlx, Continual Harness vs baseline (2026-11-05)

Both conditions ran on the same Ollama server (0.32.6, same model digest `12f40a7f…`). 20 tasks (15 train, 5 held-out), 3 seeds,
max 15 steps, temperature 0.7.

- Baseline (C0): `runs/20261105-001629_qwen3.5-0.8b-mlx` (14.5 min)
- Continual Harness (C1): `runs/20261105-003144_phase1_qwen3.5-0.8b-mlx` (55 min)

## Main numbers

| | C0 baseline | C1 eval (frozen harness) |
|---|---|---|
| all | 19/60 = 32% (CI 21–44%) | 9/60 = 15% (CI 8–26%) |
| train | 15/45 = 33% (CI 21–48%) | 7/45 = 16% (CI 8–29%) |
| held-out | 4/15 = 27% (CI 11–52%) | 2/15 = 13% (CI 4–38%) |

- C1 during learning (harness changing): 12/45 = 27%.
- Paired (task, seed): C1 better 5, worse 15, same 40.
- Per seed at eval: 3, 2, 4 of 20.
- Forgetting: 8 of the 12 tasks passed during learning failed at eval.

**Result: Continual Harness made qwen 0.8B worse (32% → 15%).**

## Why (from the logs)

- The refiner (the same 0.8B model) wrote harmful content. Final strategies are ~3000 chars of
  wrong advice: "use `cd /task`", "install pytest with pip", "chmod +x", paths outside the folder.
- Skills it wrote are broken or fake: they write to `/test/.txt` or just set `success = True`.
  The agent called `run_skill` only 3 times; `call_subagent` never.
- 13 of 68 refiner passes gave unparseable output or an error; 58 edits were applied with no check.
- Longer prompt: mean prompt tokens per task run went from ~14k (C0) to ~42–48k (C1), and runs hit
  the 15-step limit far more often (C0: 29/60 max_steps; C1: 53/60).

This matches the capability floor the original paper reports: a weak model cannot refine its own
harness. It is the motivation for Phase 2 (refiner checks: validation gate, stronger refiner,
structured edits).
