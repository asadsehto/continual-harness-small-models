# Phase 0 timing test (2026-11-04)

Ollama 0.40.7. One run only, so treat these numbers as rough.

## Run

`gemma4:e2b-it-qat`, task `fix_fib`, seed 0, temperature 0.7, max 15 steps.

| Result | Steps | Prompt tokens | Output tokens | Seconds |
|--------|-------|---------------|---------------|---------|
| PASS (genuine: read files, ran tests, fixed `range(n-1)` → `range(n)`, re-ran tests) | 7 | 9,122 | 1,904 | 33.7 |

Notes from the log:
- Step 1 took 5.5 s (includes loading the model).
- Step 4 took 24 s: the model wrote 1,579 tokens of explanation before the fix. Long text is the main cost, not tool calls.
- It wrote the same fix twice (steps 4 and 5): a small wasted step.

## Estimate

Assumed time per task run: 34 s (this run) to 60 s (harder tasks, failures that hit 15 steps).

| Run | Task runs | Estimated time |
|-----|-----------|----------------|
| Phase 0 baseline: 20 tasks × 3 seeds × 2 models | 120 | 1.1 – 2 hours |
| Full grid: 20 tasks × 3 seeds × 9 conditions × 3 models | 1,620 | 15 – 27 hours, plus refiner calls (≈ +20–30%) |

qwen3.5:0.8b should be faster (≈ 95 vs 67 tokens/s); a larger model like gemma4:e4b slower.
The real number comes from the full Phase 0 baseline run.

# Phase 0 baseline (2026-11-04)

## gemma4:e2b-it-qat — complete (20 tasks × 3 seeds = 60 runs)

- Success: **56/60 = 93%** (95% CI 84–97%). Train 96%, held-out 87%. Coding 93%, PC 94%.
- Failures: palindrome 1/3, failed_rows 2/3, us_spelling 2/3.
- Time: 13.5 minutes total, **13.5 s mean per run** (max 65.6 s). Mean 5 steps. 0 format errors.
- **Too easy:** 93% is far above the 20–60% target. There is no room to show improvement.

## qwen3.5:0.8b-mlx — complete, in the sandbox (20 tasks × 3 seeds = 60 runs)

Run folder: `runs/20261104-230654_qwen3.5-0.8b-mlx`. Earlier stopped/crashed runs are in
`runs/stopped_unsandboxed/` and are not used.

- Success: **24/60 = 40%** (95% CI 29–53%). Train 33%, held-out 60%. Coding 48%, PC 33%.
- Never solved: fix_import, organize, find_errors, rename_photos, merge_parts, scaffold (0/3 each).
- Time: 10.4 minutes total, **10.4 s mean per run** (max 58.9 s). Mean 10 steps; 29 of 60 runs hit the 15-step limit. 0 format errors.
- Memory: model unloaded after every task (MLX cache growth fix); peak 2.4 GB.
- Inside the 20–60% target range, so qwen can show improvement on these tasks.

## Sandbox

`run_shell` now runs in macOS `sandbox-exec`: writes only inside the task folder, no reading
of files under /Users (except the Python install and the project .venv), no network,
process group killed on timeout.

What qwen tried (from the logs):
- 33 shell commands aimed outside the task folder (`/app`, `ls /`, `find /`, `~`); 29 tool results blocked by the sandbox.
- In an earlier crashed run, on delete_tmp it ran `rm -rf <its own task folder>` and then
  `rm -f /private/var/folders/.../T/*.tmp` — the system temp folder of other apps. The sandbox blocked it.
  Without the sandbox this would have deleted other apps' temp files.
- Fix: `run_shell` returns an error if the task folder was deleted; the runner marks that task as failed.

## Updated estimate

Mean 10–14 s per run. Full grid of 1,620 runs ≈ 5–7 hours plus refiner calls,
before making tasks harder. Harder tasks will make runs longer.
