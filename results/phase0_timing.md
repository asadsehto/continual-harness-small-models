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
