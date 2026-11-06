# Phase 3 result: gemma4:e2b-it-qat, C0 / C1 / C8 (2026-11-07)

Same tasks and settings as the qwen runs: 20 tasks (15 train, 5 held-out), 3 seeds, task order
shuffled per seed (same orders as qwen), max 15 steps, temperature 0.7. Eval uses the frozen
final harness.

Servers:
- Worker `gemma4:e2b-it-qat` (digest `07ea59a4…`) ran on one Ollama server (0.40.7) for all three conditions.
- In C8, the refiner `gemma4:e4b` (digest `c6eb396d…`) and the embedding model `nomic-embed-text`
  (digest `0a109f42…`) ran on a second Ollama server (0.32.6), the same server and models used
  for the qwen runs.
- The model is unloaded after every task on both servers, as in all other runs.

Conditions:
- **C0** = static harness (baseline).
- **C1** = Continual Harness as-is (gemma refines its own harness).
- **C8** = C1 + all five techniques: separate refiner (gemma4:e4b), regression gate, memory
  database, verified skills, structured errors.

## Main numbers

McNemar = exact two-sided test on tasks where one condition passed and C0 failed, or the other way.
Forgetting = tasks passed while learning but failed at eval.

| Condition | All (95% CI) | Train | Held-out (95% CI) | Better / worse vs C0 | McNemar p | Forgetting | s / task | prompt tokens / task | learning min |
|---|---|---|---|---|---|---|---|---|---|
| C0 baseline | 55/60 = 92% (82–96) | 43/45 = 96% | 12/15 = 80% (55–93) | - | - | - | 14.5 | 3403 | - |
| C1 | 46/60 = 77% (65–86) | 37/45 = 82% | 9/15 = 60% (36–80) | 2 / 11 | 0.022 | 7 of 42 | 14.7 | 8400 | 10.2 |
| C8 full system | 48/60 = 80% (68–88) | 37/45 = 82% | 11/15 = 73% (48–89) | 2 / 9 | 0.065 | 3 of 40 | 11.4 | 5154 | 7.6 |

Seed results (out of 20): C0 19, 20, 16 / C1 16, 14, 16 / C8 18, 12, 18.

## Findings

**1. Gemma's static baseline is near the ceiling.** C0 passes 92% of tasks (96% of train tasks),
so there is little room for any harness learning to help on this task set.

**2. Continual Harness as-is hurts gemma too.** C1 is significantly worse than C0 (77% vs 92%,
2 better, 11 worse, p = 0.022). This repeats the qwen result (H1) on a second model.

**3. C8 recovers part of the loss, but stays below the baseline.** C8 gives 80% (not significantly
different from C0, p = 0.065, and close to C1). On held-out tasks C8 is 73% vs 60% for C1 and
80% for C0. Unlike qwen, C8 does not reach or beat the baseline here.

**4. The regression gate rejected most edits.** 6 gate checks over 3 seeds: 5 undone (the edits
lowered the pass count on recent tasks), 1 kept (a tie). So for gemma most refiner edits were
judged harmful, and C8's harness stayed close to the starting harness.

**5. Forgetting is much lower with C8.** 3 of 40 tasks passed while learning failed at eval,
vs 7 of 42 for C1. This matches the gate keeping few edits (H4 supported for gemma).

**6. Cost.** C8 is faster per task than C1 (11.4 vs 14.7 s) and uses fewer prompt tokens
(5154 vs 8400), because fewer edits were kept. C1 more than doubles the prompt size of C0.

## Tool use (trajectory logs: learning, gate and eval)

| Condition | `run_skill` | `call_subagent` |
|---|---|---|
| C1 | 8 | 10 |
| C8 | 2 | 7 |

## Comparison with qwen3.5:0.8b-mlx (results/phase3_qwen.md)

| Condition | qwen 0.8B | gemma e2b |
|---|---|---|
| C0 | 32% | 92% |
| C1 | 15% (p = 0.041, worse) | 77% (p = 0.022, worse) |
| C8 | 35% (p = 0.845) | 80% (p = 0.065) |

- In both models, C1 is significantly worse than the static baseline.
- In both, C8 is better than C1.
- For the weak model (qwen) C8 reaches the baseline; for the strong model (gemma, near the
  ceiling) it does not. The combined techniques limit the harm of self-editing, but on this
  task set they do not give a clear gain over a good static harness.

## Runs

- C0 `runs/20261107-021257_gemma4-e2b-it-qat`
- C1 `runs/20261107-022733_phase1_gemma4-e2b-it-qat`
- C8 `runs/20261107-025921_phase1_gemma4-e2b-it-qat_refiner-gemma4-e4b_gate_memdb_verified_errors`

Table made with:
`uv run python -m scripts.analyze runs/<C0> name=runs/<run> ...`

C8 command (worker and refiner/embeddings on different servers):
`uv run python -m scripts.run_phase1 --model gemma4:e2b-it-qat --host <worker server> --refiner-model gemma4:e4b --refiner-host <refiner server> --embed-host <refiner server> --gate --memory-db --verified-skills --structured-errors --seeds 3`

## Notes on the runs

- All three runs finished on the first try; none was stopped.
- New code options for this run: `--refiner-host` and `--embed-host` in `scripts/run_phase1.py`
  (and per-model servers in `harness/model.py`). Without them, everything uses `--host` as before,
  so earlier runs are not affected.
- Limits: 20 tasks, 3 seeds, a near-ceiling baseline. Differences of a few tasks are within noise
  (for example C8 seed 1 gave 12 of 20 while seeds 0 and 2 gave 18).
