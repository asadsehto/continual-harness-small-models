# Fix A and Fix B: qwen3.5:0.8b-mlx (2026-11-05)

All runs: model served from the Mac mini (Ollama 0.32.6, same qwen digest), 20 tasks
(15 train, 5 held-out), 3 seeds, max 15 steps, temperature 0.7. Eval uses the frozen final harness.

| Condition | Who refines | Gate | All | Train | Held-out | vs C0 paired (better / worse) |
|---|---|---|---|---|---|---|
| C0 baseline | none | - | 19/60 = 32% (21–44) | 15/45 = 33% | 4/15 = 27% | - |
| C1 Phase 1 | qwen 0.8B | no | 9/60 = 15% (8–26) | 7/45 = 16% | 2/15 = 13% | 5 / 15 |
| C3 Fix A | gemma4:e4b | no | 13/60 = 22% (13–34) | 11/45 = 24% | 2/15 = 13% | 6 / 12 |
| C2 Fix B | qwen 0.8B | yes | 12/60 = 20% (12–32) | 8/45 = 18% | 4/15 = 27% | 3 / 10 |

Runs:
- C0 `runs/20261105-001629_qwen3.5-0.8b-mlx`
- C1 `runs/20261105-003144_phase1_qwen3.5-0.8b-mlx`
- C3 `runs/20261105-020938_phase1_qwen3.5-0.8b-mlx_refiner-gemma4-e4b`
- C2 `runs/20261105-034043_phase1_qwen3.5-0.8b-mlx_gate`

(Wilson 95% CIs in brackets. Note: the CIs of C1, C2, C3 overlap; with 3 seeds the
differences between the fixes are not significant. Only C1 vs C0 is clearly lower.)

## Fix B (gate) details

The gate reruns up to 3 recent train tasks with the new harness and keeps the edits if the new
harness passes at least as many as the old one did.

- 16 refiner generations checked: kept 13, undone 3.
- Of the 13 kept, 10 were ties (7 were 0 vs 0, 3 were 1 vs 1) and only 3 were real gains.
  **The gate is weak:** with few, mostly-failing tasks it cannot tell a good edit from a bad one,
  so most edits pass on a tie.
- Held-out stayed at the baseline (27%); train dropped (33% → 18%).
- Forgetting: 8 of 14 tasks passed while learning failed at eval.
- The agent used `run_skill` 13 times, `call_subagent` once.

Bug found and fixed during the first Fix B attempt: `write_file` crashed the run when the model
wrote to a folder path. It now returns an error to the model. Earlier runs never hit this case
(they would have crashed), so they are not affected. The crashed run is in `runs/stopped/`.

## What this says so far

1. Continual Harness hurts qwen 0.8B (32% → 15%).
2. Both fixes reduce the harm (to 22% and 20%) but neither reaches the baseline.
3. Next options: a stricter gate (do not keep edits on 0-vs-0 ties), or both fixes together
   (gemma refines + gate).
