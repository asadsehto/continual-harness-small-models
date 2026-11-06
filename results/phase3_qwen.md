# Phase 3 result: qwen3.5:0.8b-mlx, full system (C8) and Reflexion baseline (2026-11-07)

All runs: same Ollama server (0.32.6, same qwen digest `12f40a7f…`), 20 tasks (15 train,
5 held-out), 3 seeds, task order shuffled per seed, max 15 steps, temperature 0.7.
Eval uses the frozen final harness. Same setup as Phase 1 and Phase 2.

- **C8** = C1 + all five techniques together: separate refiner (gemma4:e4b), regression gate,
  memory database, verified skills, structured errors.
- **Reflexion** = static harness (C0) + Reflexion-style memory: after each learning task the
  worker writes a short reflection (up to 500 characters); the last 3 reflections are added
  to the prompt of the next task. No harness edits.
- The Self-Harness-style gated-edits baseline is the regression gate condition from Phase 2.

## Main numbers (all conditions)

McNemar = exact two-sided test on tasks where one condition passed and C0 failed, or the other way.
Forgetting = tasks passed while learning but failed at eval.

| Condition | All (95% CI) | Train | Held-out (95% CI) | Better / worse vs C0 | McNemar p | Forgetting | s / task | prompt tokens / task | learning min |
|---|---|---|---|---|---|---|---|---|---|
| C0 baseline | 19/60 = 32% (21–44) | 15/45 = 33% | 4/15 = 27% (11–52) | - | - | - | 14.5 | 14240 | - |
| C1 | 9/60 = 15% (8–26) | 7/45 = 16% | 2/15 = 13% (4–38) | 5 / 15 | 0.041 | 8 of 12 | 29.7 | 44875 | 13.5 |
| + separate refiner | 13/60 = 22% (13–34) | 11/45 = 24% | 2/15 = 13% (4–38) | 6 / 12 | 0.238 | 8 of 13 | 19.3 | 46132 | 13.7 |
| + regression gate | 12/60 = 20% (12–32) | 8/45 = 18% | 4/15 = 27% (11–52) | 3 / 10 | 0.092 | 8 of 14 | 18.3 | 30745 | 11.8 |
| + memory database | 12/60 = 20% (12–32) | 9/45 = 20% | 3/15 = 20% (7–45) | 7 / 14 | 0.189 | 8 of 12 | 22.9 | 43247 | 10.6 |
| + verified skills | 10/60 = 17% (9–28) | 8/45 = 18% | 2/15 = 13% (4–38) | 7 / 16 | 0.093 | 7 of 11 | 29.2 | 37521 | 15.2 |
| + structured errors | 5/60 = 8% (4–18) | 4/45 = 9% | 1/15 = 7% (1–30) | 2 / 16 | 0.001 | 6 of 10 | 26.7 | 46262 | 17.0 |
| **C8 full system** | **21/60 = 35% (24–48)** | 12/45 = 27% | **9/15 = 60% (36–80)** | 14 / 12 | 0.845 | 10 of 18 | 28.1 | 33662 | 23.9 |
| Reflexion baseline | 3/60 = 5% (2–14) | 3/45 = 7% | 0/15 = 0% (0–20) | 1 / 17 | <0.001 | 6 of 7 | 22.3 | 27280 | 22.4 |

## Findings

**1. Only the full system reaches the baseline.** Every single technique leaves qwen 0.8B below
C0 (8–22%). All five together (C8) give 35%, equal to or slightly above C0 (32%). The paired
difference vs C0 is not significant (14 better, 12 worse, p = 0.845).

**2. C8 is best on held-out tasks, but not significantly.** C8 passes 9 of 15 held-out tasks (60%)
vs 4 of 15 for C0 (27%). The CIs overlap (36–80 vs 11–52), so with 5 held-out tasks and 3 seeds
this is a hint, not a result. On train tasks C8 is below C0 (27% vs 33%).

**3. Reflexion-style memory hurts this model badly.** 3 of 60 (5%), significantly worse than C0
(1 better, 17 worse, p < 0.001), and 0 of 15 on held-out. Most eval tasks used all 15 steps
without finishing. The reflections add text to an already long prompt that the 0.8B model
does not use well. Seed results: 2, 1 and 0 out of 20.

**4. Forgetting stays high in every learning condition.** Between 6 of 7 and 10 of 18 tasks that
passed while learning failed at eval, including C8 (10 of 18) and the gate condition. The gate
does not stop forgetting at this scale (H4 not supported in the form stated).

**5. Cost.** C8 takes about 2x the time per task of C0 (28.1 vs 14.5 s) and 2.4x the prompt
tokens, plus 24 minutes of learning per seed.

## C8 details

- Regression gate over 3 seeds: 14 refiner generations checked, 11 kept, 3 undone. Of the 11
  kept, 9 were ties and only 2 were real gains (same pattern as the gate-only run).
- Tool use across all run logs (learning, gate and eval): `run_skill` 38 times and
  `call_subagent` 5 times. The Reflexion run used neither.
- Seed results: 7, 6 and 8 out of 20.

## Hypotheses so far (qwen 0.8B only)

| Hypothesis | Result |
|---|---|
| H1. Continual Harness as-is gives little gain or makes results worse | Supported: C1 is significantly worse than C0 (15% vs 32%, p = 0.041) |
| H2. Each technique raises the gain on its own | Not supported: no single technique reaches C0; only the combination does |
| H3. A bigger local refiner beats self-refinement | Weakly supported: separate refiner 22% vs C1 15%, not significant |
| H4. Without a gate, edits cause forgetting | Forgetting is high, but also with the gate |

## Runs

- C0 `runs/20261105-001629_qwen3.5-0.8b-mlx`
- C1 `runs/20261105-003144_phase1_qwen3.5-0.8b-mlx`
- separate refiner `runs/20261105-020938_phase1_qwen3.5-0.8b-mlx_refiner-gemma4-e4b`
- regression gate `runs/20261105-034043_phase1_qwen3.5-0.8b-mlx_gate`
- memory database `runs/20261105-174004_phase1_qwen3.5-0.8b-mlx_memdb`
- verified skills `runs/20261105-225822_phase1_qwen3.5-0.8b-mlx_verified`
- structured errors `runs/20261106-010306_phase1_qwen3.5-0.8b-mlx_errors`
- C8 `runs/20261106-100238_phase1_qwen3.5-0.8b-mlx_refiner-gemma4-e4b_gate_memdb_verified_errors`
- Reflexion `runs/20261106-215702_reflexion_qwen3.5-0.8b-mlx`

Table made with:
`uv run python -m scripts.analyze runs/<C0> name=runs/<run> ...`

## Notes on the runs

- The first Reflexion run stopped in seed 1 because the connection to the model server was
  lost. It is not counted (kept in `runs/stopped/`), and Reflexion was rerun from the start.
  Its seed 0 had passed 6 of 20 vs 2 of 20 in the counted run, which shows the large run-to-run
  variance of this model at temperature 0.7.
- Limits: one worker model, 20 tasks, 3 seeds. Differences of a few tasks are within noise.
