# Phase 2 result: qwen3.5:0.8b-mlx, one technique at a time on top of C1 (2026-11-06)

All runs: same Ollama server (0.32.6, same qwen digest `12f40a7f…`), 20 tasks (15 train,
5 held-out), 3 seeds, task order shuffled per seed, max 15 steps, temperature 0.7.
Each condition is C1 (Continual Harness as-is) plus ONE technique. Eval uses the frozen final harness.

## Main numbers

| Condition | Technique | All (95% CI) | Train | Held-out | Paired vs C0 (better / worse) |
|---|---|---|---|---|---|
| C0 | baseline, static harness | 19/60 = 32% (21–44) | 15/45 = 33% | 4/15 = 27% | - |
| C1 | Continual Harness as-is | 9/60 = 15% (8–26) | 7/45 = 16% | 2/15 = 13% | 5 / 15 |
| + separate refiner | gemma4:e4b refines, qwen works | 13/60 = 22% (13–34) | 11/45 = 24% | 2/15 = 13% | 6 / 12 |
| + regression gate | keep edits only if recent tasks do not get worse | 12/60 = 20% (12–32) | 8/45 = 18% | 4/15 = 27% | 3 / 10 |
| + memory database | up to 200 entries, top 5 retrieved per task (nomic-embed-text) | 12/60 = 20% (12–32) | 9/45 = 20% | 3/15 = 20% | 7 / 14 |
| + verified skills | keep a skill only if its code runs on a real task folder | 10/60 = 17% (9–28) | 8/45 = 18% | 2/15 = 13% | 7 / 16 |
| + structured errors | failed tool calls return a typed error with a fixed hint | 5/60 = 8% (4–18) | 4/45 = 9% | 1/15 = 7% | 2 / 16 |

**No single technique brings qwen 0.8B back to the baseline.** Four of them (separate refiner,
gate, memory database, verified skills) reduce the harm of C1; structured errors makes it worse.
The CIs of all C1-based conditions overlap, so with 3 seeds the differences between techniques
are not significant. The clear finding is that every C1-based condition is below C0.

Runs:
- C0 `runs/20261105-001629_qwen3.5-0.8b-mlx`
- C1 `runs/20261105-003144_phase1_qwen3.5-0.8b-mlx`
- separate refiner `runs/20261105-020938_phase1_qwen3.5-0.8b-mlx_refiner-gemma4-e4b`
- regression gate `runs/20261105-034043_phase1_qwen3.5-0.8b-mlx_gate`
- memory database `runs/20261105-174004_phase1_qwen3.5-0.8b-mlx_memdb`
- verified skills `runs/20261105-225822_phase1_qwen3.5-0.8b-mlx_verified`
- structured errors `runs/20261106-010306_phase1_qwen3.5-0.8b-mlx_errors`

## Details per technique

**Separate refiner.** A bigger model writing the edits helps most on train tasks (24%), but
held-out stays at C1's level (13%). The capability of the refiner matters, but is not enough.

**Regression gate.** 16 refiner generations checked: kept 13, undone 3. Of the 13 kept, 10 were
ties (7 were 0 vs 0, 3 were 1 vs 1) and only 3 were real gains. With few, mostly failing tasks
the gate cannot tell a good edit from a bad one. Held-out stayed at the baseline (27%).

**Memory database.** The agent used skills more (`run_skill` 20 times vs 3 in C1), but forgetting
did not change: 8 of 12 tasks passed while learning failed at eval.

**Verified skills.** The check works (it rejected 8 broken skills, e.g. code using paths like
`/testbed/...` that do not exist), but qwen rarely called the skills it kept (`run_skill` 3 times).

**Structured errors.** The worst condition. The agent never used skills or sub-agents
(`run_skill` 0, `call_subagent` 0). Longer tool results may crowd an already long prompt.

## Forgetting (passed while learning, failed at eval)

| Condition | Forgotten |
|---|---|
| C1 | 8 of 12 |
| separate refiner | 8 of 13 |
| regression gate | 8 of 14 |
| memory database | 8 of 12 |
| verified skills | 7 of 11 |
| structured errors | 6 of 10 |

## Notes on the runs

- Runs that stopped because of an infrastructure problem (computer sleeping, connection lost)
  are never counted; they are kept in `runs/stopped/` and the condition was rerun from the start.
- Bug fixed during the gate run: `write_file` crashed the run when the model wrote to a folder
  path. It now returns an error to the model. Earlier runs never hit this case (they would have
  crashed), so they are not affected.
