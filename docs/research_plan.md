# Research plan

## Working title

Making Continual Harness Work for Small Local Models

## Research question

Continual Harness (Karten et al., 2026, arXiv:2605.09998) improves an agent's own harness
(system prompt, sub-agents, skills, memory) while the agent runs. It was tested only with
large Gemini models. The authors report that weaker models fall below a "capability floor",
and that open models up to 31B could not act as both teacher and student.

**Can a small model (under 5B parameters) running locally on a laptop improve its own
harness, and which added techniques make this possible?**

## Hypotheses

- H1. Continual Harness ported as-is to a 2B to 5B model gives little or no gain, or makes results worse.
- H2. Each added technique raises the gain on its own: memory database, separate refiner,
  verified skill library, structured error feedback, constrained tool calls, regression gate.
- H3. A bigger local refiner (e.g. gemma4:e4b) improving a smaller worker (gemma4:e2b)
  gives more gain than self-refinement.
- H4. Without a regression gate, harness edits cause forgetting: tasks that passed before start failing.

## Conditions

| ID | Condition |
|----|-----------|
| C0 | Static harness (baseline, this repo's Phase 0) |
| C1 | Continual Harness ported as-is (same model refines itself) |
| C2–C7 | C1 + one technique each (ablation) |
| C8 | Full system (all techniques) |

## Models

- Worker: `gemma4:e2b-it-qat`, `qwen3.5:0.8b-mlx` (plus one more size later, e.g. `gemma4:e4b`)
- Embeddings for memory retrieval: `embeddinggemma`

## Tasks

Coding plus PC tasks (files, shell, data). Every task runs in its own temporary folder,
and an automatic checker looks at the final state of that folder (pass or fail).
Split: train tasks (the harness may learn from them) and held-out tasks (only evaluated).
The task set is fixed before seeing any results.

## Metrics

- Success rate (with 95% confidence intervals over seeds)
- Held-out success rate
- Backward transfer (forgetting): change in success on earlier tasks after harness edits
- Cost: wall-clock time, tokens, number of steps
- Tool-call format errors

## Rigor checklist

- 3+ seeds per condition, task order shuffled per seed
- Held-out tasks never seen by the refiner
- Every harness edit logged (before and after)
- Baselines: Reflexion-style memory, Self-Harness-style gated edits
- Report everything, including negative results

## Phases

0. Baseline agent, pilot tasks, checkers, timing (this phase)
1. Port Continual Harness refiner to Ollama (C1)
2. Add techniques one at a time (C2–C7)
3. Full grid runs, analysis
4. Paper writing, arXiv, workshop submission
