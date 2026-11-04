# Notes on the original Continual Harness code

Source: https://github.com/sethkarten/continual-harness (MIT license).
Paper: Karten et al., 2026, arXiv:2605.09998 (preprint, CC BY-NC-SA 4.0 — do not copy text or figures).

## What we must rebuild for Ollama

The original code only supports cloud APIs (Gemini, OpenAI, Anthropic, OpenRouter, Vertex).
It is also tied to Pokémon emulators. So we re-implement the same design for our tasks.

## How their evolution works (from `System-Design/architecture/harness_evolver.md`)

- Class `HarnessEvolver` in `agents/utils/harness_evolver.py`.
- **When:** after each agent step, `should_evolve(step)` is checked.
  - No evolution before step 25 (`MIN_WARMUP_STEPS = 25`).
  - Up to step 200: evolve every 25 steps. After 200: every 100 steps.
- **What it reads:** the recent trajectory (steps saved in `trajectory_history.jsonl`).
- **Four passes, in order** (each in its own try/except, so one failing does not stop the others):
  1. **prompt** — rewrite the base system prompt.
  2. **subagents** — model returns JSON `create` / `update` / `retire`.
  3. **skills** — JSON `add` / `update` (code the agent can run with `run_skill`).
  4. **memory** — JSON `add` / `update`, using only the last ~20 steps.
- **Where it saves:** `subagents.json`, `skills.json`, `memory.json`, and an audit log
  `evolution_log.jsonl` (one line per evolution: generation, step, summaries).
- The same model that plays also does the refining (no separate teacher).
- There is **no acceptance test** for an edit: every edit is kept.

## What this means for our design

- Our trajectory log (`runs/.../trajectory.jsonl`) uses the same idea: one JSON line per step.
- Our stores will be JSON files at first (same as theirs), then a SQLite memory database as a technique.
- "Step" counts across tasks, so evolution can happen between tasks.
- Our added techniques target the weak points above: same-model refining, no edit gate, raw error text.
