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

## Our port (Phase 1, condition C1) — `harness/refiner.py`, `harness/state.py`, `scripts/run_phase1.py`

Kept the same as the original:
- Schedule: warmup 25 steps, then every 25 steps up to 200, then every 100.
- Window: last 25 steps (prompt, sub-agents, skills), last 20 steps (memory).
- Four passes in order, each in its own try/except. Same model does tasks and refining.
- Prompt pass returns the whole strategy as markdown; other passes return JSON, parsed the same way
  (strip fences, json.loads, else first `{...}` block). Pass prompts are the original texts with game words
  replaced by task words.
- Fixed system prompt + editable strategy ("base prompt"). Every edit is kept (no acceptance test).
- Sub-agent tools filtered to an allowlist; max_turns capped (original cap 50, ours 8).

Changed for our setting:
- Edits happen between tasks (steps are counted across tasks; if a task passes an evolution point, the Refiner runs after it).
- The trajectory shows each task's result (PASS/FAIL), standing in for game progress in the original.
- Skill code runs as a Python file inside the sandbox in the task folder (no game `tools[...]` API).
- Skill code is kept when a skill is added (the original's add pass does not save `code`; only update does).
- Sub-agent fields reduced (no handler_type/directive/return_condition).
- The agent does not get process_memory / run_code tools: memory is shown in the system prompt,
  and run_shell already runs code. run_skill and call_subagent appear only once the harness has skills / sub-agents,
  so before the first edit the C1 agent is exactly the C0 baseline agent.
- Caps: strategy 4,000 chars; 20 memory entries (500 chars each); 10 skills (3,000 chars of code); 5 sub-agents.
- Safety and RAM: sandbox for run_shell and run_skill; Refiner has no tools and only returns text;
  model unloaded after every task and Refiner generation; run stops above 10 GB; num_ctx fixed at 32768 (same as baseline).
