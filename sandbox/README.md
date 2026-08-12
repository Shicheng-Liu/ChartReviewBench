# chartsandbox

A **sandbox runtime** for a long-horizon *chart-agent* benchmark built on ChartNet.

This repo is **only the sandbox**: it takes a task instance (produced separately by
the data pipeline), runs an agent inside an isolated, *stateful* Python workspace with
a small tool set, and scores the result with pluggable verifiers. It does **not** know
how tasks are constructed from ChartNet — that is a separate workstream, connected only
through the task-instance contract.

## Why this shape

ChartNet uniquely ships `image + code + csv + summary` per chart, so tasks have
**executable ground truth** — we can run, re-render, and verify programmatically
instead of doing image-only QA. Long-horizon ability is measured by giving the agent
tasks that decompose into many dependent steps and tracking *how far* it gets, not just
final success.

## Core abstraction

> **A task = an ordered list of subgoals; each subgoal carries verifiers.**

This single model covers all four task families (`reproduce_edit`, `data_insight`,
`debug_repair`, `multi_chart`) and yields long-horizon **progress curves** and
**partial credit** for free.

## Architecture

```
task.yaml ─► Runner ─► [ Runtime ] persistent Python kernel (state across steps)
                       [ Tools   ] execute_python / read / write / list / view_image / finish
                       [ Verify  ] execution · data · vlm_judge(mock) · (weighted subgoals)
                     ─► result.json + trajectory.jsonl  (score, progress curve, metrics)
```

- **Runtime** (`runtime/`): each episode gets a persistent worker subprocess. Variables
  and imports persist across `execute_python` calls (this is what makes long-horizon,
  stateful work possible). Deterministic rendering (Agg backend, fixed hash seed).
  Per-step SIGALRM timeout + parent-side hard timeout with kill/restart. Best-effort
  network block (`SANDBOX_ALLOW_NET=1` to disable).
- **Tools** (`tools.py`): the agent's action space, path-sandboxed to the workspace so
  the oracle can never be read. `TOOL_SCHEMAS` are provider-neutral tool declarations.
- **Verifiers** (`verifiers/`): a registry of `type -> fn(ctx) -> Result`. Built-ins:
  `execution` (artifact produced / no error), `data` (numeric CSV compare vs oracle),
  `vlm_judge` (**mocked** rubric scorer, swappable — see below).
- **Runner** (`runner.py`): copies `workspace/` into a fresh run dir, drives the agent
  loop under a step/wall-time budget, checks cheap verifiers each step for the progress
  curve, then runs full verification and writes results.

## Task-instance contract

```
tasks/<id>/
├── task.yaml      # id, family, instruction, subgoals[].verifiers[], budgets
├── workspace/     # copied into the agent's writable workspace
└── oracle/        # reference files; read-only, NEVER exposed to the agent
```

See `src/chartsandbox/contract.py` for the exact schema and `tasks/*/task.yaml` for
worked examples. This file is the seam with the data pipeline — keep it stable.

## Run it

With **uv** (recommended):

```bash
uv sync
uv run chartsandbox validate tasks/reproduce_edit_01
uv run chartsandbox run      tasks/reproduce_edit_01
```

Or with any Python that has the deps:

```bash
PYTHONPATH=src python3 -m chartsandbox.cli run tasks/debug_repair_01
```

`run` uses the **scripted (simulated) agent** — it replays an oracle trajectory from
`tasks/<id>/sim_agent.json` to exercise the sandbox end to end. This is for testing the
*sandbox*, not for measuring a model.

Rebuild the example tasks from the downloaded ChartNet sample:

```bash
python3 scripts/build_examples.py
```

## Extending

- **Add a verifier**: write `verifiers/<name>.py` with `@register("<type>")`, reference
  `type: <type>` in a subgoal.
- **Wire the real judge**: implement one function and install it —
  ```python
  from chartsandbox.verifiers import set_judge_backend
  set_judge_backend(lambda rubric, image_path, workspace: {"passed": True, "score": 0.9, "detail": "..."})
  ```
- **Plug a real LLM agent**: implement `LLMAgent.act` in `agent.py` against
  `tools.TOOL_SCHEMAS`; the runner is unchanged.

## Status (MVP) and what's deliberately deferred

Working now: contract + validation, stateful runtime, 4 tools, `execution`/`data`/mock
`vlm_judge` verifiers, subgoal scoring + progress curve, runner + CLI, two live example
tasks passing end to end via the simulated agent.

Deferred (hooks already in place):
- **Real judge API** — `vlm_judge` is a labelled mock; `set_judge_backend` swaps it in.
- **Real LLM agent** — `LLMAgent` skeleton only.
- **Visual-similarity verifier** — intentionally omitted for now.
- **Hard isolation** — local subprocess + best-effort net block; Docker mode for true
  isolation is future work.
- **`data_insight` / `multi_chart` example tasks** — contract already supports them.
```
