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
task.yaml ─► Runner ─► [ Runtime  ] persistent Python kernel (state across steps)
                       [ Tools    ] execute_python / read / write / list / view_image / finish
                       [ Agent    ] scripted replay │ real multimodal LLM
                       [ Providers] anthropic (Claude) │ openai (GPT) │ vllm (self-hosted)
                       [ Verify   ] execution · data · vlm_judge · (weighted subgoals)
                     ─► result.json + trajectory.jsonl  (score, progress curve, cost)
```

- **Runtime** (`runtime/`): each episode gets a persistent worker subprocess. Variables
  and imports persist across `execute_python` calls (this is what makes long-horizon,
  stateful work possible). Deterministic rendering (Agg backend, fixed hash seed).
  Per-step SIGALRM timeout + parent-side hard timeout with kill/restart. Best-effort
  network block (`SANDBOX_ALLOW_NET=1` to disable).
- **Tools** (`tools.py`): the agent's action space, path-sandboxed to the workspace so
  the oracle can never be read. `TOOL_SCHEMAS` are provider-neutral tool declarations.
- **Agents** (`agent.py`): `ScriptedAgent` replays a fixed trajectory (sandbox smoke
  test, no API calls); `LLMAgent` drives a real tool-calling model and **sees its own
  charts** — `view_image` output goes back as a real image block, not a description.
- **Providers** (`providers/`): one neutral conversation format, one module per backend —
  Claude, GPT, and open weights served by vLLM. Each handles what differs: image
  placement in tool results, tool-schema shape, structured output, and echoing
  model-internal blocks (Claude thinking signatures, GPT reasoning items) so a chain of
  thought survives across tool calls. They target three different wire protocols
  (Messages, Responses, Chat Completions) behind one interface.
- **Verifiers** (`verifiers/`): a registry of `type -> fn(ctx) -> Result`. Built-ins:
  `execution` (artifact produced / no error), `data` (numeric CSV compare vs oracle),
  `vlm_judge` (rubric scorer — labelled mock by default, real model via `--judge`).
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

`run` defaults to the **scripted (simulated) agent** — it replays an oracle trajectory
from `tasks/<id>/sim_agent.json` to exercise the sandbox end to end, calling no API.
This tests the *sandbox*, not a model.

## Run a real model

```bash
# a real multimodal agent solves the task, a real judge scores the chart
uv run chartsandbox run tasks/debug_repair_01 \
    --agent llm --model claude-opus-5 --judge claude-opus-5

uv run chartsandbox run tasks/debug_repair_01 \
    --agent llm --model gpt-5.4 --judge gpt-5.4
```

Both flags cost money on hosted backends, so neither is on by default. Credentials come
from the environment: `ANTHROPIC_API_KEY` (or an `ant auth login` profile) and
`OPENAI_API_KEY`. Backend is routed from the model ID; for an ID the prefixes don't
cover, name it explicitly — `--model openai:some-new-id`.

### Open-weight models via vLLM

Serve the model yourself, with tool calling turned on — `scripts/serve_vllm.sh` wraps
this, including the separate uv venv vLLM wants:

```bash
vllm serve /data2/Qwen/Qwen3-VL-8B-Instruct \
    --served-model-name Qwen3-VL-8B-Instruct \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --limit-mm-per-prompt '{"image": 16}'
```

then point the harness at it:

```bash
uv run chartsandbox run tasks/debug_repair_01 \
    --agent llm --model vllm:Qwen3-VL-8B-Instruct \
    --judge vllm:Qwen3-VL-8B-Instruct \
    --base-url http://localhost:8000/v1        # or $VLLM_BASE_URL
```

The model id must match `--served-model-name` (default: the path you served, slashes
and all). Self-hosted names are arbitrary, so the `vllm:` prefix is required —
everything after the first colon is the id. Notes specific to this backend:

- **`--effort` is ignored.** Reasoning depth is a property of the weights you loaded,
  so the flag is accepted and dropped rather than sent and rejected.
- **`temperature` defaults to 0** for reproducibility (the frontier APIs no longer take
  the parameter; vLLM still does).
- **The judge uses guided decoding** — `response_format` with a JSON schema, falling
  back to `guided_json` on older servers, with a salvage path if the model wraps its
  JSON in prose anyway. A verdict that still can't be parsed fails closed.
- **`--limit-mm-per-prompt '{"image": N}'`** matters: the agent accumulates one image
  per `view_image` call, so a low limit will truncate long episodes. Pair a small limit
  with `--max-history-images`. (The old `image=N` spelling was dropped in vLLM 0.27 —
  it now takes JSON.)
- **Tool calling is the gate.** A model that can't emit tool calls ends its episode via
  the agent's no-tool-call guard. That is a real capability result, not a harness bug —
  check `agent_stats.transcript` in `result.json` to tell the two apart.
- No pricing: tokens are reported, `cost_usd` is null. Your cost is GPU time.

Useful knobs: `--effort low|medium|high|xhigh|max` (agent reasoning depth),
`--judge-effort`, `--max-tokens` (output cap per call — covers thinking too),
`--max-history-images N` (bound context on long episodes; costs prompt-cache hits).
Token counts and a cost estimate land in `result.json` under `agent_stats` /
`judge_stats`, next to the model's own reasoning summaries per step.

To verify the wiring without spending anything — no API key needed:

```bash
python scripts/dry_run_agent.py
```

It runs a full episode against a canned model and asserts on the request payloads:
that the rendered chart's actual bytes reach the model, that tool_use/tool_result pair
up, that thinking blocks round-trip, that image bytes stay out of the trajectory log,
and that the OpenAI translation degrades correctly.

Rebuild the example tasks from the downloaded ChartNet sample:

```bash
python3 scripts/build_examples.py
```

## Extending

- **Add a verifier**: write `verifiers/<name>.py` with `@register("<type>")`, reference
  `type: <type>` in a subgoal.
- **Add a model backend**: subclass `Provider` (`providers/base.py`), implement
  `complete` and `judge`, and route it in `providers/__init__.py`. Nothing else changes.
- **Swap the judge**: `--judge <model>` installs the built-in one. For your own scorer
  (a local vLLM, an ensemble), install a callable —
  ```python
  from chartsandbox.verifiers import set_judge_backend
  set_judge_backend(lambda rubric, image_path, workspace: {"passed": True, "score": 0.9, "detail": "..."})
  ```
  A backend may also declare `oracle` and `params` to receive the task's reference
  directory and the subgoal's verifier params; both are passed only if its signature
  accepts them.
- **Show the judge the reference chart**: add `reference: reference.png` to a
  `vlm_judge` subgoal's params. Off by default — on an *edit* subgoal the reference is
  the pre-edit chart and argues against the change being graded.

## Status (MVP) and what's deliberately deferred

Working now: contract + validation, stateful runtime, 6 tools, `execution`/`data`/
`vlm_judge` verifiers, subgoal scoring + progress curve, runner + CLI, two live example
tasks passing end to end via the simulated agent — plus a real multimodal agent and a
real rubric judge on three backends (Claude, GPT, self-hosted vLLM), with token/cost
accounting.

Verified end to end on `gpt-5.4`: the agent reads the broken script, fixes it, renders,
**views its own chart**, and reports what it checked; the judge scores the image against
the rubric and cites the visual evidence.

Deferred:
- **Anthropic and vLLM paths exercised only against a fake client** — both follow their
  documented APIs and pass the dry run, but no Claude key and no GPU server were
  available here, so neither has made a live call yet. The vLLM one additionally
  depends on your server's tool-call parser working with the model you load.
- **Cost for GPT models** — `PRICE_PER_MTOK` has Anthropic list prices only; OpenAI runs
  report tokens with `cost_usd: null` rather than a guessed number.
- **Visual-similarity verifier** — intentionally omitted for now.
- **Hard isolation** — local subprocess + best-effort net block; Docker mode for true
  isolation is future work.
- **`data_insight` / `multi_chart` example tasks** — contract already supports them.
```
