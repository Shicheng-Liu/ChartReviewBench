> **A/B/C 全量、自动续跑与 OpenRouter：见 [RUNNING_ABC.md](RUNNING_ABC.md)。**

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
                       [ Agent    ] scripted replay │ tagged-text LLM │ tool-calling LLM
                       [ Providers] anthropic (Claude) │ openai (GPT) │ vllm (self-hosted)
                       [ Verify   ] execution · data · vlm_judge · (weighted subgoals)
                     ─► result.json + trajectory.jsonl  (score, progress curve, cost)
```

- **Runtime** (`runtime/`): each episode gets a persistent worker subprocess. Variables
  and imports persist across `execute_python` calls (this is what makes long-horizon,
  stateful work possible). Deterministic rendering (Agg backend, fixed hash seed).
  Bounded three ways, because one bound does not cover the others: SIGALRM caps
  wall-clock per execution, `RLIMIT_CPU` caps CPU (which is what catches a C-level loop
  that never reaches a bytecode boundary where Python could deliver SIGALRM), and
  `RLIMIT_AS` caps address space (`SANDBOX_MEM_MB`, default 8192) so a runaway
  allocation becomes a `MemoryError` the agent can read rather than an OOM kill it
  cannot. Parent-side hard timeout with kill/restart is the last resort. Best-effort
  network block (`SANDBOX_ALLOW_NET=1` to disable).
  Each execution returns stdout, stderr, a traceback, **captured warnings**
  (deduplicated — "runs without errors or critical warnings" is a scored property, and
  a warning is often the only trace of a rendering defect: a missing glyph draws a box,
  a failed `tight_layout` crops a label), the image files it wrote, and any figure it
  left open but never saved (`figures_captured`, written under `.sandbox_figures/` and
  never passed off as the agent's deliverable).
- **Tools** (`tools.py`): the agent's action space, path-sandboxed to the workspace so
  the oracle can never be read. `TOOL_SCHEMAS` are provider-neutral tool declarations.
- **Agents** (`agent.py`): `ScriptedAgent` replays a fixed trajectory (sandbox smoke
  test, no API calls); `TaggedAgent` and `LLMAgent` drive a real model under the two
  response protocols below. Both **see their own charts** — image bytes go back as
  real image blocks, not descriptions of one.
- **Providers** (`providers/`): one neutral conversation format, one module per backend —
  Claude, GPT, DeepSeek, and open weights served by vLLM. Each handles what differs: image
  placement in tool results, tool-schema shape, structured output, and echoing
  model-internal blocks (Claude thinking signatures, GPT reasoning items) so a chain of
  thought survives across tool calls. They target three different wire protocols
  (Messages, Responses, Chat Completions) behind one interface.
- **Verifiers** (`verifiers/`): a registry of `type -> fn(ctx) -> Result`. Built-ins:
  `execution` (artifact produced / no error), `data` (numeric CSV compare vs oracle),
  `vlm_judge` (rubric scorer — labelled mock by default, real model via `--judge`).
- **Runner** (`runner.py`): copies `workspace/` into a fresh run dir, drives the agent
  loop under a step/wall-time budget, checks cheap verifiers each step for the progress
  curve, then runs full verification and writes results. It also keeps a ledger of the
  last *readable* version of every image the agent rendered: the protocol scores an
  episode on the latest valid chart, so a final turn that raises part-way through
  writing its own output file is rolled back rather than scored on the wreckage
  (`restored_renders` in the result says when that happened).

## Response protocols

A real model acts under one of two protocols, chosen with `--protocol`. Same task,
same kernel, same verifiers — only how the model says what to do differs.

**`xml` (default)** — tagged text, one *iteration* per turn:

```
<reasoning>
analysis of the current chart, the issues identified, and the planned fixes
</reasoning>
<code>
# Python/Matplotlib code
</code>
<decision>continue</decision>
```

The code runs in the persistent kernel, and **the images it wrote come back
automatically** as image blocks on the next turn — inspection is not something the
model can skip. `<decision>` is the only early exit: `stop` ends the episode (code in
the same turn still runs first, so a final chart is never discarded), `continue` buys
another iteration up to `max_turns` (or indefinitely, when that is `null`). Parsing is
forgiving — a fenced block instead of
`<code>`, a missing `<decision>`, a truncated closing tag are recovered and counted in
`agent_stats.format_warnings` rather than wasting an iteration. See `protocol.py`.

**`tools`** — tool calling, one tool call per step, the model decides when to look at
an image and calls `finish` when done. Parallel tool use is disabled so the
trajectory, the progress curve and the kernel stay in lockstep.

Why keep both: a self-hosted open-weight model needs a working tool-call parser on the
serving side to be evaluable at all, and when that parser is weak the episode fails for
reasons that have nothing to do with charts. Tagged text needs nothing but text
generation. Running one model under both protocols is the response-format ablation.

### Stopping is the agent's decision

**No verifier ever ends an episode.** The environment executes what it is given, hands
back stdout/stderr and the rendered image, and stops when the agent says stop, when the
iteration budget runs out, or when the model cannot produce a parseable action. Ending
early on a passing verifier would leak the grader's verdict into the episode as a free
"you got it right" signal, inflate scores, and make stopping behaviour unobservable.

`result.json` records this at two levels. The episode level says how the loop ended and
who ended it:

| field | values |
| --- | --- |
| `stop_reason` | `finished` · `step_budget_exhausted` · `wall_time_exceeded` |
| `finish_origin` | `agent` · `no_tool_guard` · `parse_guard` · `turn_budget` (null unless finished) |

`finish_origin` is the one that matters for stopping analysis: it separates a `finish`
the agent chose from one a harness guard produced, without anyone having to sniff the
finish message. `agent_stats.stop_reason` then gives the agent's own finer account:

| `stop_reason` | meaning |
| --- | --- |
| `agent_stop` | the agent chose to stop (`<decision>stop</decision>`, or `finish` under `tools`) |
| `turn_budget_exhausted` | `max_turns` iterations spent without the agent stopping (impossible when uncapped) |
| `parse_failure` | no parseable action after the format reminders (`xml`) |
| `no_tool_call` | text with no tool call after the nudges (`tools`) |
| `null` | cut off by `step_budget` or `wall_time_s` |

With `decisions` (the `continue`/`stop` trail) and `first_pass_step` per subgoal, that
is what makes a *wrong* stopping decision — quitting on an unverified chart, or burning
iterations on one that was already correct — analysable after the run.

## Task-instance contract

```
tasks/<id>/
├── task.yaml      # id, family, instruction, subgoals[].verifiers[], budgets
├── workspace/     # copied into the agent's writable workspace
└── oracle/        # reference files; read-only, NEVER exposed to the agent
```

See `src/chartsandbox/contract.py` for the exact schema and `tasks/*/task.yaml` for
worked examples. This file is the seam with the data pipeline — keep it stable.

Two budgets, one per protocol: `max_turns` (default 10) caps iterations under `xml`,
`step_budget` (default 30) caps tool calls under `tools`. **Either may be `null`, which
means uncapped** — the released A/B/C suite (protocol `abc-uncapped-v1`) selects that on
purpose, so that the agent's own `stop` is the only thing that ends an episode. Finite
per-step guards (`step_timeout_s`, and the HTTP and kernel timeouts under it) still
apply; `wall_time_s` may also be `null`.

Where both are finite they are reconciled at reset rather than at load: an iteration
costs a step and closing the episode costs one more, so a task with fewer steps than the
requested iterations has its cap clamped down, and `agent_stats.max_turns_requested`
records what was asked for. This is why an interaction-budget sweep at 1, 2, 3, 5, 10
steps loads fine — a small step budget is a sweep rung, not a malformed task.

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

# the same model under tool calling instead of tagged text
uv run chartsandbox run tasks/debug_repair_01 \
    --agent llm --model claude-opus-5 --protocol tools

# turn-count ablation: cap iterations below the task's own max_turns
uv run chartsandbox run tasks/debug_repair_01 \
    --agent llm --model claude-opus-5 --max-turns 3

# the released protocol: no cap at all, the agent decides when to stop
uv run chartsandbox run tasks/debug_repair_01 \
    --agent llm --model claude-opus-5 --no-turn-limit
```

Both flags cost money on hosted backends, so neither is on by default. Credentials come
from the environment: `ANTHROPIC_API_KEY` (or an `ant auth login` profile) and
`OPENAI_API_KEY`. Backend is routed from the model ID; for an ID the prefixes don't
cover, name it explicitly — `--model openai:some-new-id`.

### DeepSeek

```bash
export DEEPSEEK_API_KEY=...
uv run chartsandbox run <task> --agent llm --model deepseek-v4-flash-vision-exp
```

OpenAI-compatible Chat Completions, so it reuses the vLLM translation, with three
differences that all bite in practice:

- **Reasoning arrives in its own field.** The answer is in `message.content`, the chain
  of thought in `message.reasoning_content`. A turn whose whole budget went to
  reasoning therefore returns *empty content* with a normal `finish_reason` — which
  looks exactly like a model that refused to answer. It is captured into the
  trajectory's `thinking` and never echoed back (the API rejects a replayed
  `reasoning_content`).
- **`max_tokens` covers the reasoning**, so the default is 16000 rather than the vLLM
  4096. Too small a budget does not truncate the answer, it deletes it.
- **No pricing on file**: tokens are reported, `cost_usd` stays null.

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
- **`--limit-mm-per-prompt '{"image": N}'`** matters: the agent accumulates roughly one
  image per iteration, so a low limit will truncate long episodes. Pair a small limit
  with `--max-history-images`, or lower `--max-images-per-turn`. (The old `image=N`
  spelling was dropped in vLLM 0.27 — it now takes JSON.)
- **`--protocol xml` needs no tool-call parser.** The default protocol sends no tool
  declarations at all, so `--enable-auto-tool-choice --tool-call-parser hermes` are only
  required for `--protocol tools`. This is the point: a model whose tool-call parser is
  weak or missing is still evaluable, and it fails on charts rather than on plumbing.
  Under `--protocol tools`, a model that can't emit tool calls ends its episode via the
  no-tool-call guard (`stop_reason: no_tool_call`) — a real capability result, not a
  harness bug. Under `xml`, the equivalent is `stop_reason: parse_failure`; check
  `agent_stats.format_warnings` and `.transcript` to tell a format problem from a
  capability one.
- No pricing: tokens are reported, `cost_usd` is null. Your cost is GPU time.

Useful knobs: `--protocol xml|tools`, `--max-turns N` (iteration cap for `xml`;
defaults to the task's `max_turns`, clamped down if the step budget cannot fit it),
`--no-turn-limit` (run uncapped), `--max-images-per-turn N` (default 4),
`--effort low|medium|high|xhigh|max` (agent reasoning depth), `--judge-effort`,
`--max-tokens` (output cap per call — covers thinking too), `--max-history-images N`
(bound context on long episodes; costs prompt-cache hits).
Token counts and a cost estimate land in `result.json` under `agent_stats` /
`judge_stats`, next to the model's own reasoning summaries per step.

To verify the wiring without spending anything — no API key needed:

```bash
python scripts/dry_run_agent.py     # tool-calling protocol
python scripts/dry_run_tagged.py    # tagged-text protocol
python scripts/dry_run_sandbox.py   # the environment core itself
```

`dry_run_sandbox.py` is one assertion per box of the environment design — resource
control, deterministic rendering, output capture, error capture, workspace
confinement, and the latest-valid-chart rule. Each of those fails quietly when it
regresses: a missing memory ceiling only shows up as a machine under load, an
uncaptured warning only as an unexplained visual-quality score.

Each runs full episodes against a canned model and asserts on the request payloads.
`dry_run_agent.py` covers the rendered chart's actual bytes reaching the model,
tool_use/tool_result pairing, thinking blocks round-tripping, image bytes staying out
of the trajectory log, and the OpenAI translation degrading correctly.
`dry_run_tagged.py` covers tag parsing and its recoverable deviations, the chart being
attached without being asked for, `stop` arriving with code still running that code
first, the iteration budget binding, and each `stop_reason` being reported for the
right reason.

Rebuild the example tasks from the downloaded ChartNet sample:

```bash
python3 scripts/build_examples.py
```

## Token accounting

Every episode records what it spent, in `result.json` under `token_usage`:

```json
"token_usage": {
  "agent":  {"calls": 5, "input_tokens": 18429, "output_tokens": 7553,
             "reasoning_tokens": 4102, "cache_read_input_tokens": 9216,
             "total_tokens": 25982, "model": "...", "cost_usd": null},
  "judge":  {"calls": 2, "...": "..."},
  "total":  {"calls": 7, "input_tokens": ..., "output_tokens": ..., "cost_usd": null}
}
```

Three details are what make this usable for costing a full run rather than just
looking complete:

- **Reasoning tokens are counted.** They are billed as output but arrive nested under
  a details object, so a naive read of `completion_tokens` alone reports the right
  total and the wrong composition. On a reasoning model they can be most of the
  output.
- **Cache hits are counted.** Cached input is billed well below the base rate, and it
  is reported as a *subset* of `input_tokens` rather than subtracted, so nothing is
  double-counted and nothing goes missing.
- **The judge is billed per episode, not per lifetime.** A judge is installed once and
  then scores every task in a suite; its running totals would charge each task for all
  of its predecessors. `Provider.mark()` / `episode_stats()` window the call log, and
  `scripts/run_suite.py` does this for you.

`cost_usd` is `null` for any model with no list price in `PRICE_PER_MTOK` — a partial
sum would read as a complete one. Price such a run at report time:

```bash
python scripts/token_report.py <run_dir> --project 1900 --price-in 0.28 --price-out 0.42
```

which rolls the episodes up and projects them to a full track, as a range rather than
a point: episode cost is driven by iteration count, which varies several-fold, and a
mean alone hides the tail that decides whether a run fits a budget.

## Embedding the sandbox in another harness

The A/B/C release drives this package directly rather than through the CLI, so treat
this as the surface that has to stay stable:

```python
from chartsandbox.runner import run_episode, Sandbox, VerifyContext, progress_curve
from chartsandbox.contract  import Task, LoadedTask, load_task
from chartsandbox.agent     import LLMAgent, ScriptedAgent, SYSTEM_PROMPT
from chartsandbox.tools     import TOOL_SCHEMAS, PersistentKernel   # kernel is swappable
from chartsandbox.judge     import make_judge_backend, install_judge
from chartsandbox.verifiers import set_judge_backend
```

A harness that drives the loop itself also reaches for `runner._dispatch`,
`_summarize_obs`, `_eval_subgoal` and `_initial_observation`. Those are underscored but
in practice load-bearing, so keep their signatures additive — `_initial_observation`
takes its `agent` argument optionally for exactly this reason. Two integration points
worth knowing about:

- **The kernel is replaceable.** `tools.PersistentKernel` is rebound by the release to a
  Seatbelt-confined worker; `Sandbox` resolves it through that name, so a stricter
  runtime can be dropped in without forking this package.
- **`_dispatch` is wrappable.** The release wraps it to turn a malformed tool call into
  a failed observation instead of a `TypeError`, so the agent can correct its own call.

Because budgets are nullable here now, an external nullable-budget adapter is no longer
needed: `run_episode` handles `step_budget: null` / `wall_time_s: null` itself and emits
the same `stop_reason` / `finish_origin` / `episode_wall_time_limit_s` such an adapter
would have added.

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
