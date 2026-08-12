# ChartReviewBench

A long-horizon **chart-agent benchmark** built on ChartNet.

ChartNet ships `image + code + csv + summary` per chart, so every benchmark item has
**executable ground truth** — we can run, re-render, and verify programmatically instead
of doing image-only QA.

## Two workstreams, one seam

| | What it does | Where |
|---|---|---|
| **Data → task pipeline** | turns ChartNet rows into task instances | *not in this repo yet* |
| **Sandbox runtime** | executes and scores **one task instance** | [`sandbox/`](sandbox/) |

The two connect **only** through the task-instance contract below. Neither side needs to
know the other's internals — keep that seam stable.

## Task-instance contract

Every benchmark item is a directory:

```
<task_id>/
├── task.yaml      # id, family, instruction, subgoals[].verifiers[], budgets
├── workspace/     # copied into the agent's writable workspace
└── oracle/        # reference files; read-only, NEVER exposed to the agent
```

A task is **an ordered list of subgoals, each carrying verifiers** — which is what yields
long-horizon progress curves and partial credit. The exact schema is
[`sandbox/src/chartsandbox/contract.py`](sandbox/src/chartsandbox/contract.py); worked
examples are in `sandbox/tasks/*/task.yaml`.

Task families: `reproduce_edit`, `data_insight`, `debug_repair`, `multi_chart`.

## The sandbox: one environment per benchmark item

`sandbox/` is the execution environment each item is run in. Per episode it gives the
agent a **persistent Python kernel** (state and imports survive across steps), a small
tool set (`execute_python` / `read_file` / `write_file` / `list_files` / `view_image` /
`finish`) path-sandboxed to the workspace, and pluggable verifiers. It writes
`result.json` + `trajectory.jsonl` with score, progress curve, and metrics.

The agent in the loop can be a scripted replay (for testing the sandbox) or a **real
multimodal model** — Claude or GPT, selected by model ID. `view_image` hands the model
the actual bytes of the chart it just rendered, so it can check its own output; the same
models can also score the `vlm_judge` rubrics. Token counts and cost land in
`result.json`.

Full details, extension points, and current status: [`sandbox/README.md`](sandbox/README.md).

## Quickstart

```bash
cd sandbox
uv sync
uv run chartsandbox validate tasks/reproduce_edit_01
uv run chartsandbox run      tasks/reproduce_edit_01
```

`run` drives a **scripted agent** that replays an oracle trajectory — it exercises the
sandbox end to end, it does not measure a model.

## Repo layout

```
ChartReviewBench/
├── README.md         # this file — the shared contract and overview
└── sandbox/          # per-instance execution + scoring runtime (chartsandbox)
    ├── pyproject.toml
    ├── src/chartsandbox/
    ├── scripts/build_examples.py   # stand-in task builder, for smoke tests
    └── tasks/                      # example task instances
```

ChartNet data (the ~1TB dataset and local samples) is **not** committed. Point
`CHARTNET_SAMPLE_DIR` at a local sample directory when regenerating the example tasks.
