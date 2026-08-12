"""Build example task instances from the downloaded core_permissive sample.

This stands in for the (separate) data pipeline just enough to smoke-test the
sandbox. It emits two tasks under sandbox/tasks/:

  1. reproduce_edit_01  — reproduce a chart, then apply an edit (multi-step chain)
  2. debug_repair_01    — fix intentionally-broken plotting code so it renders

Each task also gets a sim_agent.json: an oracle trajectory the ScriptedAgent
replays so we can see a full green run. Real agents would solve these themselves.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

SANDBOX = Path(__file__).resolve().parents[1]
TASKS = SANDBOX / "tasks"

# The ChartNet sample lives outside the repo (it is not committed). Point
# CHARTNET_SAMPLE_DIR at a directory holding metadata.jsonl + the referenced
# code/csv/image files; otherwise we try the usual sibling checkout layouts.
_SAMPLE_CANDIDATES = [
    SANDBOX.parents[1] / "data" / "core_permissive_sample_20",   # <parent>/data next to the repo
    SANDBOX.parent / "data" / "core_permissive_sample_20",       # <repo>/data
]


def _resolve_data() -> Path:
    env = os.environ.get("CHARTNET_SAMPLE_DIR")
    candidates = [Path(env)] if env else _SAMPLE_CANDIDATES
    for c in candidates:
        if (c / "metadata.jsonl").exists():
            return c
    raise SystemExit(
        "could not find the ChartNet sample. Set CHARTNET_SAMPLE_DIR to a directory "
        "containing metadata.jsonl. Tried: " + ", ".join(str(c) for c in candidates)
    )


def _load_rows():
    data = _resolve_data()
    rows = [json.loads(l) for l in (data / "metadata.jsonl").read_text().splitlines()]
    for r in rows:
        r["_code"] = (data / r["code"]).read_text()
        r["_csv"] = (data / r["csv"]).read_text()
        r["_img"] = data / r["image"]
    return rows


def _to_out_png(code: str) -> str:
    """Rewrite the row's savefig target to the canonical out.png."""
    code = re.sub(r"savefig\(\s*['\"][^'\"]+['\"]", "savefig('out.png'", code)
    if "savefig(" not in code:
        code += "\nimport matplotlib.pyplot as plt\nplt.savefig('out.png', dpi=150, bbox_inches='tight')\n"
    return code


def _pick(rows, n):
    """Pick matplotlib/seaborn rows that save a figure (skip plotly/kaleido)."""
    good = [r for r in rows if r["library"] in ("matplotlib", "seaborn") and "savefig" in r["_code"]]
    return good[:n]


def _write(task_dir: Path, task_yaml: str, workspace: dict, oracle: dict, sim: list):
    (task_dir / "workspace").mkdir(parents=True, exist_ok=True)
    (task_dir / "oracle").mkdir(parents=True, exist_ok=True)
    (task_dir / "task.yaml").write_text(task_yaml)
    for name, content in workspace.items():
        (task_dir / "workspace" / name).write_text(content)
    for name, content in oracle.items():
        p = task_dir / "oracle" / name
        if isinstance(content, bytes):
            p.write_bytes(content)
        else:
            p.write_text(content)
    (task_dir / "sim_agent.json").write_text(json.dumps(sim, indent=2))


def build_reproduce_edit(row):
    td = TASKS / "reproduce_edit_01"
    clean = _to_out_png(row["_code"])
    edited = clean.replace(
        "savefig('out.png'", "title('EDITED: ' + plt.gca().get_title())\nplt.savefig('out.png'", 1
    ) if "plt." in clean else clean
    instruction = (
        f"Reproduce the reference chart, then edit it.\n\n"
        f"Chart type: {row['chart_type']} (library: {row['library']}).\n"
        f"Summary of the target chart:\n{row['summary'][:600]}\n\n"
        f"Step 1: produce the chart and save it as out.png.\n"
        f"Step 2: prepend 'EDITED: ' to the chart title and save out.png again.\n"
        f"data.csv holds the underlying data."
    )
    task_yaml = f"""id: reproduce_edit_01
family: reproduce_edit
instruction: |
{_indent(instruction)}
horizon_hint: 8
step_budget: 20
workspace_files: [data.csv]
subgoals:
  - id: reproduce_base
    desc: Base chart reproduced and saved as out.png
    weight: 0.6
    verifiers:
      - type: execution
        params: {{produces: out.png}}
      - type: vlm_judge
        params: {{candidate: out.png, rubric: "chart matches the described {row['chart_type']}"}}
  - id: apply_edit
    desc: Title edited (prefixed with 'EDITED:')
    weight: 0.4
    verifiers:
      - type: execution
        params: {{produces: out.png}}
      - type: vlm_judge
        params: {{candidate: out.png, rubric: "chart title now begins with 'EDITED:'"}}
"""
    sim = [
        {"tool": "list_files", "args": {}},
        {"tool": "read_file", "args": {"path": "data.csv"}},
        {"tool": "write_file", "args": {"path": "plot.py", "content": clean}},
        {"tool": "execute_python", "args": {"code": "exec(compile(open('plot.py').read(), 'plot.py', 'exec'))"}},
        {"tool": "view_image", "args": {"path": "out.png"}},
        {"tool": "write_file", "args": {"path": "plot.py", "content": edited}},
        {"tool": "execute_python", "args": {"code": "exec(compile(open('plot.py').read(), 'plot.py', 'exec'))"}},
        {"tool": "view_image", "args": {"path": "out.png"}},
        {"tool": "finish", "args": {"message": "reproduced and edited"}},
    ]
    _write(td, task_yaml,
           workspace={"data.csv": row["_csv"]},
           oracle={"reference.png": row["_img"].read_bytes(), "solution.py": clean},
           sim=sim)
    return td


def build_debug_repair(row):
    td = TASKS / "debug_repair_01"
    clean = _to_out_png(row["_code"])
    broken = clean.replace("savefig", "saveFig", 1)  # AttributeError at runtime
    instruction = (
        f"broken.py is supposed to render a {row['chart_type']} and save it as out.png, "
        f"but it currently crashes. Find and fix the bug so it runs and produces out.png. "
        f"Do not change what the chart shows."
    )
    task_yaml = f"""id: debug_repair_01
family: debug_repair
instruction: |
{_indent(instruction)}
horizon_hint: 5
step_budget: 15
workspace_files: [broken.py]
subgoals:
  - id: renders_ok
    desc: Fixed script runs without error and produces out.png
    weight: 1.0
    verifiers:
      - type: execution
        params: {{produces: out.png}}
      - type: vlm_judge
        params: {{candidate: out.png, rubric: "a valid {row['chart_type']} was produced"}}
"""
    sim = [
        {"tool": "read_file", "args": {"path": "broken.py"}},
        {"tool": "execute_python", "args": {"code": "exec(compile(open('broken.py').read(), 'broken.py', 'exec'))"}},
        {"tool": "write_file", "args": {"path": "broken.py", "content": clean}},
        {"tool": "execute_python", "args": {"code": "exec(compile(open('broken.py').read(), 'broken.py', 'exec'))"}},
        {"tool": "view_image", "args": {"path": "out.png"}},
        {"tool": "finish", "args": {"message": "fixed typo saveFig -> savefig"}},
    ]
    _write(td, task_yaml,
           workspace={"broken.py": broken},
           oracle={"reference.png": row["_img"].read_bytes(), "solution.py": clean},
           sim=sim)
    return td


def _indent(text: str, n: int = 2) -> str:
    pad = " " * n
    return "\n".join(pad + line for line in text.splitlines())


def main():
    rows = _load_rows()
    picks = _pick(rows, 2)
    if len(picks) < 2:
        raise SystemExit("need >=2 matplotlib/seaborn rows with savefig in the sample")
    a = build_reproduce_edit(picks[0])
    b = build_debug_repair(picks[1])
    print("built:")
    print("  ", a)
    print("  ", b)


if __name__ == "__main__":
    main()
