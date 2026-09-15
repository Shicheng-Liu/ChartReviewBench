"""Convert ChartRepairBench release rows into chartsandbox task instances.

The release ships its own converter, but that one reads an internal `data/v0/samples`
layout the release does not actually contain. These rows do contain everything except
the images (code, table, summary and the fixed per-track prompt are inline), so the
conversion can run straight off the jsonl.

Subgoals are laid out along the evaluation dimensions, one subgoal per dimension that
can be checked at all:

  code_valid      dimension 1  rule-based: the script runs and writes output.png
  flaw_fixed      dimension 3  judge: the defects that were there are corrected
  visual_quality  dimension 5  judge: the chart is legible and well formed

Dimension 5 is the reason this exists rather than being left to the release: nothing
in the released pipeline scores readability or aesthetics, and it is the one dimension
with no rule-based substitute — a figure property map cannot tell you that two tick
labels collide or that the legend now sits on the data. It also gets a rubric of its
own rather than being folded into the repair rubric, so that a beautiful chart of the
wrong numbers and an ugly chart of the right ones do not cancel out.

Dimensions 2 and 4 (data fidelity, preservation) are deliberately absent: the release
computes both by re-executing the submission and diffing captured figure properties
against the reference, which is a scorer, not a verifier, and reimplementing it here
would produce a second, disagreeing number. Run `scoring/score_submission.py` on the
episode's final chart.py for those. Dimension 6 (efficiency) needs no verifier — the
runner already records iterations, steps, tokens and cost.

    python3 scripts/release_to_tasks.py --track B --n 10
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

WORKSPACE_FILE = {"code": "chart.py", "table": "table.csv",
                  "summary": "summary.txt", "chart": "chart_current.png"}

#: Per-flaw repair rubrics, from the release's own converter. The agent never sees
#: these — they live in task.yaml and oracle/, outside the workspace.
RUBRICS = {
    "wrong-title": "The chart title names the quantity the data actually shows, matching summary.txt, instead of a different quantity.",
    "wrong-axis-label": "The axis labels name the quantities actually plotted, consistent with table.csv and summary.txt.",
    "missing-legend": "The chart carries a legend that identifies the plotted series, as described in summary.txt.",
    "swapped-legend-labels": "Each legend entry names the series it actually belongs to, consistent with table.csv.",
    "value-perturbation": "Every plotted value matches the corresponding number in table.csv.",
    "swapped-series": "Each series plots its own data from table.csv, not another series' values.",
    "dropped-series": "All series described in summary.txt and present in table.csv appear in the chart.",
    "wrong-chart-type": "The chart uses the chart type that summary.txt describes.",
    "axis-truncation": "The value axis covers the range summary.txt describes, so relative differences are not exaggerated.",
    "color-contradicts-summary": "The colors of the plotted elements match the color-to-meaning mapping described in summary.txt.",
}

#: Dimension 5. Scoped to presentation on purpose: if this rubric also asked about
#: correctness it would double-count the repair dimensions, and a judge told to weigh
#: both at once tends to answer only about the numbers.
VISUAL_QUALITY_RUBRIC = (
    "Judge only the legibility and presentation of this chart, not whether its "
    "numbers or labels are factually right. It passes if a reader can take it in "
    "without effort: title, axis labels, tick labels and legend are all readable "
    "and none are clipped by the figure edge, overlapping each other, or covering "
    "the plotted data; text is large enough to read; the plotted data is visible "
    "rather than obscured or squeezed into a corner; nothing is drawn as a missing-"
    "glyph box. It fails if any of those is violated, however correct the content is."
)

CLEAN_RUBRIC = ("The chart is unchanged and still faithful to table.csv and summary.txt: "
                "nothing that was already correct has been altered.")


def repair_rubric(row: dict) -> str:
    flaws = row.get("flaws") or []
    if not flaws:
        return CLEAN_RUBRIC
    parts = [RUBRICS.get(f["category"]) or f.get("description", "") for f in flaws]
    if len(parts) == 1:
        return parts[0]
    joined = " ".join(f"({i}) {p}" for i, p in enumerate(parts, 1))
    return f"All {len(parts)} of the following now hold: {joined}"


def build(row: dict, out_root: Path, release: Path, mods: tuple,
          step_budget: int, max_turns: int, wall_time_s: int | None,
          step_timeout_s: int = 90) -> Path:
    task_dir = out_root / row["id"]
    ws, oracle = task_dir / "workspace", task_dir / "oracle"
    for d in (ws, oracle):
        d.mkdir(parents=True, exist_ok=True)

    text = {"code": row.get("initial_code") or row.get("ground_truth_code"),
            "table": row["table"], "summary": row["summary"]}
    files = []
    for m in mods:
        target = ws / WORKSPACE_FILE[m]
        if m == "chart":
            src = release / (row.get("initial_image") or "")
            if not src.is_file():
                raise FileNotFoundError(f"missing input chart: {src}")
            shutil.copy2(src, target)
        else:
            target.write_text(text[m], encoding="utf-8")
        files.append(WORKSPACE_FILE[m])

    # oracle: reference only, never copied into the workspace
    (oracle / "solution.py").write_text(row["ground_truth_code"], encoding="utf-8")
    gt = release / (row.get("ground_truth_image") or "")
    if gt.exists():
        shutil.copy2(gt, oracle / "reference.png")
    (oracle / "task_config.json").write_text(json.dumps({
        "id": row["id"], "track": row["track"], "clean": bool(row.get("clean")),
        "arity": row["arity"], "n_flaws": row["n_flaws"],
        "combination": row["combination"], "flaws": row.get("flaws") or [],
        "library": row["library"], "chart_type": row["chart_type"],
        "prompt_file": row["prompt_file"], "modalities": list(mods),
    }, indent=2), encoding="utf-8")

    task = {
        "id": row["id"],
        "family": "debug_repair",
        # The release's fixed, checksummed per-track prompt, passed through verbatim.
        "instruction": row["prompt"],
        "horizon_hint": min(6, max_turns) if max_turns else 6,
        "step_budget": step_budget,
        "max_turns": max_turns,
        # 90s per execution, matching the release scorer's own capture timeout.
        # The default 30 is too tight for plotly: kaleido starts a subprocess on its
        # first write_image and the whole turn times out before a chart is written.
        "step_timeout_s": step_timeout_s,
        "wall_time_s": wall_time_s,
        "workspace_files": files,
        "subgoals": [
            {"id": "code_valid", "desc": "The script runs and writes output.png",
             "weight": 1.0,
             "verifiers": [{"type": "execution", "params": {"produces": "output.png"}}]},
            {"id": "flaw_fixed", "desc": "The contradiction with the data is gone",
             "weight": 1.0,
             "verifiers": [{"type": "vlm_judge",
                            "params": {"candidate": "output.png",
                                       "rubric": repair_rubric(row)}}]},
            {"id": "visual_quality", "desc": "The chart is legible and well formed",
             "weight": 1.0,
             "verifiers": [{"type": "vlm_judge",
                            "params": {"candidate": "output.png",
                                       "rubric": VISUAL_QUALITY_RUBRIC}}]},
        ],
    }
    (task_dir / "task.yaml").write_text(
        yaml.safe_dump(task, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return task_dir


def main() -> int:
    import hashlib
    import tempfile
    ap = argparse.ArgumentParser(description="Materialize A/B/C release rows; preserve existing matching tasks")
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--rows", type=Path, help="directory with data/A.jsonl etc; defaults to release")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--track", choices=("A", "B", "C", "all"), default="all")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--step-budget", type=int, default=14)
    ap.add_argument("--wall-time-s", type=int, default=1800)
    ap.add_argument("--step-timeout-s", type=int, default=90)
    ap.add_argument("--uncapped", action="store_true", help="disable episode turn/step/wall caps")
    args = ap.parse_args()
    if args.n < 0 or min(args.max_turns, args.step_budget, args.wall_time_s, args.step_timeout_s) < 1:
        ap.error("invalid count/budget")
    args.release = args.release.resolve()
    rows_root = (args.rows or args.release).resolve()
    args.out = args.out.resolve()
    if args.out == args.release or args.out.is_relative_to(args.release):
        ap.error("output must be outside the release")
    turns, steps, wall = (None, None, None) if args.uncapped else (args.max_turns, args.step_budget, args.wall_time_s)
    for track in ("ABC" if args.track == "all" else args.track):
        with (rows_root / "data" / f"{track}.jsonl").open() as f:
            rows = [json.loads(line) for line in f if line.strip()]
        if args.n:
            rows = rows[:args.n]
        out_root = args.out / f"track{track}"
        out_root.mkdir(parents=True, exist_ok=True)
        built = skipped = 0
        for row in rows:
            if Path(row["id"]).name != row["id"] or row["id"] in (".", ".."):
                raise ValueError("invalid task id")
            identity = {"row": row, "budgets": [turns, steps, wall, args.step_timeout_s]}
            digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            dest = out_root / row["id"]
            marker = dest / "source.sha256"
            if dest.exists():
                if marker.is_file() and marker.read_text().strip() == digest:
                    skipped += 1
                    continue
                raise ValueError(f"existing task differs or is incomplete: {dest}; use a new --out")
            with tempfile.TemporaryDirectory(prefix=".prepare-", dir=args.out) as tmp:
                task = build(row, Path(tmp), args.release, tuple(WORKSPACE_FILE),
                             steps, turns, wall, args.step_timeout_s)
                (task / "source.sha256").write_text(digest + "\n")
                task.rename(dest)
            built += 1
        print(f"Track {track}: {built} built, {skipped} unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
