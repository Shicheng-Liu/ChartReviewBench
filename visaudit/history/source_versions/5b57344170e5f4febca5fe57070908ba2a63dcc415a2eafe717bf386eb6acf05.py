"""Turn our instances into sandbox task directories.

This is the seam: the sandbox only ever reads its task contract (task.yaml +
workspace/ + oracle/), so the conversion happens here and neither side needs to
know the other's internals. `sandbox_run` calls this on the way into an episode.

Which of the four modalities the agent gets is a configuration, not a constant:
the default is all four, and a smaller set is the same instance at a higher
difficulty. Only the requested modalities are copied into the workspace, the
instruction names exactly those files, and the set is written into the oracle so
a scored episode can be traced back to what the agent actually saw.

The agent never sees the category and never sees the oracle.
"""
import json
import random
import shutil
from pathlib import Path

from .modalities import DEFAULT, WORKSPACE_FILE
from .modalities import code as modality_code
from . import tracks
from .modalities import instruction, parse, validate, workspace_files

SAMPLES = Path("data/v0/samples")

# The default set's text, kept under its old name because that is what this
# module published. The variants are derived from it in `modalities`.
INSTRUCTION = instruction(DEFAULT)

# What "repaired" means, for the judge only. The agent never sees these.
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
GENERIC_RUBRIC = ("The chart no longer contradicts table.csv or summary.txt: "
                  "the flaw described below is gone. {desc}")


def task_id(instance_name: str, category: str, mods=None, track=None,
            step_budget=tracks.DEFAULT_STEP_BUDGET, clean=False,
            *, wall_time_s=tracks.DEFAULT_WALL_TIME_S) -> str:
    """The task directory name for one instance, category, modality set and
    protocol.

    The default set keeps the two-part id it always had, so task dirs built
    before modalities were configurable stay addressable; Track B with the
    default step budget adds nothing either.
    """
    mods = parse(mods)
    base = "%s__%s" % (instance_name, category)
    if mods != DEFAULT:
        base = "%s__m-%s" % (base, modality_code(mods))
    return base + tracks.task_suffix(track, step_budget, clean, wall_time_s=wall_time_s)


def build_task(instance: Path, category: str, out_root: Path, modalities=None,
               track=None, step_budget=tracks.DEFAULT_STEP_BUDGET, clean=False,
               *, wall_time_s=tracks.DEFAULT_WALL_TIME_S) -> Path:
    track = tracks.validate_track(track)
    step_budget = tracks.validate_step_budget(step_budget)
    wall_time_s = tracks.validate_wall_time(wall_time_s)
    flawed_dir = instance / "perturbed_llm" / category
    if clean:
        if track != "C":
            raise ValueError("a clean twin only exists under Track C")
        # Clean inputs ship GT only. Historical twins retain their paired
        # category; additional clean-source tasks use the category "clean".
        meta = {
            "sample_folder": instance.name,
            "category": category,
            "clean": True,
            "description": "ground-truth chart; nothing contradicts the data",
        }
    else:
        meta = json.loads((flawed_dir / "perturbation.json").read_text(encoding="utf-8"))
    wanted = validate(modalities)
    tid = task_id(instance.name, category, wanted, track, step_budget, clean,
                  wall_time_s=wall_time_s)
    task_dir = out_root / tid
    ws, oracle = task_dir / "workspace", task_dir / "oracle"
    for d in (ws, oracle):
        d.mkdir(parents=True, exist_ok=True)

    # workspace: exactly what the agent may see. A modality whose source file is
    # missing (a handful of instances have no flawed render) is dropped from the
    # set rather than promised in the instruction and then not delivered.
    gt_render = instance / "gt" / "render.png"
    if clean and not gt_render.exists():
        gt_render = instance / "gt" / "chart.png"
    sources = {
        "chart": gt_render if clean else flawed_dir / "render.png",
        "code": instance / "gt" / "code.py" if clean else flawed_dir / "code.py",
        "table": instance / "gt" / "table.csv",
        "summary": instance / "gt" / "summary.txt",
    }
    missing = [m for m in wanted if not sources[m].exists()]
    mods = tuple(m for m in wanted if m not in missing)
    if missing:
        try:
            validate(mods)
        except ValueError as exc:
            raise FileNotFoundError(
                "%s: %s missing, and what is left does not define a task (%s)"
                % (tid, ", ".join(str(sources[m]) for m in missing), exc)) from exc
    for m in mods:
        shutil.copy2(sources[m], ws / WORKSPACE_FILE[m])

    # oracle: reference only, never exposed to the agent, and always complete
    # whatever the agent was given.
    shutil.copy2(instance / "gt" / "code.py", oracle / "solution.py")
    if (instance / "gt" / "render.png").exists():
        shutil.copy2(instance / "gt" / "render.png", oracle / "reference.png")
    (oracle / "perturbation.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    files = workspace_files(mods)
    (oracle / "task_config.json").write_text(json.dumps({
        "task_id": tid,
        "sample_folder": instance.name,
        "category": category,
        "modalities": list(mods),
        "modality_code": modality_code(mods),
        "requested": list(wanted),
        "missing_sources": [str(sources[m]) for m in missing],
        "workspace_files": files,
        "track": track,
        "step_budget": step_budget,
        "wall_time_s": wall_time_s,
        "clean": bool(clean),
    }, indent=1), encoding="utf-8")

    rubric = RUBRICS.get(category) or GENERIC_RUBRIC.format(desc=meta.get("description", ""))
    if clean:
        rubric = "The chart is unchanged and still faithful to table.csv and summary.txt."
    task = {
        "id": tid,
        "family": "debug_repair",
        "instruction": tracks.instruction_for(instruction(mods), track, category, clean),
        "horizon_hint": 6 if step_budget is None else min(6, step_budget),
        "step_budget": step_budget,
        "wall_time_s": wall_time_s,
        "workspace_files": files,
        "subgoals": [
            {
                "id": "renders_ok",
                "desc": "The edited script runs and produces output.png",
                "weight": 1.0,
                "verifiers": [{"type": "execution", "params": {"produces": "output.png"}}],
            },
            {
                "id": "flaw_fixed",
                "desc": "The contradiction with the data is gone",
                "weight": 1.0,
                "verifiers": [{"type": "vlm_judge",
                               "params": {"candidate": "output.png", "rubric": rubric}}],
            },
        ],
    }
    _write_yaml(task_dir / "task.yaml", task)
    return task_dir


def read_task_config(task_dir) -> dict:
    """What a built task actually shipped to the agent.

    Task dirs from before the configuration carry no sidecar; by definition they
    ran with the default set, so that is what they report.
    """
    task_dir = Path(task_dir)
    sidecar = task_dir / "oracle" / "task_config.json"
    if sidecar.exists():
        cfg = json.loads(sidecar.read_text(encoding="utf-8"))
        cfg["modalities"] = list(parse(cfg["modalities"]))
        cfg.setdefault("track", tracks.DEFAULT_TRACK)
        cfg.setdefault("step_budget", tracks.DEFAULT_STEP_BUDGET)
        cfg.setdefault("wall_time_s", tracks.DEFAULT_WALL_TIME_S)
        cfg.setdefault("clean", False)
        return cfg
    meta = json.loads((task_dir / "oracle" / "perturbation.json").read_text(encoding="utf-8"))
    return {
        "task_id": task_dir.name,
        "sample_folder": meta["sample_folder"],
        "category": meta["category"],
        "modalities": list(DEFAULT),
        "modality_code": modality_code(DEFAULT),
        "requested": list(DEFAULT),
        "missing_sources": [],
        "workspace_files": workspace_files(DEFAULT),
        "track": tracks.DEFAULT_TRACK,
        "step_budget": tracks.DEFAULT_STEP_BUDGET,
        "wall_time_s": tracks.DEFAULT_WALL_TIME_S,
        "clean": False,
    }


def _write_yaml(path: Path, task: dict) -> None:
    """Small hand-rolled writer so the file stays readable and dependency-free."""
    def esc(s):
        return '"%s"' % str(s).replace("\\", "\\\\").replace('"', '\\"')

    lines = [
        "id: %s" % task["id"],
        "family: %s" % task["family"],
        "instruction: |",
    ]
    lines += ["  " + ln for ln in task["instruction"].rstrip().splitlines()]
    lines += [
        "horizon_hint: %d" % task["horizon_hint"],
        "step_budget: %s" % ("null" if task["step_budget"] is None else task["step_budget"]),
        "wall_time_s: %s" % ("null" if task["wall_time_s"] is None else task["wall_time_s"]),
        "workspace_files: [%s]" % ", ".join(task["workspace_files"]),
        "subgoals:",
    ]
    for sg in task["subgoals"]:
        lines += [
            "  - id: %s" % sg["id"],
            "    desc: %s" % esc(sg["desc"]),
            "    weight: %s" % sg["weight"],
            "    verifiers:",
        ]
        for v in sg["verifiers"]:
            params = ", ".join("%s: %s" % (k, esc(val) if k == "rubric" else val)
                               for k, val in v["params"].items())
            lines += ["      - type: %s" % v["type"], "        params: {%s}" % params]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pick(limit: int, seed: int) -> list:
    found = []
    for folder in sorted(SAMPLES.glob("*/perturbed_llm")):
        for cat_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
            found.append((folder.parent, cat_dir.name))
    random.Random(seed).shuffle(found)
    # spread across categories instead of taking whatever the shuffle put first
    by_cat, ordered = {}, []
    for inst, cat in found:
        by_cat.setdefault(cat, []).append((inst, cat))
    while len(ordered) < limit and any(by_cat.values()):
        for cat in sorted(by_cat):
            if by_cat[cat] and len(ordered) < limit:
                ordered.append(by_cat[cat].pop())
    return ordered
