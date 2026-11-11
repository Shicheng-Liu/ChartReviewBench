"""Deterministic AxisTruncation for the v1.2 recipe (reviewer, 28.08.).

The old LLM-written axis flaw only moved a baseline away from zero, which is
not reliably wrong.  The V2 transform plans a single-axis limit that actually
clips at least one truth-backed plotted point and injects that one statement
before the image is written.  This script writes such an edit for every pool
sample whose ground truth renders and yields a plan, as a record in the same
cache the LLM generator uses (``data/v0/llm_perturbations.json``), so
``build_llm_instances.py`` materializes it like any other flaw.  Records carry
``target`` (limits and clipped points) for the binding axis postcondition and
``model = axis-truncation-v2``; existing LLM records for this category are left
untouched unless ``--replace`` is given.

Usage: .venv/bin/python src/axis_inject.py [--limit N] [--workers 6] [--replace]
"""
import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval.capture import capture  # noqa: E402
from llm_perturb import SAMPLES, cache_key, load_cache, save_cache  # noqa: E402
from v2.axis_truncation import inject_axis_truncation, plan_axis_truncation  # noqa: E402

CATEGORY = "axis-truncation"
GENERATOR = "axis-truncation-v2"
# The clip is injected into matplotlib/seaborn scripts only; plotly's capture
# exposes no computed auto-range to plan against (job_preconditions says the
# same). Skipping them here saves one GT capture per plotly source.
UNSUPPORTED_LIBRARIES = {"plotly", "plotnine"}


def _backend(spec: dict, library: str) -> str | None:
    backend = str(spec.get("backend") or library).casefold()
    if "plotly" in backend:
        return "plotly"
    if backend in {"matplotlib", "seaborn", "mpl"}:
        return "matplotlib"
    return None


def inject_one(folder: Path, meta: dict, timeout: int = 90) -> tuple[dict | None, str]:
    code = (folder / "gt/code.py").read_text(encoding="utf-8")
    extra = {"table.csv": (folder / "gt/table.csv").read_text(encoding="utf-8")}
    summary_path = folder / "gt/summary.txt"
    if summary_path.exists():
        extra["summary.txt"] = summary_path.read_text(encoding="utf-8")
    return inject_code(code, extra, meta.get("library", ""), timeout)


def inject_code(code: str, extra: dict, library: str,
                timeout: int = 90) -> tuple[dict | None, str]:
    """Plan and inject the clip on ``code`` as it is (the multi-flaw chain passes
    a script that already carries earlier flaws; the clip is planned on its
    rendered state so the clipped point is truth-backed in that context)."""
    gt = capture(code, extra, timeout)
    if gt["status"] != "ok" or not gt.get("spec"):
        return None, "gt_capture_%s" % gt["status"]
    plan = plan_axis_truncation(gt["spec"])
    if plan is None:
        return None, "no_clippable_axis"
    backend = _backend(gt["spec"], library)
    if backend is None:
        return None, "unsupported_backend"
    try:
        perturbed = inject_axis_truncation(code, backend, plan)
    except Exception as exc:  # noqa: BLE001 - one unsupported script, not a dead run
        return None, "inject_%s" % type(exc).__name__[:24]
    return {
        "category": CATEGORY,
        "perturbed_code": perturbed,
        "description": "Clip one truth-backed plotted endpoint by limiting the value "
                       "axis so the point is no longer visible.",
        "evidence": {"clipped_truth_points": [dict(p) for p in plan.clipped],
                     "table_nums": [p["value"] for p in plan.clipped
                                    if isinstance(p.get("value"), (int, float))]},
        "target": plan.target,
        "model": GENERATOR,
        "provider": "deterministic",
        "edit_ok": True,
        "edit_note": "deterministic single-statement axis clip",
        "edit_metrics": {"added": 1, "changed": 0, "removed": 0},
    }, "ok"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--replace", action="store_true",
                    help="overwrite existing LLM axis records for the same source")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    cache = load_cache()
    todo = []
    skipped = Counter()
    for meta_path in sorted(SAMPLES.glob("*/meta.json")):
        folder = meta_path.parent
        exec_path = folder / "gt/exec.json"
        if not exec_path.exists() or json.loads(exec_path.read_text()).get("status") != "ok":
            continue
        meta = json.loads(meta_path.read_text())
        if str(meta.get("library", "")).casefold() in UNSUPPORTED_LIBRARIES:
            skipped[meta.get("library")] += 1
            continue
        code = (folder / "gt/code.py").read_text(encoding="utf-8")
        key = cache_key(meta["id"], CATEGORY, code)
        existing = cache.get(key)
        if existing is not None and (existing.get("model") == GENERATOR or not args.replace):
            continue
        todo.append((folder, meta, key))
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(todo)} sources to plan (cache holds {len(cache)} records; "
          f"skipped unsupported libraries: {dict(skipped)})")
    if args.dry_run:
        return 0
    stats = Counter()
    fresh = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for (folder, meta, key), (rec, status) in zip(
            todo, ex.map(lambda item: inject_one(item[0], item[1], args.timeout), todo)
        ):
            stats[status] += 1
            if rec is not None:
                fresh[key] = rec
            if sum(stats.values()) % 50 == 0:
                save_cache(fresh)
                fresh = {}
                print(f"  {sum(stats.values())}/{len(todo)} · {dict(stats)}", flush=True)
    save_cache(fresh)
    print(f"done: {dict(stats)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
