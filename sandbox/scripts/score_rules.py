"""Score finished episodes with the release's rule-based scorer and merge the result.

The in-loop judge answers two questions — was the flaw fixed, is the chart legible —
and is structurally blind to a third: whether the agent damaged something that was
already right. On a ten-episode pilot that blindness was not academic. Four episodes
the judge scored 1.00 came back with preservation as low as 0.30 and data fidelity as
low as 0.00: the model had repaired the named defect and quietly rewritten other parts
of the chart. Nothing in the episode loop can see that, because seeing it means
diffing every captured figure property against the reference.

So this does not reimplement the check, it calls the one the release ships
(`scoring/score_submission.py`), which re-executes the submitted script in its own
environment and compares figure-object property paths. Reimplementing would produce a
second number that disagrees with the published one, which is worse than having none.

Merged into each `result.json` under `rule_based`, leaving every existing field alone.

    python scripts/score_rules.py <run_dir> --release <release> --tasks <track-tasks>
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SANDBOX = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX / "src"))
from chartsandbox.persistence import atomic_json
DEFAULT_PY = Path(sys.executable).absolute()

DIMENSIONS = ("executability", "data_fidelity", "recovery", "preservation")


def _n(v) -> str:
    """A metric cell: the number, or `--` where the metric is undefined."""
    return f"{v:>6.2f}" if isinstance(v, (int, float)) else f"{'--':>6}"


def strict(metrics: dict, key: str):
    """The strict variant of a metric that reports strict/lenient, else the value."""
    v = metrics.get(key)
    return v.get("strict") if isinstance(v, dict) else v


def score_one(episode: Path, tasks: Path, release: Path, python: Path,
              timeout: int) -> dict | None:
    submission = episode / "workspace" / "chart.py"
    config = tasks / episode.name / "oracle" / "task_config.json"
    if not submission.is_file() or not config.is_file():
        return {"error": "missing final chart.py or task_config.json"}
    cfg = json.loads(config.read_text())
    # The scorer shells out again to run each chart, and picks that interpreter in
    # `harness.py` as: CRB_PYTHON, else a .venv beside the scoring package, else
    # sys.executable. The release ships no such venv, and the fallback landed on an
    # Anaconda base carrying seaborn 0.12 / matplotlib 3.7 — old enough that
    # seaborn's `boxplot(legend=...)` raises TypeError and plotly has no kaleido at
    # all. Both surfaced as "reference capture failed", which reads like broken
    # reference data rather than the wrong interpreter, and cost 86 of 300 episodes
    # per track. CRB_PYTHON is the documented override; set it explicitly.
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")}
    env["CRB_PYTHON"] = str(python)
    env["PATH"] = f"{python.parent}{os.pathsep}{env.get('PATH', '')}"
    env["VIRTUAL_ENV"] = str(python.parent.parent)
    proc = subprocess.run(
        [str(python), "scoring/score_submission.py",
         "--instance", f"instances/{episode.name}",
         "--submission", str(submission),
         *(["--clean"] if cfg.get("clean") else ["--category", cfg["combination"]])],
        cwd=release, env=env, capture_output=True, text=True, timeout=timeout)
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"error": (proc.stderr or proc.stdout or "no output")[-300:]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--python", type=Path, default=DEFAULT_PY,
                    help="interpreter with the scoring dependencies installed")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--limit", type=int, default=0, help="0 = every episode")
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent scorer processes (default: 1). Raise with care: "
                         "the scorer's capture step is not safe to run concurrently. "
                         "At 6 workers, 86 of 300 episodes reported 'reference capture "
                         "failed' — every one of which passed on a sequential re-test, "
                         "and 75 of them were plotly, whose static renderer holds "
                         "process-external state. A wrong score is worse than a slow one.")
    args = ap.parse_args()

    # The scorer runs with cwd set to the release checkout, so every path handed to
    # it has to be absolute. A relative --run-dir would otherwise resolve against
    # the release and fail on every episode with a FileNotFoundError that looks
    # like a broken submission rather than a broken invocation.
    args.run_dir = args.run_dir.resolve()
    args.tasks = args.tasks.resolve()
    args.release = args.release.resolve()
    # absolute(), never resolve(): a venv's bin/python is a symlink to the base
    # interpreter, and following it hands back that base interpreter with none of
    # the venv on its path. Invoking the symlink is what makes sys.prefix point at
    # the venv. Resolving it here is what silently ran every capture under Anaconda.
    args.python = args.python.absolute()

    episodes = [e for e in sorted(args.run_dir.iterdir())
                if (e / "result.json").is_file()]
    if args.limit:
        episodes = episodes[: args.limit]
    rows = []
    print(f"{'task':38} {'local':>6} | {'exec':>4} {'recov':>6} {'presv':>6} {'datafi':>6}")
    print("-" * 82)

    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(score_one, e, args.tasks, args.release, args.python,
                            args.timeout): e for e in episodes}
        # Printed as each finishes rather than after the pool drains: a suite of
        # nine hundred should show progress, not silence followed by a wall of text.
        for future in as_completed(jobs):
            episode = jobs[future]
            try:
                metrics = future.result()
            except Exception as exc:
                metrics = {"error": f"{type(exc).__name__}: {exc}"}
            if metrics is None:
                continue
            result_path = episode / "result.json"
            result = json.loads(result_path.read_text())
            if "error" in metrics:
                failures += 1
                if failures <= 3:
                    print(f"{episode.name[:37]:38} scorer failed: {metrics['error'][:60]}",
                          flush=True)
                result["rule_based"] = metrics
                atomic_json(result_path, result)
                continue

            vector = {d: strict(metrics, d) for d in DIMENSIONS}
            vector["recovery_binary"] = strict(metrics, "recovery_binary")
            vector["preservation_binary"] = strict(metrics, "preservation_binary")
            result["rule_based"] = {**vector, "full": metrics}
            atomic_json(result_path, result)

            judge = result.get("final_score")
            rows.append({"id": episode.name, "judge": judge, **vector})
            # A Track C clean twin has no recovery to speak of — there was nothing
            # to recover — and the scorer reports it as undefined. Printing it as
            # 0.00 would read as a failed repair on a chart that needed none.
            print(f"{episode.name[:37]:38} {_n(judge)} | {vector['executability']:>4} "
                  f"{_n(vector['recovery'])} {_n(vector['preservation'])} "
                  f"{_n(vector['data_fidelity'])}", flush=True)
    if failures:
        print(f"  ({failures} episode(s) the scorer could not evaluate)")

    if not rows:
        print("nothing scored")
        return 1

    print("-" * 82)
    for d in ("recovery", "preservation", "data_fidelity"):
        vals = [r[d] for r in rows if r[d] is not None]
        if vals:
            print(f"  mean {d:14} {statistics.mean(vals):.2f}")
    def worst(r: dict) -> float | None:
        vals = [v for v in (r["recovery"], r["preservation"], r["data_fidelity"])
                if isinstance(v, (int, float))]
        return min(vals) if vals else None

    blind = [r for r in rows
             if (r["judge"] or 0) >= 0.95
             and (w := worst(r)) is not None and w < 0.95]
    print(f"\n  local aggregate scored >=0.95 while the rule-based scorer found damage: "
          f"{len(blind)}/{len(rows)}")
    for r in blind[:10]:
        print(f"    {r['id'][:52]:54} recov={_n(r['recovery']).strip()} "
              f"presv={_n(r['preservation']).strip()} datafi={_n(r['data_fidelity']).strip()}")
    if len(blind) > 10:
        print(f"    ... and {len(blind) - 10} more")
    atomic_json(args.run_dir / "rule_based_summary.json", rows)
    print(f"\n  merged into each result.json, summary at "
          f"{args.run_dir / 'rule_based_summary.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
