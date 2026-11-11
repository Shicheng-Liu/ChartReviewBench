"""Re-score finished runs against the current metric code, without re-running them.

An episode is expensive (it calls a model); scoring it is not (it re-executes
three scripts locally). So when the metric or the scoring context changes, the
right move is to recompute the numbers from the workspaces already on disk
rather than to buy the episodes again.

The trajectory, token cost and wall time still describe the original run.
Legacy wall-clock aborts are reclassified using their recorded termination
marker. Original episode records are backed up before replacement.

Usage:
    .venv/bin/python -m src.eval.rescore data/v0/runs/panel_haiku_icts
    .venv/bin/python -m src.eval.rescore data/v0/runs/*            # all of them
"""
import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from . import run_benchmark, sandbox_run
from .provenance import scorer_provenance  # re-exported for existing callers


def _configure_samples(samples):
    if samples is not None:
        sandbox_run.to_sandbox.SAMPLES = Path(samples)


def _one(args):
    ep_path, tasks_root, timeout, *expected_scorer = args
    target = Path(ep_path)
    try:
        original = target.read_text(encoding="utf-8")
        ep = json.loads(original)
        run_dir = target.parents[1] / "runs" / (ep.get("run_slug") or "__missing__")
        task_dir = Path(tasks_root) / ep["task_id"]
        if not task_dir.exists() or (not run_dir.exists() and ep.get("status") == "ok"):
            return ep_path, None, "task or run directory missing"
        scorer = scorer_provenance()
        if expected_scorer and scorer != expected_scorer[0]:
            return ep_path, None, "scorer sources changed during rescore"
        ours = (sandbox_run.score_repair(task_dir, run_dir, timeout) if run_dir.exists()
                else sandbox_run.score_no_submission(task_dir, timeout, ep.get("error", "")))
        result_path = run_dir / "result.json"
        result = {**(ep.get("sandbox") or {}), **(
            json.loads(result_path.read_text()) if result_path.exists() else {})}
        # Newer bridge records carry termination causes absent from upstream's
        # result.json. Historical trajectory markers still take precedence.
        if ep.get("stop_reason") and ep["stop_reason"] != "unknown":
            result["stop_reason"] = ep["stop_reason"]
        reason = sandbox_run.episode_stop_reason(result, run_dir)
        if scorer_provenance() != scorer:
            return ep_path, None, "scorer sources changed during rescore"
    except Exception as exc:
        return ep_path, None, "%s: %s" % (type(exc).__name__, exc)
    before = ep.get("executability")
    ep["executability"] = ours["executability"]
    ep["metrics"] = {k: ours[k] for k in (
        "data_fidelity", "data_fidelity_binary", "recovery", "recovery_binary",
        "preservation", "preservation_binary")}
    ep["footprint"] = ours["diagnostics"].get("footprint", [])
    ep["stop_reason"] = reason
    ep["sandbox"] = {**(ep.get("sandbox") or {}), "stop_reason": reason}
    ep["scored_against_instance"] = True
    if reason in sandbox_run.INTERRUPTED_STOP_REASONS:
        ep["status"] = "error"
        ep["error"] = "harness interruption: " + reason
    ep["rescored"] = True
    ep["scorer"] = scorer
    ep["rescored_at"] = datetime.now(timezone.utc).isoformat()
    # Content-addressed backups preserve every prior scoring version, including
    # the original status of a historically misclassified wall-clock abort.
    backup = target.parents[1] / "rescore_originals" / target.name
    backup.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(original.encode()).hexdigest()
    prior = backup / (digest + ".json")
    if not prior.exists() or prior.read_text(encoding="utf-8") != original:
        _atomic_text(prior, original)
    _atomic_text(target, json.dumps(ep, indent=1))
    return ep_path, (before, ours["executability"]), None


def rescore(out: Path, tasks_root: Path, workers: int = 8, timeout: int = 120,
            samples: Path | None = None) -> dict:
    with run_benchmark.RunLock(out):
        return _rescore(out, tasks_root, workers, timeout, samples)


def _rescore(out: Path, tasks_root: Path, workers: int, timeout: int, samples=None) -> dict:
    eps = sorted((out / "episodes").glob("*.json"))
    if not eps:
        return {"out": str(out), "n": 0}
    ignored, failed = 0, []
    scorer = scorer_provenance()
    config = out / "run.json"
    meta = json.loads(config.read_text(encoding="utf-8")) if config.exists() else {}
    samples = Path(samples or meta.get("samples") or sandbox_run.to_sandbox.SAMPLES).resolve()
    sources = {"tasks": str(tasks_root.resolve()), "samples": str(samples), "timeout": timeout}
    state = {"out": str(out), "complete": False, "scorer": scorer, "sources": sources,
             "started_at": datetime.now(timezone.utc).isoformat()}
    _write_state(out, state)
    if config.exists():
        try:
            source = meta.get("dataset_path") or meta.get("dataset")
            if source:
                dataset_path = run_benchmark.dataset_mod.dataset_path(source, tasks_root.parent)
                dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
                if meta.get("dataset_sha256") and meta["dataset_sha256"] != dataset_hash:
                    raise ValueError("dataset hash differs from run.json")
                sources.update(dataset=str(dataset_path.resolve()), dataset_sha256=dataset_hash)
            expected = run_benchmark.expected_task_ids(meta, tasks_root.parent)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            failed.append(("run.json", "cannot reconstruct pinned selection: %s" % exc))
        else:
            selected = []
            for path in eps:
                try:
                    task_id = json.loads(path.read_text(encoding="utf-8"))["task_id"]
                except (OSError, KeyError, TypeError, ValueError):
                    selected.append(path)  # the worker reports malformed episode files
                    continue
                if task_id in expected:
                    selected.append(path)
            ignored, eps = len(eps) - len(selected), selected
    jobs = [(str(p), str(tasks_root), timeout, scorer) for p in eps]
    changed = 0
    if not failed:
        with ProcessPoolExecutor(workers, initializer=_configure_samples, initargs=(samples,)) as ex:
            for path, delta, err in ex.map(_one, jobs):
                if err:
                    failed.append((Path(path).stem, err))
                elif delta and delta[0] != delta[1]:
                    changed += 1
    result = {"out": str(out), "n": len(eps), "ignored_stale": ignored,
            "executability_changed": changed,
            "failed": failed, "scorer": scorer, "sources": sources}
    if not failed:
        try:
            episodes = [json.loads(path.read_text(encoding="utf-8")) for path in eps]
            if not episodes:
                raise ValueError("no episodes belong to the pinned selection")
            if scorer_provenance() != scorer or any(ep.get("scorer") != scorer for ep in episodes):
                raise ValueError("scorer provenance differs across episodes or source files")
            report = run_benchmark.aggregate(episodes, meta, meta.get("skipped"))
            report["scorer"] = scorer
            report["rescore"] = {**state, "complete": True,
                                 "finished_at": datetime.now(timezone.utc).isoformat()}
            run_benchmark.write_report(out, report)
        except Exception as exc:
            failed.append(("report", "%s: %s" % (type(exc).__name__, exc)))
    # A partial rescore is explicitly invalid until rerun successfully. The
    # marker travels with the episode directory, even if an old report exists.
    result["complete"] = not failed
    _write_state(out, {**state, **result, "finished_at": datetime.now(timezone.utc).isoformat()})
    return result


def _write_state(out: Path, state: dict) -> None:
    _atomic_text(out / "rescore.json", json.dumps(state, indent=1))


def _atomic_text(target: Path, text: str) -> None:
    tmp = target.with_name("." + target.name + ".rescore.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(target)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", help="run directories to re-score")
    ap.add_argument("--tasks", default="data/v0/sandbox_tasks")
    ap.add_argument("--samples", type=Path, help="source sample root, useful from an isolated checkout")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args(argv)

    failed_any = False
    for r in args.runs:
        out = Path(r)
        if not (out / "episodes").is_dir():
            continue
        res = rescore(out, Path(args.tasks), args.workers, args.timeout, args.samples)
        print("%-40s %3d Episoden · executability geändert bei %d%s" % (
            out.name, res["n"], res.get("executability_changed", 0),
            " · %d Fehler" % len(res["failed"]) if res.get("failed") else ""))
        if res.get("ignored_stale"):
            print("      %d veraltete Episodendateien außerhalb der gepinnten Auswahl ignoriert"
                  % res["ignored_stale"])
        for name, err in (res.get("failed") or [])[:3]:
            print("      %s: %s" % (name, err[:120]))
        if res.get("failed"):
            failed_any = True
            print("      Report nicht neu gebaut: ein gemischter alter/neuer Score wäre "
                  "keine reproduzierbare Auswertung", file=sys.stderr)
            continue
    return int(failed_any)


if __name__ == "__main__":
    raise SystemExit(main())
