"""Run task.yaml directories with durable per-task results and automatic resume.

The input may be a single track folder or a parent containing trackA/trackB/trackC.
Completed failures are results too; only interrupted/infrastructure-failed attempts
are retried. Resume is between tasks, not inside a persistent Python kernel.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SANDBOX = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX / "src"))
from chartsandbox.contract import load_task
from chartsandbox.persistence import atomic_json
from chartsandbox.providers import resolve

TOKEN_KEYS = ("calls", "input_tokens", "output_tokens", "total_tokens",
              "reasoning_tokens", "cache_read_input_tokens")
_ACTIVE = set()
_ACTIVE_LOCK = threading.Lock()
_STOP = threading.Event()


def discover(root):
    direct = [(d, Path(d.name)) for d in sorted(root.iterdir()) if not d.name.startswith(".") and (d / "task.yaml").is_file()]
    if direct:
        return direct
    return [(d, Path(track.name) / d.name)
            for track in sorted(root.iterdir()) if track.is_dir() and not track.name.startswith(".")
            for d in sorted(track.iterdir()) if not d.name.startswith(".") and (d / "task.yaml").is_file()]


def task_digest(task):
    digest = hashlib.sha256()
    for p in sorted(task.rglob("*")):
        if p.is_file() and (p.name in {"task.yaml", "sim_agent.json"} or p.relative_to(task).parts[0] in {"workspace", "oracle"}):
            digest.update(str(p.relative_to(task)).encode())
            digest.update(b"\0")
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def run_config(args):
    return {key: getattr(args, key) for key in (
        "agent", "model", "judge", "protocol", "max_tokens", "max_turns", "effort",
        "judge_effort", "base_url", "episode_timeout", "max_history_images",
    )}


def completed(run_dir, task_id):
    try:
        r = json.loads((run_dir / "result.json").read_text())
        if r.get("task_id") != task_id or "final_score" not in r or "subgoal_results" not in r:
            return None
        if r.get("evaluation_errors"):
            return None
        trace = [json.loads(line) for line in (run_dir / "trajectory.jsonl").read_text().splitlines()]
        if len(trace) != r.get("steps_taken"):
            return None
        if not (run_dir / "workspace").is_dir():
            return None
        return r
    except (OSError, ValueError, TypeError):
        return None


def stop_process(proc):
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        except ProcessLookupError:
            pass


def run_one(task_dir, run_dir, args, record):
    if _STOP.is_set():
        return {"task_id": record["task_id"], "error": "interrupted before launch"}
    # Preserve failed/partial artifacts and API accounting for auditing.
    if run_dir.exists():
        archive = args.out / ".attempts" / record["key"] / str(time.time_ns())
        archive.parent.mkdir(parents=True, exist_ok=True)
        run_dir.rename(archive)
    run_dir.mkdir(parents=True)
    atomic_json(run_dir / "run_manifest.json", record)
    cmd = [sys.executable, "-m", "chartsandbox.cli", "run", str(task_dir),
           "--agent", args.agent, "--model", args.model, "--protocol", args.protocol,
           "--max-tokens", str(args.max_tokens), "--out", str(run_dir),
           "--effort", args.effort, "--judge-effort", args.judge_effort,
           "--max-history-images", str(args.max_history_images)]
    if args.judge:
        cmd += ["--judge", args.judge]
    if args.base_url:
        cmd += ["--base-url", args.base_url]
    if args.max_turns is not None:
        cmd += ["--max-turns", str(args.max_turns)]
    env = {**os.environ, "PYTHONPATH": str(SANDBOX / "src")}
    started = time.time()
    error = None
    with (run_dir / "process.log").open("w") as log:
        with _ACTIVE_LOCK:
            if _STOP.is_set():
                return {"task_id": record["task_id"], "error": "interrupted before launch"}
            proc = subprocess.Popen(cmd, cwd=SANDBOX, env=env, stdout=log, stderr=log,
                                    start_new_session=True)
            _ACTIVE.add(proc)
        try:
            code = proc.wait(timeout=args.episode_timeout or None)
            if code:
                error = f"episode process exited {code}; see process.log"
        except subprocess.TimeoutExpired:
            error = f"episode timeout after {args.episode_timeout}s"
            stop_process(proc)
        finally:
            with _ACTIVE_LOCK:
                _ACTIVE.discard(proc)
    r = completed(run_dir, record["task_id"])
    if r is None or error:
        failure = {"task_id": record["task_id"], "error": error or
                   "incomplete artifacts or evaluator error; inspect result.json/process.log",
                   "wall_time_s": round(time.time() - started, 2)}
        atomic_json(run_dir / "failure.json", failure)
        return failure
    return r


def write_summary(args, rows, total):
    scored = [r for r in rows if "error" not in r]
    tokens = {k: sum((r.get("token_usage", {}).get("total") or {}).get(k, 0) for r in scored)
              for k in TOKEN_KEYS}
    atomic_json(args.out / "summary.json", {
        **run_config(args), "task_root": str(args.task_root), "episodes": total,
        "scored": len(scored), "pending": total - len(rows),
        "failed": [r for r in rows if "error" in r],
        "mean_score": sum(r["final_score"] for r in scored) / len(scored) if scored else None,
        "tokens": tokens,
        "token_scope": "completed episodes only; partial attempts are retained in .attempts/",
        "per_task": [{"key": r["key"], "id": r["task_id"], "score": r["final_score"],
                      "all_passed": r["all_passed"], "stop_reason": r.get("stop_reason"),
                      "finish_origin": r.get("finish_origin"), "wall_time_s": r.get("wall_time_s"),
                      "tokens": r.get("token_usage", {}).get("total", {})} for r in scored],
    })


def run(args):
    jobs = discover(args.task_root)
    if args.limit:
        jobs = jobs[:args.limit]
    if not jobs:
        raise ValueError(f"no task.yaml directories under {args.task_root}")
    config = run_config(args)
    runtime = hashlib.sha256()
    for file in sorted((SANDBOX / "src" / "chartsandbox").rglob("*.py")):
        runtime.update(str(file.relative_to(SANDBOX)).encode())
        runtime.update(file.read_bytes())
    config["runtime_sha256"] = runtime.hexdigest()
    config["environment"] = {"python": sys.version.split()[0]}
    for package in ("matplotlib", "seaborn", "plotly", "kaleido", "numpy", "pandas", "openai", "anthropic"):
        try:
            config["environment"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            config["environment"][package] = None
    plans = []
    for task_dir, relative in jobs:
        lt = load_task(task_dir)
        record = {"task_id": lt.task.id, "key": str(relative),
                  "task_sha256": task_digest(task_dir), "config": config}
        run_dir = args.out / relative
        manifest = run_dir / "run_manifest.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != record:
                raise ValueError(f"task/config changed for {relative}; use a different --out")
        elif run_dir.exists() and any(run_dir.iterdir()):
            raise ValueError(f"untracked existing output {relative}; use a new --out (legacy runs are preserved)")
        plans.append((task_dir, run_dir, record))
    if args.validate_only:
        print(f"validated {len(plans)} tasks; no model calls or episodes launched")
        return 0
    rows, pending = [], []
    for task_dir, run_dir, record in plans:
        r = completed(run_dir, record["task_id"])
        if r is not None:
            rows.append({**r, "key": record["key"]})
        else:
            pending.append((task_dir, run_dir, record))
    print(f"{len(plans)} tasks | {len(rows)} completed, {len(pending)} pending | auto-resume", flush=True)
    write_summary(args, rows, len(plans))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, t, d, args, rec): rec for t, d, rec in pending}
        for future in as_completed(futures):
            record = futures[future]
            try:
                r = future.result()
            except Exception as exc:
                r = {"task_id": record["task_id"], "error": f"{type(exc).__name__}: {exc}"}
            rows.append({**r, "key": record["key"]})
            write_summary(args, rows, len(plans))
            print(f"[{len(rows)}/{len(plans)}] {record['key']} " +
                  (f"FAILED {r['error']}" if "error" in r else f"score={r['final_score']:.3f}"), flush=True)
    return 130 if _STOP.is_set() else 1 if any("error" in r for r in rows) else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("task_root", type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--agent", choices=("llm", "scripted"), default="llm", help="scripted is an offline pipeline test, not model evaluation")
    ap.add_argument("--judge")
    ap.add_argument("--mock-judge", action="store_true", help="explicitly allow mock grading for plumbing tests")
    ap.add_argument("--protocol", choices=("xml", "tools"), default="xml")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-turns", type=int)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--effort", default="default", choices=("default", "none", "minimal", "low", "medium", "high", "xhigh", "max"))
    ap.add_argument("--judge-effort", default="default", choices=("default", "none", "minimal", "low", "medium", "high", "xhigh", "max"))
    ap.add_argument("--base-url")
    ap.add_argument("--max-history-images", type=int, default=0)
    ap.add_argument("--episode-timeout", type=int, default=1800, help="0 disables the outer process timeout")
    ap.add_argument("--limit", type=int, default=0, help="first N tasks; remove to expand to the full folder")
    ap.add_argument("--resume", action="store_true", help="compatibility flag; resume is always enabled")
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args(argv)
    if args.workers < 1 or args.max_tokens < 1 or args.limit < 0 or args.episode_timeout < 0 or args.max_history_images < 0 or (args.max_turns is not None and args.max_turns < 1):
        ap.error("invalid nonpositive budget/worker count or negative limit")
    if not args.judge and not args.mock_judge and not args.validate_only:
        ap.error("specify --judge MODEL for evaluation, or --mock-judge for an explicit plumbing test")
    resolve(args.model)
    if args.judge:
        resolve(args.judge)
    args.task_root = args.task_root.resolve()
    args.out = args.out.resolve()
    if args.out == args.task_root or args.out.is_relative_to(args.task_root):
        ap.error("--out must be outside the input task folder")
    args.out.mkdir(parents=True, exist_ok=True)
    _STOP.clear()
    previous = {}
    def interrupt(signum, frame):
        _STOP.set()
        with _ACTIVE_LOCK:
            active = list(_ACTIVE)
        for proc in active:
            stop_process(proc)
    with (args.out / ".suite.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            ap.error("another suite is already using this output directory")
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, interrupt)
        try:
            return run(args)
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
