"""Command-line entry point.

    chartsandbox validate <task_dir>
    chartsandbox run <task_dir> [--sim <sim_agent.json>] [--out <dir>]

`run` uses the scripted (simulated) agent — for smoke-testing the sandbox. When a
real LLM agent exists, add an `--agent llm` branch here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent import ScriptedAgent
from .contract import load_task
from .metrics import summarize
from .runner import run_episode


def _cmd_validate(args) -> int:
    lt = load_task(args.task_dir)
    print(f"OK: task {lt.task.id!r} ({lt.task.family}), "
          f"{len(lt.task.subgoals)} subgoal(s), budget={lt.task.step_budget}")
    for sg in lt.task.subgoals:
        vt = ", ".join(v.type for v in sg.verifiers)
        print(f"  - {sg.id} (w={sg.weight}): [{vt}]")
    return 0


def _cmd_run(args) -> int:
    task_dir = Path(args.task_dir)
    sim_path = Path(args.sim) if args.sim else (task_dir / "sim_agent.json")
    if not sim_path.exists():
        print(f"error: no scripted-agent file at {sim_path}. "
              f"Pass --sim, or add sim_agent.json to the task.", file=sys.stderr)
        return 2
    script = json.loads(sim_path.read_text())
    agent = ScriptedAgent(script)
    result = run_episode(task_dir, agent, out_dir=args.out)
    print(summarize(result))
    print(f"\nwrote: {result['_run_dir']}/result.json")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="chartsandbox")
    sub = p.add_subparsers(dest="cmd", required=True)

    pv = sub.add_parser("validate", help="validate a task instance against the contract")
    pv.add_argument("task_dir")
    pv.set_defaults(fn=_cmd_validate)

    pr = sub.add_parser("run", help="run the scripted agent through a task")
    pr.add_argument("task_dir")
    pr.add_argument("--sim", help="path to scripted-agent JSON (default: <task>/sim_agent.json)")
    pr.add_argument("--out", help="output run dir (default: <task>/runs/run_<ts>)")
    pr.set_defaults(fn=_cmd_run)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
