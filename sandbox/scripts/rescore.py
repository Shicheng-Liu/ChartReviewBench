"""Re-run the verifiers over a finished run, without re-running the agent.

Verification happens once, at the end of an episode, against files that are still
sitting in the run's workspace. So a change to a verifier, a rubric or the judge does
not require paying for the trajectory again — which matters, because the alternative
is that nobody ever fixes a judge bug on a run that cost real money to produce.

The original `result.json` is left alone and the new numbers go to `rescored.json`,
so the two can be compared and the change attributed.

    python scripts/rescore.py <run_dir> --tasks <task_root> [--judge MODEL]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SANDBOX = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX / "src"))

from chartsandbox.contract import load_task  # noqa: E402
from chartsandbox.metrics import progress_curve  # noqa: E402
from chartsandbox.verifiers import VerifyContext, run_verifier  # noqa: E402


def rescore(run_dir: Path, task_dir: Path) -> dict:
    lt = load_task(task_dir)
    ws = run_dir / "workspace"
    before = json.loads((run_dir / "result.json").read_text())

    # `last_exec` is not recoverable from a finished run, so a verifier that depends
    # on it is skipped rather than silently scored against an empty execution.
    last_exec = None
    for line in (run_dir / "trajectory.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("tool") == "execute_python":
            last_exec = row.get("observation")

    results, num, den = [], 0.0, 0.0
    for sg in lt.task.subgoals:
        vres = [run_verifier(v.type, VerifyContext(workspace=ws, oracle=lt.oracle_dir,
                                                   params=v.params, last_exec=last_exec))
                for v in sg.verifiers]
        passed = all(v.passed for v in vres) if vres else True
        score = (sum(v.score for v in vres) / len(vres)) if vres else 1.0
        num += score * sg.weight
        den += sg.weight
        results.append({
            "id": sg.id, "desc": sg.desc, "weight": sg.weight,
            "passed": passed, "score": round(score, 4),
            "verifiers": [{"name": v.name, "passed": v.passed, "score": round(v.score, 4),
                           "detail": v.detail} for v in vres],
        })

    out = {
        **{k: before[k] for k in ("task_id", "family", "finished", "stop_reason",
                                  "finish_origin", "steps_taken", "step_budget",
                                  "tool_counts") if k in before},
        "final_score": round(num / den, 4) if den else 0.0,
        "all_passed": all(r["passed"] for r in results),
        "subgoals_total": len(results),
        "subgoals_passed": sum(1 for r in results if r["passed"]),
        "subgoal_results": results,
        "rescored_from": {"final_score": before.get("final_score"),
                          "subgoals_passed": before.get("subgoals_passed")},
        "agent_stats": before.get("agent_stats"),
    }
    (run_dir / "rescored.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="a run root holding one dir per episode")
    ap.add_argument("--tasks", type=Path, required=True, help="the task instances used")
    ap.add_argument("--judge", metavar="MODEL", help="install a real judge before rescoring")
    args = ap.parse_args()

    if args.judge:
        from chartsandbox.judge import install_judge

        install_judge(args.judge)

    episodes = [d for d in sorted(args.run_dir.iterdir())
                if (d / "result.json").exists()]
    print(f"{'task':44} {'was':>6} -> {'now':>6}  passed")
    for d in episodes:
        task_dir = args.tasks / d.name
        if not (task_dir / "task.yaml").exists():
            print(f"  {d.name[:42]:44} no task instance at {task_dir}")
            continue
        r = rescore(d, task_dir)
        was = r["rescored_from"]["final_score"]
        print(f"{d.name[:43]:44} {was:>6.2f} -> {r['final_score']:>6.2f}  "
              f"{r['subgoals_passed']}/{r['subgoals_total']}"
              f" (was {r['rescored_from']['subgoals_passed']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
