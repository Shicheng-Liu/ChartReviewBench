"""Episode runner — drive one agent through one task and score it.

    load task -> fresh workspace (copy of task/workspace) -> agent loop
              -> per-step cheap subgoal checks (progress curve)
              -> final verification -> result.json + trajectory.jsonl
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from .agent import Agent, ToolCall
from .contract import LoadedTask, load_task
from .metrics import progress_curve
from .tools import TOOL_SCHEMAS, Sandbox
from .verifiers import CHEAP, VerifyContext, run_verifier

_FILE_TOOLS = {"write_file", "read_file", "list_files", "view_image"}


def _initial_observation(lt: LoadedTask, sandbox: Sandbox) -> dict:
    task = lt.task
    tool_list = "\n".join(f"  - {t['name']}: {t['description']}" for t in TOOL_SCHEMAS)
    files = sandbox.list_files(".").get("files", [])
    text = (
        f"# Task ({task.family})\n{task.instruction}\n\n"
        f"## Workspace files\n" + "\n".join(f"  - {f['name']}" for f in files) + "\n\n"
        f"## Tools\n{tool_list}\n\n"
        f"Call finish when done. Step budget: {task.step_budget}."
    )
    return {"type": "initial", "text": text}


def _dispatch(sandbox: Sandbox, tc: ToolCall) -> dict:
    if tc.tool == "execute_python":
        return sandbox.execute_python(tc.args.get("code", ""))
    if tc.tool in _FILE_TOOLS:
        return getattr(sandbox, tc.tool)(**tc.args)
    if tc.tool == "finish":
        return {"ok": True, "finished": True, "message": tc.args.get("message", "")}
    return {"ok": False, "error": f"unknown tool {tc.tool!r}"}


def _summarize_obs(tc: ToolCall, obs: dict) -> dict:
    """Compact form of an observation for the trajectory log (drop image bytes)."""
    s = {k: v for k, v in obs.items() if k != "base64"}
    if tc.tool == "view_image" and obs.get("ok"):
        s["image"] = f"{obs.get('width')}x{obs.get('height')} {obs.get('mime')}"
    if tc.tool == "read_file" and "content" in s:
        s["content"] = s["content"][:200] + ("..." if len(s["content"]) > 200 else "")
    return s


def _eval_subgoal(sg, ctx_factory, cheap_only: bool):
    results = []
    for v in sg.verifiers:
        if cheap_only and v.type not in CHEAP:
            continue
        results.append(run_verifier(v.type, ctx_factory(v.params)))
    if not results:
        return None
    passed = all(r.passed for r in results)
    score = sum(r.score for r in results) / len(results)
    return passed, score, results


def run_episode(task_dir: str | Path, agent: Agent, out_dir: str | Path | None = None) -> dict:
    lt = load_task(task_dir)
    task = lt.task

    # fresh run workspace (never mutate the task template; oracle stays out of it)
    run_dir = Path(out_dir) if out_dir else (lt.task_dir / "runs" / time.strftime("run_%Y%m%d_%H%M%S"))
    run_ws = run_dir / "workspace"
    if run_ws.exists():
        shutil.rmtree(run_ws)
    run_ws.parent.mkdir(parents=True, exist_ok=True)
    if lt.workspace_dir.exists():
        shutil.copytree(lt.workspace_dir, run_ws)
    else:
        run_ws.mkdir(parents=True)

    sandbox = Sandbox(run_ws, task.step_timeout_s)
    agent.reset(task, sandbox)

    def ctx_factory(params):
        return VerifyContext(workspace=run_ws, oracle=lt.oracle_dir,
                             params=params, last_exec=last_exec)

    trajectory: list[dict] = []
    tool_counts: dict[str, int] = {}
    first_pass_step: dict[str, int | None] = {sg.id: None for sg in task.subgoals}
    last_exec: dict | None = None
    finished = False
    t0 = time.time()

    for step in range(task.step_budget):
        if time.time() - t0 > task.wall_time_s:
            trajectory.append({"step": step, "tool": "<wall_time_exceeded>"})
            break
        tc = agent.act(trajectory[-1]["observation"] if trajectory else _initial_observation(lt, sandbox))
        obs = _dispatch(sandbox, tc)
        tool_counts[tc.tool] = tool_counts.get(tc.tool, 0) + 1
        if tc.tool == "execute_python":
            last_exec = obs
        trajectory.append({"step": step, "tool": tc.tool, "args": tc.args,
                           "observation": _summarize_obs(tc, obs)})

        # cheap incremental subgoal check -> progress curve
        for sg in task.subgoals:
            if first_pass_step[sg.id] is None:
                r = _eval_subgoal(sg, ctx_factory, cheap_only=True)
                if r and r[0]:
                    first_pass_step[sg.id] = step

        if tc.tool == "finish":
            finished = True
            break

    # --- final verification (all verifiers, including expensive judge) ------
    subgoal_results = []
    weighted_num = weighted_den = 0.0
    for sg in task.subgoals:
        r = _eval_subgoal(sg, ctx_factory, cheap_only=False)
        passed, score, vres = (r if r else (True, 1.0, []))
        weighted_num += score * sg.weight
        weighted_den += sg.weight
        subgoal_results.append({
            "id": sg.id, "desc": sg.desc, "weight": sg.weight,
            "passed": passed, "score": round(score, 4),
            "first_pass_step": first_pass_step[sg.id],
            "verifiers": [{"name": v.name, "passed": v.passed,
                           "score": round(v.score, 4), "detail": v.detail} for v in vres],
        })

    sandbox.close()
    steps_taken = len(trajectory)
    final_score = round(weighted_num / weighted_den, 4) if weighted_den else 0.0

    result = {
        "task_id": task.id,
        "family": task.family,
        "finished": finished,
        "final_score": final_score,
        "all_passed": all(s["passed"] for s in subgoal_results),
        "subgoals_total": len(task.subgoals),
        "subgoals_passed": sum(1 for s in subgoal_results if s["passed"]),
        "subgoal_results": subgoal_results,
        "steps_taken": steps_taken,
        "step_budget": task.step_budget,
        "horizon_hint": task.horizon_hint,
        "wall_time_s": round(time.time() - t0, 2),
        "tool_counts": tool_counts,
        "progress_curve": progress_curve(first_pass_step, steps_taken, len(task.subgoals)),
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    with (run_dir / "trajectory.jsonl").open("w") as f:
        for row in trajectory:
            f.write(json.dumps(row) + "\n")
    result["_run_dir"] = str(run_dir)
    return result
