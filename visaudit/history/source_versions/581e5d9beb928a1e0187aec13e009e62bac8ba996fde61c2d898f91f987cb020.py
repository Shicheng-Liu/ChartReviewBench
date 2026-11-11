"""Local nullable-budget adapter; the external chartsandbox package stays unchanged."""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import yaml
from pydantic import Field

from chartsandbox import runner as upstream
from chartsandbox.contract import LoadedTask, Task

from . import cancellation


def write_snapshot(path, data):
    path = Path(path)
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(data, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class BudgetTask(Task):
    step_budget: int | None = Field(default=20, gt=0)
    wall_time_s: int | None = Field(default=900, gt=0)


def load_task(task_dir: str | Path) -> LoadedTask:
    path = Path(task_dir).resolve()
    task = BudgetTask(**yaml.safe_load((path / "task.yaml").read_text()))
    workspace = path / "workspace"
    for name in task.workspace_files:
        if not (workspace / name).is_file():
            raise FileNotFoundError(f"missing declared workspace file: {name}")
    return LoadedTask(task=task, task_dir=path, workspace_dir=workspace,
                      oracle_dir=path / "oracle")


def initial_observation(loaded: LoadedTask, sandbox) -> dict:
    observation = upstream._initial_observation(loaded, sandbox)
    if loaded.task.step_budget is None:
        observation["text"] = observation["text"].replace(
            "Step budget: None.", "There is no tool-step limit.")
    if loaded.task.wall_time_s is None:
        observation["text"] += " There is no episode wall-time limit."
    return observation


def run_episode(task_dir, agent, out_dir=None, extra_stats=None) -> dict:
    loaded = load_task(task_dir)
    task = loaded.task
    if task.step_budget is not None and task.wall_time_s is not None:
        return upstream.run_episode(task_dir, agent, out_dir=out_dir, extra_stats=extra_stats)

    run_dir = Path(out_dir) if out_dir else loaded.task_dir / "runs" / time.strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    if any((run_dir / name).exists() for name in ("trajectory.jsonl", "checkpoint.json", "result.json")):
        raise FileExistsError("archive the previous attempt before starting a new episode")
    (run_dir / "trajectory.jsonl").touch()
    workspace = run_dir / "workspace"
    if workspace.exists():
        shutil.rmtree(workspace)
    if loaded.workspace_dir.exists():
        shutil.copytree(loaded.workspace_dir, workspace)
    else:
        workspace.mkdir(parents=True)
    sandbox = upstream.Sandbox(workspace, task.step_timeout_s)
    trajectory, tool_counts = [], {}
    first_pass = {sg.id: None for sg in task.subgoals}
    last_exec = None
    finished, finish_origin = False, None
    started = time.time()
    status, stop_reason, phase, pending_call = "running", None, "initializing", None
    subgoal_results, weighted, weights = [], 0.0, 0.0

    def checkpoint(*, final=False, error=None):
        result = {
            "task_id": task.id, "family": task.family, "status": status,
            "finished": finished, "finish_origin": finish_origin, "stop_reason": stop_reason,
            "phase": phase, "pending_call": pending_call,
            "final_score": round(weighted / weights, 4) if weights and status == "ok" else None,
            "all_passed": status == "ok" and all(sg["passed"] for sg in subgoal_results),
            "subgoals_total": len(task.subgoals),
            "subgoals_passed": sum(sg["passed"] for sg in subgoal_results),
            "subgoal_results": subgoal_results, "steps_taken": len(trajectory),
            "step_budget": task.step_budget, "episode_wall_time_limit_s": task.wall_time_s,
            "horizon_hint": task.horizon_hint, "wall_time_s": round(time.time() - started, 2),
            "tool_counts": dict(tool_counts),
            "progress_curve": upstream.progress_curve(first_pass, len(trajectory), len(task.subgoals)),
        }
        stats = getattr(agent, "stats", None)
        if stats:
            result["agent_stats"] = stats
        if error:
            result["error"] = error
        if final and extra_stats:
            result.update(extra_stats())
        write_snapshot(run_dir / "checkpoint.json", result)
        if final:
            write_snapshot(run_dir / "result.json", result)
        return result

    def record(entry):
        trajectory.append(entry)
        with (run_dir / "trajectory.jsonl").open("a") as stream:
            stream.write(json.dumps(entry) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def context(params):
        return upstream.VerifyContext(workspace=workspace, oracle=loaded.oracle_dir,
                                      params=params, last_exec=last_exec)

    try:
        agent.reset(task, sandbox)
        observation = initial_observation(loaded, sandbox)
        checkpoint()
        while task.step_budget is None or len(trajectory) < task.step_budget:
            cancellation.check()
            step = len(trajectory)
            if task.wall_time_s is not None and time.time() - started > task.wall_time_s:
                record({"step": step, "tool": "<wall_time_exceeded>"})
                stop_reason = "wall_time_exceeded"
                break
            phase = "awaiting_model"
            checkpoint()
            call = agent.act(observation)
            if call.tool == "finish" and isinstance(call.args, dict):
                cancellation.finish_episode_work()
            pending_call = {"tool": call.tool, "args": call.args}
            phase = "tool_ready"
            checkpoint()
            cancellation.check()
            response = upstream._dispatch(sandbox, call)
            observation = {**response, "_tool": call.tool}
            tool_counts[call.tool] = tool_counts.get(call.tool, 0) + 1
            if call.tool == "execute_python":
                last_exec = response
            record({"step": step, "tool": call.tool, "args": call.args,
                    "observation": upstream._summarize_obs(call, response)})
            pending_call, phase = None, "tool_completed"
            checkpoint()
            cancellation.check()
            for subgoal in task.subgoals:
                if first_pass[subgoal.id] is None:
                    checked = upstream._eval_subgoal(subgoal, context, cheap_only=True)
                    if checked and checked[0]:
                        first_pass[subgoal.id] = step
            if call.tool == "finish":
                finished, stop_reason = True, "finished"
                exhausted_nudges = getattr(agent, "_nudges", 0) > getattr(agent, "max_nudges", 2)
                finish_origin = ("no_tool_guard" if exhausted_nudges and
                    str(call.args.get("message", "")).startswith("no tool call after ") else "agent")
                break
        else:
            stop_reason = "step_budget_exhausted"

        phase = "verifying"
        checkpoint()
        for subgoal in task.subgoals:
            cancellation.check()
            checked = upstream._eval_subgoal(subgoal, context, cheap_only=False)
            passed, score, verifiers = checked if checked else (True, 1.0, [])
            weighted += score * subgoal.weight
            weights += subgoal.weight
            subgoal_results.append({
                "id": subgoal.id, "desc": subgoal.desc, "weight": subgoal.weight,
                "passed": passed, "score": round(score, 4), "first_pass_step": first_pass[subgoal.id],
                "verifiers": [{"name": v.name, "passed": v.passed, "score": round(v.score, 4),
                               "detail": v.detail} for v in verifiers],
            })
        cancellation.check()
        status, phase = "ok", "completed"
        result = checkpoint(final=True)
    except (cancellation.Cancelled, cancellation.Deferred) as exc:
        status = "interrupted" if cancellation.requested() or isinstance(exc, cancellation.Cancelled) else "deferred"
        stop_reason = phase = status
        result = checkpoint(final=True, error=cancellation.reason() if cancellation.requested() else str(exc))
    except Exception as exc:
        try:
            cancellation.check()
        except cancellation.Cancelled:
            status, stop_reason, phase = "interrupted", "interrupted", "interrupted"
            result = checkpoint(final=True, error=cancellation.reason())
        except cancellation.Deferred as deferred:
            status, stop_reason, phase = "deferred", "deferred", "deferred"
            result = checkpoint(final=True, error=str(deferred))
        else:
            status, stop_reason, phase = "error", "episode_exception", "failed"
            checkpoint(final=True, error=f"{type(exc).__name__}: {exc}")
            raise
    finally:
        sandbox.close()
    return result
