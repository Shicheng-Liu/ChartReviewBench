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
from typing import Callable

from .agent import Agent, ToolCall
from .contract import LoadedTask, load_task
from .metrics import progress_curve
from .persistence import atomic_json, append_jsonl
from .providers import token_summary
from .tools import TOOL_SCHEMAS, VIEWABLE_MIME, Sandbox
from .verifiers import CHEAP, VerifyContext, run_verifier

_FILE_TOOLS = {"write_file", "read_file", "list_files", "view_image"}
_IMAGE_SUFFIXES = {"." + ext for ext in VIEWABLE_MIME}


def _is_valid_image(path: Path) -> bool:
    """Non-empty, and openable if it claims to be an image."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    if path.suffix.lower() not in _IMAGE_SUFFIXES:
        return True
    try:
        from PIL import Image

        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


class _LatestValidRenders:
    """The last readable version of every image the agent rendered.

    The interaction protocol evaluates an episode on the *latest valid* chart. A
    ledger is what makes that true: a final turn that raises part-way through
    writing its own output file leaves a truncated PNG behind, and without this
    the verifiers would score the wreckage rather than the chart the agent last
    actually produced. The store sits beside the run and never inside the
    workspace, so it is not something the agent can read, overwrite, or mistake
    for its own output.
    """

    def __init__(self, workspace: Path, store: Path):
        self.workspace = workspace
        self.store = store
        self.restored: list[str] = []

    def record(self, obs: dict) -> None:
        for rel in obs.get("images_created") or []:
            src = self.workspace / rel
            if not _is_valid_image(src):
                continue
            dst = self.store / rel
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            except OSError:
                pass

    def restore(self) -> list[str]:
        """Put back every render that is no longer readable in the workspace."""
        if not self.store.exists():
            return []
        for saved in sorted(self.store.rglob("*")):
            if not saved.is_file():
                continue
            rel = saved.relative_to(self.store)
            live = self.workspace / rel
            if _is_valid_image(live):
                continue
            try:
                live.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(saved, live)
                self.restored.append(str(rel))
            except OSError:
                pass
        return self.restored


def _action_brief(task, agent: Agent | None = None) -> str:  # noqa: ANN001
    """How the agent acts, in its own protocol's terms.

    An agent may describe its own action space (the tagged-text one has no tools to
    list); the tool listing is the fallback, which is what every tool-calling agent
    gets — and what an omitted agent gets, see `_initial_observation`.
    """
    brief = getattr(agent, "action_brief", None)
    if callable(brief):
        return brief(task)
    tool_list = "\n".join(f"  - {t['name']}: {t['description']}" for t in TOOL_SCHEMAS)
    budget = ("Step budget: %s." % task.step_budget if task.step_budget is not None
              else "There is no step budget; end the episode yourself.")
    return f"## Tools\n{tool_list}\n\nCall finish when done. {budget}"


def _initial_observation(lt: LoadedTask, sandbox: Sandbox, agent: Agent | None = None) -> dict:
    """The opening observation: the task, the workspace listing, and how to act.

    `agent` is optional because this is called from outside as well: the released
    benchmark's budget adapter calls it with two arguments to build the same opening
    text before driving the loop itself. Omitting it yields the tool listing, which
    is what a tool-calling episode wants anyway — so keep it optional.
    """
    task = lt.task
    files = sandbox.list_files(".").get("files", [])
    text = (
        f"# Task ({task.family})\n{task.instruction}\n\n"
        f"## Workspace files\n" + "\n".join(f"  - {f['name']}" for f in files) + "\n\n"
        + _action_brief(task, agent)
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
    """Compact form of an observation for the trajectory *log* (drop image bytes).

    This is for humans and result files only — never feed it back to the agent.
    The agent gets the full observation (see `run_episode`), because dropping
    `base64` here is exactly what would blind a multimodal agent to its own chart.
    """
    s = {k: v for k, v in obs.items() if k != "base64"}
    if tc.tool == "view_image" and obs.get("ok"):
        s["image"] = f"{obs.get('width')}x{obs.get('height')} {obs.get('mime')}"
    if tc.tool == "read_file" and "content" in s:
        s["content"] = s["content"][:200] + ("..." if len(s["content"]) > 200 else "")
    return s


#: `agent_stats.stop_reason` -> who the `finish` came from. A consumer should not
#: have to sniff the finish message to tell the agent's own stop from a guard.
_FINISH_ORIGIN = {
    "agent_stop": "agent",
    "no_tool_call": "no_tool_guard",
    "parse_failure": "parse_guard",
    "turn_budget_exhausted": "turn_budget",
}


def _finish_origin(agent: Agent, finished: bool) -> str | None:
    """Whether `finish` was the agent's judgement or a harness guard giving up."""
    if not finished:
        return None
    stats = getattr(agent, "stats", None) or {}
    return _FINISH_ORIGIN.get(stats.get("stop_reason"), "agent")


def _token_usage(result: dict) -> dict:
    """Per-role and total token counts for one episode.

    Kept next to the score rather than buried in the provider blocks because it is
    the number a suite gets aggregated on: cost per task, and cost per point of
    score, are both questions asked after the run, of the result file.
    """
    roles = {}
    for key, role in (("agent_stats", "agent"), ("judge_stats", "judge")):
        block = result.get(key) or {}
        roles[role] = token_summary(block.get("usage") or {},
                                    calls=block.get("calls", 0),
                                    model=block.get("model"))
    total_in = sum(r["input_tokens"] for r in roles.values())
    total_out = sum(r["output_tokens"] for r in roles.values())
    costs = [r.get("cost_usd") for r in roles.values()]
    return {
        **roles,
        "total": {
            "calls": sum(r["calls"] for r in roles.values()),
            "input_tokens": total_in,
            "output_tokens": total_out,
            "total_tokens": total_in + total_out,
            "reasoning_tokens": sum(r["reasoning_tokens"] for r in roles.values()),
            "cache_read_input_tokens": sum(r["cache_read_input_tokens"] for r in roles.values()),
            # None when any model in play has no price on file — a partial sum
            # would read as a complete one.
            "cost_usd": (round(sum(c for c in costs if c), 6)
                         if all(c is not None for c in costs) else None),
        },
    }


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


def run_episode(task_dir: str | Path, agent: Agent, out_dir: str | Path | None = None,
                extra_stats: Callable[[], dict] | None = None) -> dict:
    """Drive `agent` through one task and score it.

    `extra_stats` is called once, after verification, and merged into the result —
    it exists so the caller can record the cost of a *verifier* (the LLM judge is
    installed globally, so the runner has no other way to see it).
    """
    lt = load_task(task_dir)
    task = lt.task

    # fresh run workspace (never mutate the task template; oracle stays out of it)
    run_dir = Path(out_dir) if out_dir else (lt.task_dir / "runs" / time.strftime("run_%Y%m%d_%H%M%S"))
    if (run_dir / "result.json").exists():
        raise ValueError(f"completed result already exists at {run_dir}; choose a new output")
    run_ws = run_dir / "workspace"
    if run_ws.exists():
        shutil.rmtree(run_ws)
    run_ws.parent.mkdir(parents=True, exist_ok=True)
    if lt.workspace_dir.exists():
        shutil.copytree(lt.workspace_dir, run_ws)
    else:
        run_ws.mkdir(parents=True)

    # A result is the commit marker: callers must choose a fresh/archived run dir.
    trace_path = run_dir / "trajectory.jsonl"
    trace_path.write_text("")
    shutil.copy2(lt.task_dir / "task.yaml", run_dir / "task.yaml")
    config = lt.oracle_dir / "task_config.json"
    if config.is_file():
        shutil.copy2(config, run_dir / "task_config.json")
    sandbox = Sandbox(run_ws, task.step_timeout_s)
    try:
        renders = _LatestValidRenders(run_ws, run_dir / ".last_valid")
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
        # The agent sees the *full* observation, including image bytes from view_image;
        # `_tool` tells it which tool produced this one. The trajectory keeps the
        # summarized copy so result files stay readable.
        agent_obs: dict = _initial_observation(lt, sandbox, agent)
        atomic_json(run_dir / "agent_config.json", {
            "system_prompt": getattr(agent, "system", None),
            "initial_observation": agent_obs,
        })

        # `None` for either budget is uncapped, which the released A/B/C suite selects on
        # purpose: with no ceiling the agent's own stop is what ends the episode. Nothing
        # else bounds a run in that mode — per-step timeouts still fire, but an agent that
        # never stops runs until it is interrupted.
        step = 0
        stop_reason = "step_budget_exhausted"   # the loop's own exit, if nothing else fires
        while task.step_budget is None or step < task.step_budget:
            if task.wall_time_s is not None and time.time() - t0 > task.wall_time_s:
                trajectory.append({"step": step, "tool": "<wall_time_exceeded>"})
                append_jsonl(trace_path, trajectory[-1])
                stop_reason = "wall_time_exceeded"
                break
            tc = agent.act(agent_obs)
            # Persist the returned model response before executing its action.
            atomic_json(run_dir / "agent_checkpoint.json", {
                "task_id": task.id, "next_step": step,
                "agent_stats": getattr(agent, "stats", None),
                "action": {"tool": tc.tool, "args": tc.args},
            })
            obs = _dispatch(sandbox, tc)
            agent_obs = {**obs, "_tool": tc.tool}
            tool_counts[tc.tool] = tool_counts.get(tc.tool, 0) + 1
            if tc.tool == "execute_python":
                last_exec = obs
                renders.record(obs)
            trajectory.append({"step": step, "tool": tc.tool, "args": tc.args,
                               "observation": _summarize_obs(tc, obs)})

            # Preserve each image version: workspace images may be overwritten later.
            artifacts = []
            image_paths = list(obs.get("images_created") or []) + list(obs.get("figures_captured") or [])
            if tc.tool == "view_image" and obs.get("ok"):
                image_paths.append(tc.args["path"])
            for rel in dict.fromkeys(image_paths):
                src = run_ws / rel
                if src.is_file() and src.resolve().is_relative_to(run_ws.resolve()):
                    dest = run_dir / "renders" / str(step) / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dest)
                    artifacts.append(str(dest.relative_to(run_dir)))
            trajectory[-1]["render_artifacts"] = artifacts
            append_jsonl(trace_path, trajectory[-1])

            # cheap incremental subgoal check -> progress curve
            for sg in task.subgoals:
                if first_pass_step[sg.id] is None:
                    r = _eval_subgoal(sg, ctx_factory, cheap_only=True)
                    if r and r[0]:
                        first_pass_step[sg.id] = step

            if tc.tool == "finish":
                finished = True
                stop_reason = "finished"
                break
            step += 1

        # The episode is scored on the latest *valid* chart, so anything the final
        # turn left unreadable is rolled back to the last version that rendered.
        restored = renders.restore()

        atomic_json(run_dir / "agent_checkpoint.json", {
            "task_id": task.id, "agent_complete": True,
            "agent_stats": getattr(agent, "stats", None),
            "stop_reason": stop_reason, "restored_renders": restored,
        })
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

        steps_taken = len(trajectory)
        final_score = round(weighted_num / weighted_den, 4) if weighted_den else 0.0

        result = {
            "task_id": task.id,
            "family": task.family,
            "finished": finished,
            # Why the episode ended, and who ended it. Both are episode-level facts the
            # runner is the only place that knows; `agent_stats.stop_reason` refines the
            # second with the agent's own account.
            "stop_reason": stop_reason,
            "finish_origin": _finish_origin(agent, finished),
            # Non-empty means the final turn damaged a render and the score comes from
            # an earlier one. Worth knowing before trusting the number.
            "restored_renders": restored,
            "final_score": final_score,
            "all_passed": all(s["passed"] for s in subgoal_results),
            "subgoals_total": len(task.subgoals),
            "subgoals_passed": sum(1 for s in subgoal_results if s["passed"]),
            "subgoal_results": subgoal_results,
            "steps_taken": steps_taken,
            "step_budget": task.step_budget,
            # The cap, not the elapsed time — `wall_time_s` below is what the run took.
            "episode_wall_time_limit_s": task.wall_time_s,
            "horizon_hint": task.horizon_hint,
            "wall_time_s": round(time.time() - t0, 2),
            "tool_counts": tool_counts,
            "progress_curve": progress_curve(first_pass_step, steps_taken, len(task.subgoals)),
        }
        # Agents that talk to a model expose token/cost accounting; the scripted one
        # doesn't. Recording it here makes a run's price part of its result record.
        agent_stats = getattr(agent, "stats", None)
        if agent_stats:
            result["agent_stats"] = agent_stats
        if extra_stats is not None:
            result.update(extra_stats())

        # One place to read "what did this task cost", per role and in total. Built
        # from whatever stats blocks are present rather than from the providers, so a
        # scripted episode simply reports zeros instead of failing.
        result["token_usage"] = _token_usage(result)

        run_dir.mkdir(parents=True, exist_ok=True)
        result["artifact_version"] = 1
        result["evaluation_errors"] = [
            {"subgoal": sg["id"], "detail": v["detail"]}
            for sg in subgoal_results for v in sg["verifiers"]
            if v["detail"].startswith("verifier error:")
            or "judge returned unparseable output" in v["detail"]
        ]
        atomic_json(run_dir / "result.json", result)
        result["_run_dir"] = str(run_dir)
        return result
    finally:
        sandbox.close()
