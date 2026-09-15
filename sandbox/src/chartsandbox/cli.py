"""Command-line entry point.

    chartsandbox validate <task_dir>
    chartsandbox run <task_dir> [--agent scripted|llm] [--model ID] [--judge ID] ...

`--agent scripted` (the default) replays `sim_agent.json` and calls no API — it
smoke-tests the sandbox. `--agent llm` puts a real multimodal model in the loop;
`--judge <model>` replaces the mock rubric scorer with a real one. Either flag
spends money, so neither is on by default.

`--protocol` picks how a real model acts: `xml` (the default) is the tagged
`<reasoning>/<code>/<decision>` loop, `tools` is tool calling. Running one model
under both is the response-protocol ablation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent import LLMAgent, ScriptedAgent, TaggedAgent
from .contract import load_task
from .metrics import summarize
from .providers import DEFAULT_MODEL, get_provider
from .runner import run_episode

_EFFORTS = ("default", "none", "minimal", "low", "medium", "high", "xhigh", "max")


def _cmd_validate(args) -> int:
    lt = load_task(args.task_dir)
    print(f"OK: task {lt.task.id!r} ({lt.task.family}), "
          f"{len(lt.task.subgoals)} subgoal(s), budget={lt.task.step_budget}")
    for sg in lt.task.subgoals:
        vt = ", ".join(v.type for v in sg.verifiers)
        print(f"  - {sg.id} (w={sg.weight}): [{vt}]")
    return 0


def _build_agent(args):
    if args.agent == "scripted":
        task_dir = Path(args.task_dir)
        sim_path = Path(args.sim) if args.sim else (task_dir / "sim_agent.json")
        if not sim_path.exists():
            print(f"error: no scripted-agent file at {sim_path}. "
                  f"Pass --sim, or add sim_agent.json to the task.", file=sys.stderr)
            return None
        return ScriptedAgent(json.loads(sim_path.read_text()))

    provider = get_provider(args.model, max_tokens=args.max_tokens, base_url=args.base_url,
                            **({"effort": args.effort} if args.effort != "default" else {}))
    if args.protocol == "tools":
        return LLMAgent(provider, max_history_images=args.max_history_images)
    kwargs = {}
    if args.no_turn_limit:
        kwargs["max_turns"] = None          # explicit: no cap
    elif args.max_turns is not None:
        kwargs["max_turns"] = args.max_turns
    return TaggedAgent(provider, max_history_images=args.max_history_images,
                       max_images_per_turn=args.max_images_per_turn, **kwargs)


def _cmd_run(args) -> int:
    agent = _build_agent(args)
    if agent is None:
        return 2

    extra_stats = None
    if args.judge:
        from .judge import install_judge

        judge_provider = install_judge(args.judge, base_url=args.base_url,
                                       **({"judge_effort": args.judge_effort}
                                          if args.judge_effort != "default" else {}))
        # Windowed from here: a judge outlives one episode, and its lifetime totals
        # would bill this task for every task scored before it.
        judge_mark = judge_provider.mark()

        def extra_stats() -> dict:
            return {"judge_stats": judge_provider.episode_stats(judge_mark)}

    result = run_episode(args.task_dir, agent, out_dir=args.out, extra_stats=extra_stats)
    print(summarize(result))
    tu = result.get("token_usage") or {}
    for key, label in (("agent_stats", "agent"), ("judge_stats", "judge")):
        stats = result.get(key)
        if stats:
            t = tu.get(label, {})
            print(f"\n{label}:      {stats['model']} via {stats['provider']}, "
                  f"{stats['calls']} call(s)")
            print(f"             in={t.get('input_tokens', 0)} out={t.get('output_tokens', 0)}"
                  + (f" (reasoning {t['reasoning_tokens']})" if t.get("reasoning_tokens") else "")
                  + (f" (cached in {t['cache_read_input_tokens']})"
                     if t.get("cache_read_input_tokens") else ""))
            cost = stats["cost_usd"]
            print(f"             cost={'$%.4f' % cost if cost is not None else 'n/a (no price on file)'}")
    if tu.get("total"):
        t = tu["total"]
        cost = t.get("cost_usd")
        print(f"\ntokens:      {t['total_tokens']} total ({t['input_tokens']} in, "
              f"{t['output_tokens']} out) over {t['calls']} call(s)"
              + (f"  cost=${cost:.4f}" if cost is not None else ""))
    print(f"\nwrote: {result['_run_dir']}/result.json")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="chartsandbox")
    sub = p.add_subparsers(dest="cmd", required=True)

    pv = sub.add_parser("validate", help="validate a task instance against the contract")
    pv.add_argument("task_dir")
    pv.set_defaults(fn=_cmd_validate)

    pr = sub.add_parser("run", help="run an agent through a task")
    pr.add_argument("task_dir")
    pr.add_argument("--agent", choices=("scripted", "llm"), default="scripted",
                    help="scripted replays sim_agent.json (default); llm calls a real model")
    pr.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"agent model: claude-opus-5, gpt-5.4, or vllm:<hf-repo-id> "
                         f"for a locally served one (default: {DEFAULT_MODEL})")
    pr.add_argument("--base-url", default=None,
                    help="endpoint override for vllm, DeepSeek or OpenRouter; "
                         "defaults to $VLLM_BASE_URL or http://localhost:8000/v1")
    pr.add_argument("--protocol", choices=("xml", "tools"), default="xml",
                    help="how an llm agent acts: xml = <reasoning>/<code>/<decision> "
                         "iterations (default), tools = tool calling")
    pr.add_argument("--max-turns", type=int, default=None,
                    help="iteration cap for --protocol xml (default: the task's max_turns); "
                         "clamped down if the task's step_budget cannot fit it")
    pr.add_argument("--no-turn-limit", action="store_true",
                    help="run --protocol xml uncapped: only the agent's own "
                         "<decision>stop</decision> ends the episode")
    pr.add_argument("--max-images-per-turn", type=int, default=4,
                    help="images auto-attached per xml iteration (default: 4)")
    pr.add_argument("--effort", choices=_EFFORTS, default="high",
                    help="reasoning effort for the agent; ignored on vllm (default: high)")
    pr.add_argument("--max-tokens", type=int, default=16000,
                    help="output cap per model call; covers thinking too (default: 16000)")
    pr.add_argument("--max-history-images", type=int, default=0,
                    help="keep only the N most recent images in context (0 = keep all)")
    pr.add_argument("--judge", metavar="MODEL",
                    help="score vlm_judge rubrics with this model instead of the mock")
    pr.add_argument("--judge-effort", choices=_EFFORTS, default="high",
                    help="reasoning effort for the judge (default: high)")
    pr.add_argument("--sim", help="path to scripted-agent JSON (default: <task>/sim_agent.json)")
    pr.add_argument("--out", help="output run dir (default: <task>/runs/run_<ts>)")
    pr.set_defaults(fn=_cmd_run)

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except (RuntimeError, ValueError) as exc:
        # Misconfiguration (unroutable model, unreachable endpoint, missing
        # credentials) — the message is the useful part, not the traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
