"""Dry-run the tagged-text agent against a fake model: no API key, no cost, no network.

Companion to `dry_run_agent.py`, which covers the tool-calling protocol. Here the
model answers in `<reasoning>/<code>/<decision>` and the things that are easy to get
silently wrong are different:

    the chart the agent rendered is attached automatically, without being asked
    the episode ends when the *agent* says so — and stop_reason says which way

plus code arriving with a `stop` still running before the episode closes, a turn
budget that actually binds, unparseable turns costing reminders but no iteration,
and no tool parameters being sent to a backend that was told there are none.

    python scripts/dry_run_tagged.py
"""
from __future__ import annotations

import base64
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

SANDBOX = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX / "src"))

from chartsandbox.agent import TaggedAgent                    # noqa: E402
from chartsandbox.protocol import CONTINUE, STOP, parse_tagged  # noqa: E402
from chartsandbox.providers.anthropic_provider import AnthropicProvider  # noqa: E402
from chartsandbox.metrics import summarize                    # noqa: E402
from chartsandbox.runner import run_episode                   # noqa: E402

TASK = SANDBOX / "tasks" / "debug_repair_01"

failures: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(label)


# --- the parser -------------------------------------------------------------

print("tag parsing")

t = parse_tagged("<reasoning>looks wrong</reasoning>\n<code>\nprint(1)\n</code>\n<decision>continue</decision>")
check("the documented shape parses", t.reasoning == "looks wrong" and t.code == "print(1)"
      and t.decision == CONTINUE and not t.warnings, f"{t!r}")

t = parse_tagged("<CODE>\nprint(1)\n</CODE><Decision> STOP. </Decision>")
check("tags and decisions are case/punctuation insensitive",
      t.code == "print(1)" and t.decision == STOP, f"{t!r}")

t = parse_tagged("<code>\n```python\nprint(1)\n```\n</code><decision>continue</decision>")
check("a fence inside <code> is stripped", t.code == "print(1)", f"code={t.code!r}")

t = parse_tagged("Here you go:\n```python\nprint(1)\n```\nthat should do it")
check("fenced code with no tags is recovered", t.code == "print(1)" and t.actionable, f"{t!r}")
check("...and the deviation is recorded", any("fenced" in w for w in t.warnings), f"{t.warnings}")

t = parse_tagged("<code>\nprint(1)\n</code>")
check("a missing <decision> defaults to continue, not stop", t.decision == CONTINUE, f"{t!r}")
check("...and is recorded", any("decision" in w for w in t.warnings), f"{t.warnings}")

t = parse_tagged("<reasoning>thinking</reasoning>\n<code>\nprint(1)")
check("a truncated closing tag still yields code", t.code == "print(1)", f"code={t.code!r}")

t = parse_tagged("<decision>stop</decision>")
check("stop with no code is a valid action", t.actionable and t.stopping, f"{t!r}")

t = parse_tagged("I think the chart looks fine now.")
check("prose with no action is rejected", not t.actionable, f"{t!r}")

t = parse_tagged("<code>\nprint(1)\n</code><code>\nprint(2)\n</code><decision>continue</decision>")
check("two <code> blocks do not double-execute", t.code == "print(1)", f"code={t.code!r}")
check("...and the ambiguity is recorded", any("multiple" in w for w in t.warnings), f"{t.warnings}")

t = parse_tagged("<code>\nprint(1)\n</code><decision>the chart is finished</decision>")
check("a wordy decision is read, not discarded", t.decision == STOP, f"{t!r}")


# --- fake model -------------------------------------------------------------


def _text(t):
    return SimpleNamespace(type="text", text=t)


def _thinking(t):
    return SimpleNamespace(type="thinking", thinking=t, signature="sig-abc")


def _message(content):
    return SimpleNamespace(
        content=content, stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=1200, output_tokens=180,
                              cache_read_input_tokens=800, cache_creation_input_tokens=0),
    )


class FakeTagged(AnthropicProvider):
    """AnthropicProvider with the one network call replaced by a script of replies."""

    def __init__(self, script):
        super().__init__("claude-opus-5", client=object())
        self.script = list(script)
        self.requests: list[dict] = []

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        reply = self.script[min(len(self.requests) - 1, len(self.script) - 1)]
        return _message([_thinking("weighing the options"), _text(reply)])


def _images_in(messages: list[dict]) -> list[dict]:
    found = []
    for m in messages:
        if isinstance(m["content"], list):
            found += [b for b in m["content"]
                      if isinstance(b, dict) and b.get("type") == "image"]
    return found


def _run(script, task=TASK, **agent_kw):
    provider = FakeTagged(script)
    agent = TaggedAgent(provider, **agent_kw)
    out_dir = Path(tempfile.mkdtemp(prefix="chartsandbox_tagged_"))
    result = run_episode(task, agent, out_dir=out_dir)
    return provider, agent, result, out_dir


def _task_variant(**budget_overrides):
    """A copy of the example task with its budget fields overridden."""
    import yaml

    root = Path(tempfile.mkdtemp(prefix="chartsandbox_task_"))
    tdir = root / "task"
    shutil.copytree(TASK, tdir, ignore=shutil.ignore_patterns("runs"))
    spec = yaml.safe_load((tdir / "task.yaml").read_text())
    spec.update(budget_overrides)
    (tdir / "task.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    return tdir


def _opening_text(provider):
    return provider.requests[0]["messages"][0]["content"][0]["text"]


SOLUTION = (TASK / "oracle" / "solution.py").read_text()

TRY_BROKEN = """<reasoning>
Run broken.py first to see how it fails.
</reasoning>
<code>
exec(compile(open('broken.py').read(), 'broken.py', 'exec'))
</code>
<decision>continue</decision>"""

FIX = f"""<reasoning>
The traceback names the bug. Rewriting broken.py and re-running it.
</reasoning>
<code>
from pathlib import Path
Path('broken.py').write_text({SOLUTION!r})
exec(compile(open('broken.py').read(), 'broken.py', 'exec'))
</code>
<decision>continue</decision>"""

CONFIRM_AND_STOP = """<reasoning>
The rendered pie chart matches the original, so the repair is done.
</reasoning>
<code>
print('final check:', __import__('os').path.exists('out.png'))
</code>
<decision>stop</decision>"""


# --- a normal episode -------------------------------------------------------

print("\nepisode: agent works, then stops on its own")
provider, agent, result, out_dir = _run([TRY_BROKEN, FIX, CONFIRM_AND_STOP])
try:
    png = out_dir / "workspace" / "out.png"
    check("three iterations plus the close", result["steps_taken"] == 4,
          f"steps_taken={result['steps_taken']}")
    check("iterations counted, not steps", agent.stats["turns_taken"] == 3,
          f"turns_taken={agent.stats['turns_taken']}")
    check("episode closed cleanly", result["finished"] is True)
    check("stop_reason attributes it to the agent",
          agent.stats["stop_reason"] == "agent_stop", f"{agent.stats['stop_reason']}")
    check("the decision trail is recorded",
          agent.stats["decisions"] == [CONTINUE, CONTINUE, STOP], f"{agent.stats['decisions']}")
    check("no format deviations on well-formed replies",
          agent.stats["format_warnings"] == [], f"{agent.stats['format_warnings']}")
    check("the fake agent actually rendered a chart", png.exists() and png.stat().st_size > 0)

    # --- the blocker: the chart comes back without being asked for ----------
    print("\nautomatic inspection")
    expected = base64.b64encode(png.read_bytes()).decode()
    # request[0] opens the episode, [1] follows the failing run, [2] follows the fix.
    check("no image before anything was rendered",
          not _images_in(provider.requests[0]["messages"])
          and not _images_in(provider.requests[1]["messages"]))
    images = _images_in(provider.requests[2]["messages"])
    check("the chart is attached on the very next turn", len(images) == 1, f"found {len(images)}")
    check("it carries the real bytes of out.png",
          bool(images) and images[0]["source"]["data"] == expected)
    check("declared media_type matches",
          bool(images) and images[0]["source"]["media_type"] == "image/png")
    check("the agent never had to call a view tool",
          "view_image" not in result["tool_counts"], f"{result['tool_counts']}")

    print("\nrequest shape")
    check("no tool parameters sent under this protocol",
          all("tools" not in r and "tool_choice" not in r for r in provider.requests))
    check("system prompt carries a cache breakpoint",
          provider.requests[0]["system"][0].get("cache_control") is not None)
    check("the opening observation describes tags, not tools",
          "<decision>" in provider.requests[0]["messages"][0]["content"][0]["text"]
          and "execute_python" not in provider.requests[0]["messages"][0]["content"][0]["text"])
    check("the iteration budget is stated to the model",
          "Iteration budget: 10" in provider.requests[0]["messages"][0]["content"][0]["text"])

    print("\ntoken accounting")
    tu = result["token_usage"]
    # FakeTagged reports 1200 in / 180 out / 800 cached per call, over 3 calls.
    check("agent tokens are recorded per episode",
          tu["agent"]["input_tokens"] == 3600 and tu["agent"]["output_tokens"] == 540,
          f"{tu['agent']}")
    check("the cheap cached span is kept as a subset of input",
          tu["agent"]["cache_read_input_tokens"] == 2400
          and tu["agent"]["cache_read_input_tokens"] <= tu["agent"]["input_tokens"],
          f"{tu['agent']}")
    check("totals add up across roles", tu["total"]["total_tokens"]
          == tu["total"]["input_tokens"] + tu["total"]["output_tokens"])
    check("a judge that never ran contributes zero, not a missing key",
          tu["judge"]["total_tokens"] == 0, f"{tu['judge']}")
    check("call count matches the provider's", tu["agent"]["calls"] == len(provider.requests),
          f"{tu['agent']['calls']} vs {len(provider.requests)}")

    print("\ntrajectory log")
    lines = (out_dir / "trajectory.jsonl").read_text()
    check("no base64 image bytes in trajectory.jsonl", "base64" not in lines)
    check("the executed code is logged", "broken.py" in lines)
    check("stop_reason lands in result.json",
          result["agent_stats"]["stop_reason"] == "agent_stop")
    check("the episode-level stop_reason says it finished",
          result["stop_reason"] == "finished", f"{result['stop_reason']}")
    check("finish_origin attributes it to the agent, not a guard",
          result["finish_origin"] == "agent", f"{result['finish_origin']}")
finally:
    shutil.rmtree(out_dir, ignore_errors=True)


# --- stop arriving together with code ---------------------------------------

print("\nepisode: stop and code in the same turn")
provider, agent, result, out_dir = _run([FIX.replace("<decision>continue</decision>",
                                                     "<decision>stop</decision>")])
try:
    png = out_dir / "workspace" / "out.png"
    check("the code still ran before the episode closed", png.exists(),
          "out.png was never written")
    check("one iteration was spent", agent.stats["turns_taken"] == 1,
          f"turns_taken={agent.stats['turns_taken']}")
    check("attributed to the agent", agent.stats["stop_reason"] == "agent_stop")
    check("that final chart is what got verified", result["final_score"] > 0,
          f"final_score={result['final_score']}")
finally:
    shutil.rmtree(out_dir, ignore_errors=True)


# --- the budget binds -------------------------------------------------------

print("\nepisode: agent never stops")
provider, agent, result, out_dir = _run([TRY_BROKEN], max_turns=3)
try:
    check("stopped at the iteration cap", agent.stats["turns_taken"] == 3,
          f"turns_taken={agent.stats['turns_taken']}")
    check("not attributed to the agent",
          agent.stats["stop_reason"] == "turn_budget_exhausted", f"{agent.stats['stop_reason']}")
    check("the cap is what bound it, not step_budget",
          result["steps_taken"] < result["step_budget"],
          f"steps={result['steps_taken']} budget={result['step_budget']}")
    check("finish_origin names the turn budget, not the agent",
          result["finish_origin"] == "turn_budget", f"{result['finish_origin']}")
    check("the last iteration is announced as last",
          any("this is the last one" in b.get("text", "")
              for m in provider.requests[-1]["messages"] if isinstance(m["content"], list)
              for b in m["content"] if isinstance(b, dict)))
finally:
    shutil.rmtree(out_dir, ignore_errors=True)


# --- the model cannot produce a parseable action -----------------------------

print("\nepisode: model never emits a usable action")
provider, agent, result, out_dir = _run(["I would suggest checking the axis labels."])
try:
    check("no iteration was spent on a non-action", agent.stats["turns_taken"] == 0,
          f"turns_taken={agent.stats['turns_taken']}")
    check("reminders were sent (max_nudges + 1 attempts)", agent.stats["parse_failures"] == 3,
          f"parse_failures={agent.stats['parse_failures']}")
    check("reported as a parse failure, not an agent stop",
          agent.stats["stop_reason"] == "parse_failure", f"{agent.stats['stop_reason']}")
    check("the episode still closed and scored", result["finished"] is True)
    check("finish_origin names the parse guard, not the agent",
          result["finish_origin"] == "parse_guard", f"{result['finish_origin']}")
finally:
    shutil.rmtree(out_dir, ignore_errors=True)


# --- a recoverable deviation -------------------------------------------------

print("\nepisode: model answers in a fenced block instead of tags")
provider, agent, result, out_dir = _run([
    "Let me look at the failure.\n```python\nprint('hello')\n```",
    CONFIRM_AND_STOP,
])
try:
    check("the iteration was not wasted", agent.stats["turns_taken"] == 2,
          f"turns_taken={agent.stats['turns_taken']}")
    check("no reminder was needed", agent.stats["parse_failures"] == 0,
          f"parse_failures={agent.stats['parse_failures']}")
    check("the deviation is recorded for format-adherence reporting",
          any("fenced" in w for w in agent.stats["format_warnings"]),
          f"{agent.stats['format_warnings']}")
finally:
    shutil.rmtree(out_dir, ignore_errors=True)


# --- the released uncapped protocol -----------------------------------------

print("\nepisode: uncapped budgets, as the released A/B/C suite writes them")
tdir = _task_variant(step_budget=None, wall_time_s=None, max_turns=None)
provider, agent, result, out_dir = _run([TRY_BROKEN, FIX, CONFIRM_AND_STOP], task=tdir)
try:
    check("a task with null budgets loads at all", result["task_id"] == "debug_repair_01")
    check("no iteration cap is in force", agent.stats["max_turns"] is None,
          f"max_turns={agent.stats['max_turns']}")
    check("the agent's own stop is what ended it",
          agent.stats["stop_reason"] == "agent_stop", f"{agent.stats['stop_reason']}")
    check("the model is told there is no cap", "no iteration cap" in _opening_text(provider),
          _opening_text(provider)[-120:])
    check("per-iteration feedback says so too",
          any("no iteration cap" in b.get("text", "")
              for m in provider.requests[-1]["messages"] if isinstance(m["content"], list)
              for b in m["content"] if isinstance(b, dict)))
    check("the summary renders it rather than crashing on None",
          "uncapped" in summarize(result), summarize(result).splitlines()[3])
finally:
    shutil.rmtree(out_dir, ignore_errors=True)
    shutil.rmtree(tdir.parent, ignore_errors=True)


# --- an interaction-budget sweep rung ---------------------------------------

print("\nepisode: step-ladder rung too small for the requested iterations")
tdir = _task_variant(step_budget=3, max_turns=10)
provider, agent, result, out_dir = _run([TRY_BROKEN], task=tdir)
try:
    check("a 3-step ladder rung loads (it is a sweep, not a malformed task)",
          result["step_budget"] == 3)
    check("the cap is clamped to what the steps allow", agent.stats["max_turns"] == 2,
          f"max_turns={agent.stats['max_turns']}")
    check("what was originally asked for is recorded",
          agent.stats["max_turns_requested"] == 10, f"{agent.stats['max_turns_requested']}")
    check("the agent was not cut off mid-budget by the runner",
          agent.stats["stop_reason"] == "turn_budget_exhausted", f"{agent.stats['stop_reason']}")
    check("the clamp is visible in the summary",
          "clamped from 10" in summarize(result))
finally:
    shutil.rmtree(out_dir, ignore_errors=True)
    shutil.rmtree(tdir.parent, ignore_errors=True)


print()
if failures:
    print(f"{len(failures)} check(s) FAILED:")
    for f in failures:
        print(f"  - {f}")
    raise SystemExit(1)
print("all checks passed — the tagged-text loop is wired correctly (no API calls were made)")
