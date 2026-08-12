"""Dry-run the LLM agent against a fake model: no API key, no cost, no network.

It drives a real episode end to end — real kernel, real files, real verifiers —
with a canned trajectory standing in for the model, then asserts on the request
payloads the provider *would* have sent. What it is really checking is the thing
that is easy to get silently wrong:

    the bytes of the chart the agent rendered actually reach the model

plus tool_use/tool_result pairing, thinking-block round-tripping, that image bytes
stay out of the trajectory log, and that the OpenAI translation degrades correctly
(its tool messages are text-only, so images must follow as a user message).

    python scripts/dry_run_agent.py
"""
from __future__ import annotations

import base64
import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

SANDBOX = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX / "src"))

from chartsandbox.agent import LLMAgent                       # noqa: E402
from chartsandbox.providers.anthropic_provider import AnthropicProvider  # noqa: E402
from chartsandbox.providers.openai_provider import IMAGE_HANDOFF, OpenAIProvider  # noqa: E402
from chartsandbox.providers.vllm_provider import (  # noqa: E402
    IMAGE_HANDOFF as VLLM_HANDOFF,
    VLLMProvider,
    _parse_verdict,
)
from chartsandbox.runner import run_episode                   # noqa: E402

TASK = SANDBOX / "tasks" / "debug_repair_01"

failures: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(label)


def _kind(block) -> str | None:
    """Block type, whether it's a dict we built or a provider-native object."""
    return block.get("type") if isinstance(block, dict) else getattr(block, "type", None)


def _field(block, key):
    return block.get(key) if isinstance(block, dict) else getattr(block, key, None)


def _blocks_of(messages: list[dict], kind: str) -> list:
    """Every block of `kind` in a request — assistant turns are echoed as native
    objects (that is how thinking signatures survive), so both forms count."""
    out = []
    for m in messages:
        content = m["content"]
        if isinstance(content, list):
            out += [b for b in content if _kind(b) == kind]
    return out


def _images_in(messages: list[dict]) -> list[dict]:
    """Every Anthropic image block in a request, including nested in tool_results."""
    found: list[dict] = []
    for m in messages:
        content = m["content"]
        if not isinstance(content, list):
            continue
        for b in content:
            if not isinstance(b, dict):
                continue  # provider-native object (an echoed thinking block)
            if b.get("type") == "image":
                found.append(b)
            elif b.get("type") == "tool_result":
                found += [c for c in b.get("content", [])
                          if isinstance(c, dict) and c.get("type") == "image"]
    return found


# --- fake model -------------------------------------------------------------


def _text(t):
    return SimpleNamespace(type="text", text=t)


def _thinking(t):
    return SimpleNamespace(type="thinking", thinking=t, signature="sig-abc")


def _tool_use(i, name, args):
    return SimpleNamespace(type="tool_use", id=f"toolu_{i}", name=name, input=args)


def _message(content, stop_reason="tool_use"):
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=1200, output_tokens=180,
                              cache_read_input_tokens=800, cache_creation_input_tokens=0),
    )


class FakeAnthropic(AnthropicProvider):
    """AnthropicProvider with the single network call replaced by a script."""

    def __init__(self, script):
        super().__init__("claude-opus-5", client=object())
        self.script = list(script)
        self.requests: list[dict] = []

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        name, args = self.script[len(self.requests) - 1]
        i = len(self.requests)
        blocks = [_thinking(f"step {i}: deciding what to do"),
                  _text(f"Calling {name}.")]
        if name == "finish":
            blocks.append(_tool_use(i, "finish", args))
            return _message(blocks, stop_reason="tool_use")
        blocks.append(_tool_use(i, name, args))
        return _message(blocks)


# --- run --------------------------------------------------------------------

solution = (TASK / "oracle" / "solution.py").read_text()
script = [
    ("read_file", {"path": "broken.py"}),
    ("execute_python", {"code": "exec(compile(open('broken.py').read(), 'broken.py', 'exec'))"}),
    ("write_file", {"path": "broken.py", "content": solution}),
    ("execute_python", {"code": "exec(compile(open('broken.py').read(), 'broken.py', 'exec'))"}),
    ("view_image", {"path": "out.png"}),
    ("finish", {"message": "fixed the typo, re-rendered, and checked the chart"}),
]

provider = FakeAnthropic(script)
agent = LLMAgent(provider)
out_dir = Path(tempfile.mkdtemp(prefix="chartsandbox_dryrun_"))
try:
    result = run_episode(TASK, agent, out_dir=out_dir)
    workspace = out_dir / "workspace"
    png = workspace / "out.png"

    print("\nepisode")
    check("all six scripted tool calls executed", result["steps_taken"] == 6,
          f"steps_taken={result['steps_taken']}")
    check("agent finished cleanly", result["finished"] is True)
    check("the fake agent actually rendered a chart", png.exists() and png.stat().st_size > 0)

    # --- the blocker: does the image reach the model? -----------------------
    print("\nmultimodal round trip")
    expected_b64 = base64.b64encode(png.read_bytes()).decode()
    req = provider.requests[5]  # the request sent after view_image returned
    images = _images_in(req["messages"])
    check("an image block is present in the request", len(images) == 1, f"found {len(images)}")
    check("it carries the real bytes of out.png",
          bool(images) and images[0]["source"]["data"] == expected_b64)
    check("declared media_type matches", bool(images) and images[0]["source"]["media_type"] == "image/png")
    check("the image is inside the tool_result, not a loose user message",
          any(b.get("type") == "tool_result"
              and any(c.get("type") == "image" for c in b.get("content", []))
              for m in req["messages"] if isinstance(m["content"], list)
              for b in m["content"] if isinstance(b, dict)))
    check("no image sent before the agent asked to see one",
          all(not _images_in(r["messages"]) for r in provider.requests[:5]))

    # --- protocol invariants ------------------------------------------------
    print("\nprotocol")
    last = provider.requests[-1]["messages"]
    uses = _blocks_of(last, "tool_use")
    results = _blocks_of(last, "tool_result")
    check("every tool_use has a matching tool_result", len(uses) == len(results),
          f"{len(uses)} tool_use vs {len(results)} tool_result")
    check("tool_use ids line up with tool_result ids",
          [_field(u, "id") for u in uses] == [_field(r, "tool_use_id") for r in results])
    check("thinking blocks survive the round trip (echoed via raw)",
          any(not isinstance(b, dict) for b in _blocks_of(last, "thinking")))
    check("system prompt carries a cache breakpoint",
          provider.requests[0]["system"][0].get("cache_control") == {"type": "ephemeral"})
    check("parallel tool use disabled",
          provider.requests[0]["tool_choice"].get("disable_parallel_tool_use") is True)
    check("all six sandbox tools offered", len(provider.requests[0]["tools"]) == 6)

    # --- the log must not grow image bytes ----------------------------------
    print("\ntrajectory log")
    traj = (out_dir / "trajectory.jsonl").read_text()
    check("no base64 image bytes in trajectory.jsonl", expected_b64[:64] not in traj)
    check("view_image still logged with its dimensions", '"image":' in traj)

    # --- accounting ---------------------------------------------------------
    print("\naccounting")
    stats = result.get("agent_stats", {})
    check("token usage recorded", stats.get("usage", {}).get("output_tokens") == 6 * 180)
    check("cost estimated for a priced model", isinstance(stats.get("cost_usd"), float)
          and stats["cost_usd"] > 0, f"cost_usd={stats.get('cost_usd')}")
    check("model reasoning captured for debugging", len(stats.get("transcript", [])) == 6
          and bool(stats["transcript"][0]["thinking"]))

    # --- cross-provider translation ----------------------------------------
    print("\nopenai translation of the same history")
    native = OpenAIProvider("gpt-test", client=object())._to_native(agent.messages)
    outputs = [m for m in native if m.get("type") == "function_call_output"]
    check("tool results become function_call_output items", len(outputs) == 5,
          f"{len(outputs)} found")
    check("every tool output is a plain string (the API takes no image there)",
          all(isinstance(m["output"], str) for m in outputs))
    idx = next(i for i, m in enumerate(native)
               if m.get("type") == "function_call_output" and IMAGE_HANDOFF in m["output"])
    follow = native[idx + 1] if idx + 1 < len(native) else {}
    check("the image follows as a user message instead",
          follow.get("role") == "user"
          and any(c.get("type") == "input_image" for c in follow.get("content", [])))
    check("that input_image is a real data URI with the same bytes",
          follow.get("role") == "user"
          and any(c["image_url"] == f"data:image/png;base64,{expected_b64}"
                  for c in follow.get("content", []) if c.get("type") == "input_image"))
    check("function_call arguments serialised as a JSON string",
          all(isinstance(m["arguments"], str)
              for m in native if m.get("type") == "function_call"))
    check("call_ids carried through unchanged",
          [m["call_id"] for m in native if m.get("type") == "function_call"][:5]
          == [m["call_id"] for m in outputs])

    print("\nvllm translation of the same history")
    vnative = VLLMProvider("local-vlm", client=object())._to_native(agent.messages)
    vtools = [m for m in vnative if m.get("role") == "tool"]
    check("tool results become role=tool messages", len(vtools) == 5, f"{len(vtools)} found")
    check("every tool message is a plain string", all(isinstance(m["content"], str) for m in vtools))
    vidx = next(i for i, m in enumerate(vnative)
                if m.get("role") == "tool" and VLLM_HANDOFF in m["content"])
    vfollow = vnative[vidx + 1] if vidx + 1 < len(vnative) else {}
    check("the image follows as a user message with a data URI",
          vfollow.get("role") == "user"
          and any(c.get("type") == "image_url"
                  and c["image_url"]["url"] == f"data:image/png;base64,{expected_b64}"
                  for c in vfollow.get("content", [])))

    print("\nvllm judge output salvage (guided decoding can still be wrapped)")
    clean = '{"passed": true, "score": 0.8, "detail": "ok"}'
    check("clean json parses", _parse_verdict(clean).score == 0.8)
    check("fenced json parses", _parse_verdict(f"```json\n{clean}\n```").passed is True)
    check("json wrapped in prose parses",
          _parse_verdict(f"Here is my verdict:\n{clean}\nHope that helps.").detail == "ok")
    check("unparseable output fails closed rather than crashing",
          _parse_verdict("I think it looks fine!").passed is False)
finally:
    shutil.rmtree(out_dir, ignore_errors=True)

print()
if failures:
    print(f"{len(failures)} check(s) failed: " + "; ".join(failures))
    raise SystemExit(1)
print("all checks passed — the agent loop is wired correctly (no API calls were made)")
