"""Agent interface, a scripted agent, and a real multimodal LLM agent.

The sandbox is agent-agnostic: anything implementing `Agent.act` can be driven by
the runner. `ScriptedAgent` replays a fixed trajectory and is how we smoke-test
the sandbox itself. `LLMAgent` drives a real tool-calling model through the same
interface — see `providers/` for the per-backend wire translation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .providers import Provider, image_block, text_block, tool_result_block, user
from .tools import TOOL_SCHEMAS

SYSTEM_PROMPT = """\
You are an autonomous data-visualization agent working in a sandboxed Python \
workspace. Solve the task by calling tools, one call per turn.

The Python kernel is persistent: variables, imports and open figures survive \
across calls, so you can build up state instead of re-running everything. Only \
files inside the workspace exist, and there is no network access — no downloading \
data, no installing packages.

Look at your own output. After rendering a chart, view the image and check it \
against what the task asked for before you move on: a script that exits cleanly \
can still draw the wrong chart, mislabel an axis, or silently drop a series.

Call finish when the task is complete, or when you are blocked and no further tool \
call would help. In that final message, say what you did and what you verified \
visually — and if something is unfinished, say plainly what and why."""


@dataclass
class ToolCall:
    tool: str
    args: dict[str, Any]


class Agent:
    def reset(self, task, sandbox) -> None:  # noqa: ANN001
        """Called once at episode start."""

    def act(self, observation: dict) -> ToolCall:
        raise NotImplementedError


class ScriptedAgent(Agent):
    """Replays a predefined trajectory. `script` is a list of {tool, args}.

    This is the *simulated* agent: use it to verify the sandbox/verifiers, not to
    measure model ability. For smoke tests the script is typically an oracle
    trajectory that solves the task.
    """

    def __init__(self, script: list[dict]):
        self.script = script
        self.i = 0

    def reset(self, task, sandbox) -> None:  # noqa: ANN001
        self.i = 0

    def act(self, observation: dict) -> ToolCall:
        if self.i >= len(self.script):
            return ToolCall("finish", {"message": "end of script"})
        step = self.script[self.i]
        self.i += 1
        return ToolCall(step["tool"], step.get("args", {}))


class LLMAgent(Agent):
    """A real tool-calling, image-viewing agent.

    The runner hands us one observation per step and expects one ToolCall back, so
    the model's conversation lives here: each observation is appended as a tool
    result — images included — and the model's next tool call is returned.

    Parallel tool use is disabled at the provider, because one action per step is
    what keeps the trajectory, the progress curve and the persistent kernel in
    lockstep. `_pending` still queues a batch if a model emits one anyway: every
    tool_use in a turn must be answered, and all the results must go back in a
    single message, so they are collected and flushed together.
    """

    def __init__(self, provider: Provider, *, system: str = SYSTEM_PROMPT,
                 max_history_images: int = 0, max_nudges: int = 2):
        self.provider = provider
        self.system = system
        # 0 keeps every image. A positive cap bounds context on long episodes at
        # the cost of prompt-cache hits: demoting an old image rewrites history
        # behind the cache breakpoint, so the prefix has to be re-read.
        self.max_history_images = max_history_images
        self.max_nudges = max_nudges
        self.messages: list[dict] = []
        self.transcript: list[dict] = []
        self._pending: list[dict] = []
        self._results: list[dict] = []
        self._inflight: dict | None = None
        self._nudges = 0

    # -- lifecycle -----------------------------------------------------------
    def reset(self, task, sandbox) -> None:  # noqa: ANN001
        self.messages = []
        self.transcript = []
        self._pending = []
        self._results = []
        self._inflight = None
        self._nudges = 0

    @property
    def stats(self) -> dict:
        return {**self.provider.stats(), "transcript": self.transcript}

    # -- the loop ------------------------------------------------------------
    def act(self, observation: dict) -> ToolCall:
        if "_tool" not in observation:
            # Episode start: the task instruction and workspace listing.
            self.messages.append(user([text_block(observation.get("text", ""))]))
        elif self._inflight is not None:
            self._results.append(tool_result_block(
                self._inflight["id"],
                _observation_blocks(observation),
                is_error=_is_error(observation),
            ))

        if self._pending:  # mid-batch: no API call needed
            return self._dispatch(self._pending.pop(0))

        if self._results:  # batch done: every result goes back in one message
            self.messages.append(user(self._results))
            self._results = []

        for _ in range(self.max_nudges + 1):
            turn = self.provider.complete(self.system, self._history(), TOOL_SCHEMAS)
            self.messages.append({"role": "assistant", "content": turn.blocks,
                                  "raw": turn.raw, "provider": self.provider.name})
            self.transcript.append({"text": turn.text, "thinking": turn.thinking,
                                    "stop_reason": turn.stop_reason,
                                    "tools": [c["name"] for c in turn.tool_calls]})
            if turn.tool_calls:
                self._nudges = 0
                self._pending = list(turn.tool_calls)
                return self._dispatch(self._pending.pop(0))

            # Text with no tool call — say so and let it try again. This loop is
            # inside one step, so it costs wall time but not step budget.
            self._nudges += 1
            self.messages.append(user([text_block(
                "That turn contained no tool call, so nothing ran. Continue by "
                "calling a tool, or call finish if the task is complete or you are blocked."
            )]))

        return ToolCall("finish", {"message": (
            f"no tool call after {self.max_nudges} prompts; giving up"
        )})

    def _dispatch(self, call: dict) -> ToolCall:
        self._inflight = call
        return ToolCall(call["name"], call["args"])

    def _history(self) -> list[dict]:
        if self.max_history_images <= 0:
            return self.messages
        return _prune_images(self.messages, self.max_history_images)


# --- observation rendering --------------------------------------------------

_MAX_STREAM = 4000
_MAX_FILE = 20000


def _clip(s: str, limit: int) -> str:
    s = s or ""
    return s if len(s) <= limit else s[:limit] + f"\n... [clipped, {len(s)} chars total]"


def _is_error(obs: dict) -> bool:
    return obs.get("ok") is False or bool(obs.get("error"))


def _observation_blocks(obs: dict) -> list[dict]:
    """Turn one sandbox observation into blocks the model can actually read.

    The only place in the harness where image bytes reach the model: `view_image`
    returns base64, and it goes back as a real image block, not a description of one.
    """
    tool = obs.get("_tool")

    if tool == "view_image" and obs.get("ok") and obs.get("base64"):
        return [
            text_block(f"{obs.get('path')} — {obs.get('width')}x{obs.get('height')} "
                       f"{obs.get('mime')}"),
            image_block(obs.get("mime") or "image/png", obs["base64"]),
        ]

    if tool == "execute_python":
        parts = ["error" if obs.get("error") else "ran without error"]
        if obs.get("stdout"):
            parts.append("stdout:\n" + _clip(obs["stdout"], _MAX_STREAM))
        if obs.get("stderr"):
            parts.append("stderr:\n" + _clip(obs["stderr"], _MAX_STREAM))
        if obs.get("error"):
            parts.append("traceback:\n" + _clip(obs["error"], _MAX_STREAM))
        if obs.get("files_changed"):
            parts.append("files changed: " + ", ".join(obs["files_changed"][:50]))
        if obs.get("images_created"):
            parts.append("images written: " + ", ".join(obs["images_created"][:50]))
        return [text_block("\n\n".join(parts))]

    if tool == "read_file" and obs.get("ok"):
        return [text_block(_clip(obs.get("content", ""), _MAX_FILE))]

    if tool == "list_files" and obs.get("ok"):
        lines = [f"{f['name']}{'/' if f['is_dir'] else ''}"
                 + ("" if f["is_dir"] else f"  {f['bytes']} bytes")
                 for f in obs.get("files", [])]
        return [text_block("\n".join(lines) or "(empty)")]

    # write_file, errors, anything unrecognised: compact JSON minus image bytes.
    return [text_block(json.dumps(
        {k: v for k, v in obs.items() if k not in ("base64", "_tool")},
        default=str)[:_MAX_FILE])]


def _prune_images(messages: list[dict], keep: int) -> list[dict]:
    """Copy of `messages` with all but the last `keep` images replaced by a stub."""
    positions: list[tuple[int, int, int | None]] = []  # (msg, block, nested block)
    for mi, m in enumerate(messages):
        for bi, b in enumerate(m["content"]):
            if b["type"] == "image":
                positions.append((mi, bi, None))
            elif b["type"] == "tool_result":
                for ci, c in enumerate(b["content"]):
                    if c["type"] == "image":
                        positions.append((mi, bi, ci))
    if len(positions) <= keep:
        return messages

    drop = set(positions[:-keep] if keep else positions)
    out: list[dict] = []
    for mi, m in enumerate(messages):
        if not any(p[0] == mi for p in drop):
            out.append(m)
            continue
        content = []
        for bi, b in enumerate(m["content"]):
            if (mi, bi, None) in drop:
                content.append(_stub(b))
            elif b["type"] == "tool_result":
                inner = [_stub(c) if (mi, bi, ci) in drop else c
                         for ci, c in enumerate(b["content"])]
                content.append({**b, "content": inner})
            else:
                content.append(b)
        out.append({**m, "content": content, "raw": None})
    return out


def _stub(block: dict) -> dict:
    return text_block("[earlier image dropped from history to bound context]")
