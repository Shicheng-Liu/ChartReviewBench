"""Agent interface, a scripted agent, and two real multimodal LLM agents.

The sandbox is agent-agnostic: anything implementing `Agent.act` can be driven by
the runner. `ScriptedAgent` replays a fixed trajectory and is how we smoke-test
the sandbox itself.

Two response protocols put a real model in the loop, both through the same `act`
interface and the same `providers/` wire translation:

* `LLMAgent` — **tool calling**. One tool call per step, the model chooses when to
  look at an image and when to call `finish`.
* `TaggedAgent` — **tagged text** (`<reasoning>`/`<code>`/`<decision>`, see
  `protocol.py`). One iteration per step: code runs, and the images it wrote are
  fed back automatically, so inspection is not something the model can skip. Needs
  no tool-calling support from the backend, which is what makes weaker open-weight
  models evaluable at all.

Both report `stop_reason` in `stats`, so why an episode ended is comparable across
protocols.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .protocol import STOP, parse_tagged
from .providers import Provider, image_block, text_block, tool_result_block, user
from .tools import VIEWABLE_MIME, TOOL_SCHEMAS

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
        self.stop_reason: str | None = None

    # -- lifecycle -----------------------------------------------------------
    def reset(self, task, sandbox) -> None:  # noqa: ANN001
        self.messages = []
        self.transcript = []
        self._pending = []
        self._results = []
        self._inflight = None
        self._nudges = 0
        self.stop_reason = None

    @property
    def stats(self) -> dict:
        return {**self.provider.stats(), "protocol": "tools",
                "stop_reason": self.stop_reason, "transcript": self.transcript}

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

        self.stop_reason = "no_tool_call"
        return ToolCall("finish", {"message": (
            f"no tool call after {self.max_nudges} prompts; giving up"
        )})

    def _dispatch(self, call: dict) -> ToolCall:
        self._inflight = call
        if call["name"] == "finish":
            self.stop_reason = "agent_stop"
        return ToolCall(call["name"], call["args"])

    def _history(self) -> list[dict]:
        if self.max_history_images <= 0:
            return self.messages
        return _prune_images(self.messages, self.max_history_images)


TAGGED_SYSTEM_PROMPT = """\
You are an autonomous data-visualization agent working in a sandboxed Python \
workspace. You work in iterations: on each one you reason about the current state \
of the chart, write Python that changes it, and decide whether another iteration is \
needed.

Reply with exactly this structure, and nothing outside it:

<reasoning>
analysis of the current chart, the issues you identified, and the fixes you plan
</reasoning>
<code>
# Python/Matplotlib code
</code>
<decision>continue</decision>

<decision> is the only thing that ends the episode early, and it takes one of two \
values:

  continue — the environment executes your code, returns the result, and you get \
another iteration.
  stop — the episode ends. The latest chart that rendered successfully is what gets \
evaluated, so code in this same turn still runs before it closes.

The code runs in a persistent Python kernel: variables, imports and open figures \
survive across iterations, so you can build on earlier state instead of re-running \
everything. It is your only action — there are no tools. Read and write workspace \
files with ordinary Python (open, pathlib, pandas). Only files inside the workspace \
exist, and there is no network access.

Save every chart to a file (for example plt.savefig('output.png')). A figure you \
never write is one that nothing can evaluate. After each iteration you are shown \
stdout, stderr, any traceback, any warnings, and the image files your code wrote — \
so you see your own chart.

Your submission is judged on all of the following at once, so a change that buys one \
at the cost of another is not an improvement:

  - it runs without errors, and without warnings that signal a real defect;
  - the chart is faithful to the data table and to what the summary says it shows;
  - the things that were actually wrong are corrected;
  - everything that was already correct is untouched — do not restyle, retitle, \
resize or otherwise improve anything you were not asked to fix;
  - the chart is readable: no overlapping or clipped text, no legend sitting on top \
of the data, nothing too small or too faint to read;
  - you get there in as few iterations as you can.

Check your work two ways, because neither covers the other. Look at the rendered \
image for anything visual — layout, overlap, legibility, whether the chart reads the \
way it should. Query the figure object for anything exact — ax.get_ylim(), \
ax.get_legend_handles_labels(), line.get_color(), ax.get_title() — because a value \
you read off a picture is a guess, while a value you read off the figure is the one \
that will actually be checked.

When you stop, put in <reasoning> what you changed and what you verified. If \
something is unfinished, say plainly what and why."""


#: `max_turns` was not overridden — defer to the task. Distinct from `None`, which
#: is an explicit request for no iteration cap.
_UNSET = object()


class TaggedAgent(Agent):
    """A real multimodal agent that acts through tagged text instead of tool calls.

    One `act` is one iteration of edit-execute-inspect-revise: the model emits
    `<reasoning>/<code>/<decision>`, the code runs in the persistent kernel, and the
    images it wrote come back automatically as image blocks on the next turn.

    Two things differ from `LLMAgent` beyond the wire format, and both are
    deliberate. Inspection is not optional — a tool-calling agent can decline to
    call `view_image` and grade its own work blind, whereas here the rendered chart
    is always in front of the model. And stopping is the agent's alone: the
    environment ends an episode when the agent says `stop`, when the iteration
    budget runs out, or when the model cannot produce a parseable action — never
    because a verifier passed. Which of those happened is recorded in `stop_reason`,
    which is what makes a wrong stopping decision analysable after the fact.
    """

    #: Sent back when a turn carries no runnable action. Costs wall-clock time but
    #: no iteration budget, mirroring the no-tool-call nudge in `LLMAgent`.
    _FORMAT_REMINDER = (
        "That turn carried no action, so nothing ran. Reply with a <code> block to "
        "execute, or <decision>stop</decision> if the task is complete or you are "
        "blocked. Use exactly:\n"
        "<reasoning>...</reasoning>\n<code>\n# python\n</code>\n<decision>continue</decision>"
    )

    def __init__(self, provider: Provider, *, system: str = TAGGED_SYSTEM_PROMPT,
                 max_turns=_UNSET, max_history_images: int = 0,
                 max_nudges: int = 2, max_images_per_turn: int = 4):
        self.provider = provider
        self.system = system
        # Unset defers to the task's own budget; an int overrides it and `None` asks
        # for no cap at all, which is how both the turn-count ablation and the
        # uncapped release protocol are driven from the CLI.
        self.max_turns_override = max_turns
        self.max_turns: int | None = max_turns if isinstance(max_turns, int) else 10
        # Set only when the task's own step budget forced the iteration cap down.
        self.max_turns_requested: int | None = None
        self.max_history_images = max_history_images
        self.max_nudges = max_nudges
        self.max_images_per_turn = max_images_per_turn
        self.sandbox = None
        self.messages: list[dict] = []
        self.transcript: list[dict] = []
        self.decisions: list[str] = []
        self.format_warnings: list[str] = []
        self.parse_failures = 0
        self.turn = 0
        self.stop_reason: str | None = None
        self._stop_after_run = False

    # -- lifecycle -----------------------------------------------------------
    def reset(self, task, sandbox) -> None:  # noqa: ANN001
        self.sandbox = sandbox
        self.max_turns_requested = None
        self.max_turns = self._reconcile_budget(task)
        self.messages = []
        self.transcript = []
        self.decisions = []
        self.format_warnings = []
        self.parse_failures = 0
        self.turn = 0
        self.stop_reason = None
        self._stop_after_run = False

    def _reconcile_budget(self, task) -> int | None:  # noqa: ANN001
        """The iteration cap actually in force, given what the task's steps allow.

        `None` is uncapped at either end, which the released A/B/C suite selects on
        purpose: with no ceiling, the agent's own `stop` is the only thing that ends
        an episode, which is the behaviour under test.

        Where both are finite they have to be reconciled, because an iteration costs
        a step and closing the episode costs one more. A task with fewer steps than
        the requested iterations would have the runner cut the agent off mid-budget,
        and that reads in the trajectory exactly like a stop the agent never chose.
        Clamping here and recording what was asked for keeps the two apart — an
        interaction-budget sweep at 1, 2, 3, 5, 10 steps is a normal thing to run,
        not a malformed task.
        """
        wanted = self.max_turns_override
        if wanted is _UNSET:
            wanted = getattr(task, "max_turns", 10)
        steps = getattr(task, "step_budget", None)
        if steps is None:
            return wanted
        allowed = max(1, steps - 1)
        if wanted is None or wanted > allowed:
            self.max_turns_requested = wanted
            return allowed
        return wanted

    @property
    def stats(self) -> dict:
        return {**self.provider.stats(), "protocol": "xml",
                "turns_taken": self.turn, "max_turns": self.max_turns,
                "max_turns_requested": self.max_turns_requested,
                "stop_reason": self.stop_reason, "decisions": self.decisions,
                "parse_failures": self.parse_failures,
                "format_warnings": self.format_warnings,
                "transcript": self.transcript}

    def action_brief(self, task) -> str:  # noqa: ANN001
        """Replaces the tool listing in the opening observation (see `runner`)."""
        return (
            "## How to act\n"
            "You have no tools. Each turn, emit one <reasoning> block, one <code> "
            "block of Python to run in the persistent workspace kernel, and one "
            "<decision> of continue or stop. The images your code writes are shown "
            "back to you automatically.\n\n"
            + (f"Iteration budget: {self.max_turns}. The episode ends when you answer "
               "<decision>stop</decision> or the budget runs out."
               if self.max_turns is not None else
               "There is no iteration cap. The episode ends when you answer "
               "<decision>stop</decision>, so decide for yourself when the chart is right.")
        )

    # -- the loop ------------------------------------------------------------
    def act(self, observation: dict) -> ToolCall:
        if "_tool" not in observation:
            self.messages.append(user([text_block(observation.get("text", ""))]))
        else:
            self.messages.append(user(self._feedback(observation)))

        # A stop that arrived with code attached: the code has now run, and its
        # output is in history for the record even though nothing will read it.
        if self._stop_after_run:
            return self._finish("agent_stop", self._last_reasoning())

        if self.max_turns is not None and self.turn >= self.max_turns:
            return self._finish(
                "turn_budget_exhausted",
                f"iteration budget ({self.max_turns}) exhausted before the agent stopped",
            )

        for _ in range(self.max_nudges + 1):
            turn = self.provider.complete(self.system, self._history(), [])
            self.messages.append({"role": "assistant", "content": turn.blocks,
                                  "raw": turn.raw, "provider": self.provider.name})
            parsed = parse_tagged(turn.text)
            self.format_warnings.extend(parsed.warnings)
            self.transcript.append({
                "text": turn.text, "thinking": turn.thinking,
                "stop_reason": turn.stop_reason, "reasoning": parsed.reasoning,
                "decision": parsed.decision, "has_code": bool(parsed.code),
                "warnings": parsed.warnings,
            })

            if parsed.actionable:
                self.decisions.append(parsed.decision)
                if not (parsed.code and parsed.code.strip()):
                    return self._finish("agent_stop", parsed.reasoning)
                self.turn += 1
                # Run the code first either way: a final turn's chart is what the
                # verifiers will grade, so discarding it on `stop` would score the
                # agent on work it did not submit.
                self._stop_after_run = parsed.decision == STOP
                return ToolCall("execute_python", {"code": parsed.code})

            self.parse_failures += 1
            detail = ("\n\nProblems with that turn: " + "; ".join(parsed.warnings)
                      if parsed.warnings else "")
            self.messages.append(user([text_block(self._FORMAT_REMINDER + detail)]))

        return self._finish(
            "parse_failure",
            f"no parseable <code> or <decision> after {self.max_nudges} reminders; giving up",
        )

    # -- helpers -------------------------------------------------------------
    def _finish(self, reason: str, message: str) -> ToolCall:
        self.stop_reason = reason
        return ToolCall("finish", {"message": (message or reason)[:2000]})

    def _last_reasoning(self) -> str:
        return next((t["reasoning"] for t in reversed(self.transcript) if t.get("reasoning")), "")

    def _feedback(self, obs: dict) -> list[dict]:
        """Execution result + the images it produced + where we are in the budget."""
        blocks = list(_observation_blocks(obs))
        blocks.extend(self._rendered_images(obs))
        if self.max_turns is None:
            blocks.append(text_block(
                f"[iteration {self.turn}; no iteration cap — the episode runs until you "
                f"answer <decision>stop</decision>]"))
            return blocks
        remaining = self.max_turns - self.turn
        if remaining == 1:
            blocks.append(text_block(
                f"[iteration {self.turn}/{self.max_turns} — this is the last one; "
                f"the episode ends after it whatever you decide]"))
        elif remaining > 1:
            blocks.append(text_block(
                f"[iteration {self.turn}/{self.max_turns}, {remaining} left]"))
        return blocks

    def _rendered_images(self, obs: dict) -> list[dict]:
        """Attach the images this iteration wrote, so inspection cannot be skipped.

        The kernel also counts .svg/.pdf as images it wrote, but no provider takes
        those as image blocks — they are named in the execution summary instead, and
        the agent can rasterise one itself if it wants to look at it.
        """
        if self.sandbox is None:
            return []
        # A figure left open but never saved still gets shown, so a forgotten
        # savefig costs the agent its deliverable but not its eyes. Only on a clean
        # run, though: after a traceback the figure is in an indeterminate,
        # half-drawn state, and the protocol says a failed turn returns error info
        # rather than a chart. Showing that partial render would invite the agent
        # to trust a chart that never finished drawing.
        produced = list(obs.get("images_created") or [])
        if not produced and not obs.get("error"):
            produced = list(obs.get("figures_captured") or [])
        paths = [p for p in produced
                 if p.rsplit(".", 1)[-1].lower() in VIEWABLE_MIME][: self.max_images_per_turn]
        blocks: list[dict] = []
        for path in paths:
            try:
                r = self.sandbox.view_image(path)
            except Exception:  # a file deleted between the snapshot and now
                continue
            if r.get("ok") and r.get("base64"):
                blocks.append(text_block(f"{path} — {r.get('width')}x{r.get('height')}"))
                blocks.append(image_block(r.get("mime") or "image/png", r["base64"]))
        return blocks

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
        if obs.get("warnings"):
            # A warning is frequently the only trace of a rendering defect that
            # neither an exception nor a figure property shows: a missing glyph
            # draws a box, a failed tight_layout crops a label.
            parts.append("warnings:\n" + _clip("\n".join(obs["warnings"]), _MAX_STREAM))
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
