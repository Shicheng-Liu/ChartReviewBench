"""Agent interface + a scripted (simulated) agent.

The sandbox is agent-agnostic: anything implementing `Agent.act` can be driven by
the runner. Today we ship `ScriptedAgent`, which replays a fixed list of tool
calls — used to smoke-test the sandbox end to end while the real LLM API is not
yet wired up. When the API lands, implement `LLMAgent.act` (skeleton below) and
nothing else in the sandbox needs to change.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
    """Skeleton for a real tool-calling LLM agent (to be completed when API is ready).

    Wire `complete_fn` to a provider's tool-calling endpoint using
    `chartsandbox.tools.TOOL_SCHEMAS`, maintain the message history from the
    observations the runner returns, and translate the model's tool call into a
    ToolCall. The rest of the sandbox is unchanged.
    """

    def __init__(self, complete_fn=None):
        self.complete_fn = complete_fn
        self.messages: list[dict] = []

    def act(self, observation: dict) -> ToolCall:  # pragma: no cover - not yet wired
        raise NotImplementedError(
            "LLMAgent.act is not implemented yet — plug a tool-calling API here "
            "using chartsandbox.tools.TOOL_SCHEMAS."
        )
