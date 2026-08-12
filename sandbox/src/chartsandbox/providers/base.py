"""Provider-neutral conversation format + the Provider interface.

Why a neutral format: this sandbox is one harness that has to score several model
families, so the agent loop is written once against the blocks below and each
provider translates them to its own wire format on every call. Translation is a
pure function of the history — nothing is mutated or cached across turns.

Blocks:
    {"type": "text",        "text": str}
    {"type": "image",       "mime": str, "b64": str}
    {"type": "tool_call",   "id": str, "name": str, "args": dict}
    {"type": "tool_result", "id": str, "content": [text|image blocks], "is_error": bool}

A message is::

    {"role": "user"|"assistant", "content": [blocks],
     "raw": <provider-native content | None>, "provider": <name | None>}

`raw` carries the provider's own assistant content verbatim. It exists because
model-internal blocks — Claude's thinking blocks and their signatures, OpenAI's
reasoning items — must be echoed back **unchanged** on the next turn; rebuilding
them from the neutral form would drop them, which can be rejected outright. A
provider uses `raw` only when it produced it, so a history stays replayable
across providers (the internal blocks are simply absent there).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

# --- neutral blocks ---------------------------------------------------------


def text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def image_block(mime: str, b64: str) -> dict:
    return {"type": "image", "mime": mime, "b64": b64}


def tool_call_block(id: str, name: str, args: dict) -> dict:
    return {"type": "tool_call", "id": id, "name": name, "args": args}


def tool_result_block(id: str, content: list[dict], is_error: bool = False) -> dict:
    return {"type": "tool_result", "id": id, "content": content, "is_error": is_error}


def user(content: list[dict]) -> dict:
    return {"role": "user", "content": content, "raw": None, "provider": None}


def assistant(content: list[dict], raw: Any = None, provider: str | None = None) -> dict:
    return {"role": "assistant", "content": content, "raw": raw, "provider": provider}


@dataclass
class ProviderTurn:
    """One assistant turn, normalised."""

    tool_calls: list[dict] = field(default_factory=list)  # [{id, name, args}]
    text: str = ""
    blocks: list[dict] = field(default_factory=list)      # neutral assistant blocks
    raw: Any = None                                        # provider-native content
    usage: dict = field(default_factory=dict)
    stop_reason: str | None = None
    thinking: str = ""                                     # summary only, for the trajectory


class JudgeVerdict(BaseModel):
    """Schema the judge model is constrained to. Kept flat — nested/constrained
    JSON-Schema features are not universally supported by structured outputs."""

    passed: bool
    score: float
    detail: str


class Provider:
    """One model backend. Subclasses implement `_create` and `judge`."""

    name = "base"

    def __init__(self, model: str):
        self.model = model
        self.usage: dict[str, int] = {}
        self.calls = 0

    # -- accounting ----------------------------------------------------------
    def _account(self, usage: dict) -> None:
        self.calls += 1
        for k, v in usage.items():
            if isinstance(v, int):
                self.usage[k] = self.usage.get(k, 0) + v

    @property
    def cost_usd(self) -> float | None:
        return usage_cost(self.model, self.usage)

    def stats(self) -> dict:
        return {"provider": self.name, "model": self.model, "calls": self.calls,
                "usage": dict(self.usage), "cost_usd": self.cost_usd}

    # -- interface -----------------------------------------------------------
    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> ProviderTurn:
        raise NotImplementedError

    def judge(self, rubric: str, images: list[dict], context: str = "") -> JudgeVerdict:
        raise NotImplementedError


# --- pricing ----------------------------------------------------------------
# USD per million tokens (input, output), Anthropic first-party list prices.
# Cached 2026-06-24 — re-check before quoting numbers to anyone.
# Models without an entry report tokens only and cost_usd = None.
PRICE_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-fable-5": (10.0, 50.0),
    "claude-mythos-5": (10.0, 50.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    # Sonnet 5 list price; an introductory $2/$10 runs through 2026-08-31, so a
    # bill dated before then will come in lower than this estimate.
    "claude-sonnet-5": (3.0, 15.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

_CACHE_READ_MULT = 0.1    # cached input is ~0.1x base input price
_CACHE_WRITE_MULT = 1.25  # 5-minute TTL write premium


def usage_cost(model: str, usage: dict) -> float | None:
    """Estimate USD for accumulated usage, or None if the model has no price entry."""
    price = PRICE_PER_MTOK.get(model)
    if not price:
        return None
    p_in, p_out = price
    total = (
        usage.get("input_tokens", 0) * p_in
        + usage.get("output_tokens", 0) * p_out
        + usage.get("cache_read_input_tokens", 0) * p_in * _CACHE_READ_MULT
        + usage.get("cache_creation_input_tokens", 0) * p_in * _CACHE_WRITE_MULT
    )
    return round(total / 1_000_000, 6)
