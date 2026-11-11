"""Claude backend — official `anthropic` SDK.

Only this file may contain Anthropic SDK calls; the OpenAI backend lives in its
own module. The sandbox's `TOOL_SCHEMAS` are already Anthropic-shaped
(`name` / `description` / `input_schema`), so tools pass through untranslated.
"""
from __future__ import annotations

from typing import Any

import anthropic

from .base import (
    JudgeVerdict,
    Provider,
    ProviderTurn,
    text_block,
    tool_call_block,
)

# Adaptive thinking is the only supported on-mode on these; on older models it
# would be rejected, so we omit the parameter there instead of guessing a budget.
_ADAPTIVE = ("claude-opus-5", "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6",
             "claude-sonnet-5", "claude-sonnet-4-6", "claude-fable-5", "claude-mythos-5")

_JUDGE_SYSTEM = (
    "You grade chart images for a benchmark. Judge only what is visible in the image "
    "you are given; do not assume anything about the code that produced it. When a "
    "reference image is supplied, the candidate does not need to match it pixel for "
    "pixel — it needs to convey the same data and the same chart type.\n\n"
    "score is your confidence that the rubric is satisfied, from 0.0 to 1.0. passed is "
    "whether it is satisfied. detail is one or two sentences naming the specific visual "
    "evidence you based that on."
)


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, model: str = "claude-opus-5", *, effort: str = "high",
                 max_tokens: int = 16000, judge_effort: str = "high",
                 client: Any = None):
        super().__init__(model)
        self.effort = effort
        self.max_tokens = max_tokens
        self.judge_effort = judge_effort
        # A bare client resolves ANTHROPIC_API_KEY, then ANTHROPIC_AUTH_TOKEN, then
        # an `ant auth login` profile — so an unset env var is not necessarily an error.
        self.client = client if client is not None else anthropic.Anthropic()

    # -- wire translation ----------------------------------------------------
    @staticmethod
    def _img(block: dict) -> dict:
        return {"type": "image",
                "source": {"type": "base64", "media_type": block["mime"], "data": block["b64"]}}

    def _native_blocks(self, blocks: list[dict]) -> list[dict]:
        out: list[dict] = []
        for b in blocks:
            kind = b["type"]
            if kind == "text":
                out.append({"type": "text", "text": b["text"]})
            elif kind == "image":
                out.append(self._img(b))
            elif kind == "tool_call":
                out.append({"type": "tool_use", "id": b["id"], "name": b["name"], "input": b["args"]})
            elif kind == "tool_result":
                # Claude accepts image blocks inside a tool_result, so a rendered
                # chart goes back to the model in place, as the tool's own output.
                tr: dict = {"type": "tool_result", "tool_use_id": b["id"],
                            "content": self._native_blocks(b["content"])}
                if b.get("is_error"):
                    tr["is_error"] = True
                out.append(tr)
        return out

    def _to_native(self, messages: list[dict]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            if m["role"] == "assistant" and m.get("raw") is not None and m.get("provider") == self.name:
                # Echo our own assistant content verbatim: thinking blocks carry
                # signatures that must survive the round trip untouched.
                out.append({"role": "assistant", "content": m["raw"]})
            else:
                out.append({"role": m["role"], "content": self._native_blocks(m["content"])})
        return out

    # -- the one seam the dry run overrides ----------------------------------
    def _create(self, **kwargs) -> Any:
        # Stream unconditionally: it is the documented way to avoid HTTP timeouts
        # on long or high-max_tokens requests, and costs nothing when replies are short.
        with self.client.messages.stream(**kwargs) as stream:
            return stream.get_final_message()

    # -- agent loop ----------------------------------------------------------
    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> ProviderTurn:
        native = self._to_native(messages)
        # Two cache breakpoints: tools+system (stable for the whole episode, since
        # tools render before system) and the tail of the growing conversation.
        if native and isinstance(native[-1].get("content"), list) and native[-1]["content"]:
            tail = native[-1]["content"][-1]
            if isinstance(tail, dict):
                tail["cache_control"] = {"type": "ephemeral"}

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": native,
            "tools": tools,
            # One action per step keeps the trajectory, the progress curve and the
            # persistent kernel in lockstep. _pending in LLMAgent handles a batch
            # anyway, in case a model ignores this.
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
            "output_config": {"effort": self.effort},
        }
        if self.model.startswith(_ADAPTIVE):
            # display=summarized is free (thinking is billed the same either way)
            # and puts the model's reasoning in the trajectory for debugging.
            kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}

        msg = self._create(**kwargs)
        return self._parse(msg)

    def _parse(self, msg: Any) -> ProviderTurn:
        blocks: list[dict] = []
        tool_calls: list[dict] = []
        texts: list[str] = []
        thinking: list[str] = []

        for b in getattr(msg, "content", []) or []:
            kind = getattr(b, "type", None)
            if kind == "text":
                texts.append(b.text)
                blocks.append(text_block(b.text))
            elif kind == "thinking":
                got = getattr(b, "thinking", "") or ""
                if got:
                    thinking.append(got)
            elif kind == "tool_use":
                args = b.input if isinstance(b.input, dict) else dict(b.input or {})
                tool_calls.append({"id": b.id, "name": b.name, "args": args})
                blocks.append(tool_call_block(b.id, b.name, args))

        usage = _usage_dict(getattr(msg, "usage", None))
        self._account(usage)
        return ProviderTurn(
            tool_calls=tool_calls,
            text="\n".join(texts).strip(),
            blocks=blocks,
            raw=getattr(msg, "content", None),
            usage=usage,
            stop_reason=getattr(msg, "stop_reason", None),
            thinking="\n".join(thinking).strip(),
        )

    # -- judge ---------------------------------------------------------------
    def judge(self, rubric: str, images: list[dict], context: str = "") -> JudgeVerdict:
        content: list[dict] = []
        for i, img in enumerate(images):
            content.append({"type": "text", "text": img.get("label") or f"image {i + 1}"})
            content.append(self._img(img))
        content.append({"type": "text",
                        "text": (f"{context}\n\nRubric: {rubric}" if context else f"Rubric: {rubric}")})

        resp = self._judge_create(
            model=self.model,
            max_tokens=2048,
            system=_JUDGE_SYSTEM,
            messages=[{"role": "user", "content": content}],
            output_config={"effort": self.judge_effort},
            output_format=JudgeVerdict,
        )
        self._account(_usage_dict(getattr(resp, "usage", None)))
        parsed = getattr(resp, "parsed_output", None)
        if parsed is None:
            return JudgeVerdict(passed=False, score=0.0,
                                detail="judge returned no parseable verdict")
        return parsed

    def _judge_create(self, **kwargs) -> Any:
        # Structured outputs: the response is validated against JudgeVerdict, so a
        # malformed score can't silently become part of a benchmark result.
        return self.client.messages.parse(**kwargs)


def _usage_dict(usage: Any) -> dict:
    if usage is None:
        return {}
    out = {}
    for k in ("input_tokens", "output_tokens",
              "cache_read_input_tokens", "cache_creation_input_tokens"):
        v = getattr(usage, k, None)
        if isinstance(v, int):
            out[k] = v
    return out
