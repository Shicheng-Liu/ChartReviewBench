"""GPT backend — official `openai` SDK, Responses API.

Kept in its own module so no file mixes two vendors' SDKs.

It targets `/v1/responses` rather than Chat Completions because current GPT models
reject the combination this harness needs: on gpt-5.4, Chat Completions returns
`Function tools with reasoning_effort are not supported ... use /v1/responses or
set reasoning_effort to 'none'`. A chart agent that can't reason is not worth
benchmarking, so the Responses API it is. The wire shapes below were verified
against the live API, not inferred.

Three differences from the Claude path are handled here rather than leaking into
the agent:

1. `function_call_output` carries a text string. When a tool returns an image, the
   output gets a text stub and the image follows as a separate user message, so
   the model still sees the pixels.
2. Tool schemas are flat (`{"type": "function", "name", "description", "parameters"}`),
   and arguments arrive as a JSON *string*.
3. Reasoning items must be echoed back verbatim to preserve the model's chain of
   thought across tool calls. We ask for `reasoning.encrypted_content` so this
   works with `store=False` — nothing about a benchmark run is retained server-side.

Model IDs are whatever you pass on the command line; this module pins none, since
OpenAI's catalogue moves independently of this repo.
"""
from __future__ import annotations

import json
from typing import Any

import openai

from .base import (
    JudgeVerdict,
    Provider,
    ProviderTurn,
    text_block,
    tool_call_block,
)

#: Left in the text-only tool output to point at the images that follow it.
IMAGE_HANDOFF = "image(s) attached in the next message"

_JUDGE_SYSTEM = (
    "You grade chart images for a benchmark. Judge only what is visible in the image "
    "you are given; do not assume anything about the code that produced it. When a "
    "reference image is supplied, the candidate does not need to match it pixel for "
    "pixel — it needs to convey the same data and the same chart type.\n\n"
    "score is your confidence that the rubric is satisfied, from 0.0 to 1.0. passed is "
    "whether it is satisfied. detail is one or two sentences naming the specific visual "
    "evidence you based that on."
)


class OpenAIProvider(Provider):
    name = "openai"

    # Effort is named on Anthropic's five-level ladder throughout this repo; OpenAI
    # takes a subset, so the top two collapse rather than 400.
    _EFFORT = {"low": "low", "medium": "medium", "high": "high",
               "xhigh": "high", "max": "high"}

    def __init__(self, model: str, *, effort: str | None = "high",
                 max_tokens: int = 16000, judge_effort: str | None = "high",
                 client: Any = None):
        super().__init__(model)
        self.effort = self._EFFORT.get(effort or "", None)
        self.max_tokens = max_tokens
        self.judge_effort = self._EFFORT.get(judge_effort or "", None)
        self.client = client if client is not None else openai.OpenAI()

    # -- wire translation ----------------------------------------------------
    @staticmethod
    def _img(block: dict) -> dict:
        return {"type": "input_image",
                "image_url": f"data:{block['mime']};base64,{block['b64']}"}

    @staticmethod
    def _tools(tools: list[dict]) -> list[dict]:
        return [{"type": "function", "name": t["name"], "description": t["description"],
                 "parameters": t["input_schema"]}
                for t in tools]

    def _input_content(self, blocks: list[dict]) -> list[dict]:
        out: list[dict] = []
        for b in blocks:
            if b["type"] == "text":
                out.append({"type": "input_text", "text": b["text"]})
            elif b["type"] == "image":
                out.append(self._img(b))
        return out

    def _to_native(self, messages: list[dict]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            if m["role"] == "assistant" and m.get("raw") is not None and m.get("provider") == self.name:
                out.extend(m["raw"])  # our own output items, reasoning included
                continue

            if m["role"] == "assistant":
                # Cross-provider replay: rebuild what we can from neutral blocks.
                for b in m["content"]:
                    if b["type"] == "text" and b["text"]:
                        out.append({"role": "assistant",
                                    "content": [{"type": "output_text", "text": b["text"]}]})
                    elif b["type"] == "tool_call":
                        out.append({"type": "function_call", "call_id": b["id"],
                                    "name": b["name"], "arguments": json.dumps(b["args"])})
                continue

            plain: list[dict] = []
            trailing_images: list[dict] = []
            for b in m["content"]:
                if b["type"] != "tool_result":
                    plain.append(b)
                    continue
                texts = [c["text"] for c in b["content"] if c["type"] == "text"]
                images = [c for c in b["content"] if c["type"] == "image"]
                if images:
                    texts.append(f"({len(images)} {IMAGE_HANDOFF})")
                out.append({"type": "function_call_output", "call_id": b["id"],
                            "output": "\n".join(texts) or "(no output)"})
                trailing_images.extend(images)
            if plain:
                out.append({"role": "user", "content": self._input_content(plain)})
            if trailing_images:
                out.append({"role": "user", "content": self._input_content(trailing_images)})
        return out

    @staticmethod
    def _echoable(item: Any) -> dict:
        """One output item, stripped of fields the input side rejects."""
        d = item.model_dump(exclude_none=True) if hasattr(item, "model_dump") else dict(item)
        d.pop("status", None)  # output-only: 'Unknown parameter: input[N].status'
        return d

    # -- the one seam the dry run overrides ----------------------------------
    def _create(self, **kwargs) -> Any:
        return self.client.responses.create(**kwargs)

    # -- agent loop ----------------------------------------------------------
    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> ProviderTurn:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "instructions": system,
            "input": self._to_native(messages),
            "tools": self._tools(tools),
            "parallel_tool_calls": False,
            "max_output_tokens": self.max_tokens,
            "store": False,             # a benchmark run is not the vendor's to keep
            "include": ["reasoning.encrypted_content"],
        }
        if self.effort:
            # summary=auto costs nothing extra and puts the model's reasoning in
            # the trajectory, which is most of what you want when a run goes wrong.
            kwargs["reasoning"] = {"effort": self.effort, "summary": "auto"}
        return self._parse(self._create(**kwargs))

    def _parse(self, resp: Any) -> ProviderTurn:
        blocks: list[dict] = []
        tool_calls: list[dict] = []
        texts: list[str] = []
        thinking: list[str] = []

        for item in getattr(resp, "output", []) or []:
            kind = getattr(item, "type", None)
            if kind == "function_call":
                try:
                    args = json.loads(item.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                if not isinstance(args, dict):
                    args = {}
                # call_id (not id) is what function_call_output must reference.
                tool_calls.append({"id": item.call_id, "name": item.name, "args": args})
                blocks.append(tool_call_block(item.call_id, item.name, args))
            elif kind == "message":
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", None) == "output_text":
                        texts.append(c.text)
                        blocks.append(text_block(c.text))
            elif kind == "reasoning":
                for s in getattr(item, "summary", []) or []:
                    got = getattr(s, "text", "") or ""
                    if got:
                        thinking.append(got)

        usage = _usage_dict(getattr(resp, "usage", None))
        self._account(usage)
        return ProviderTurn(
            tool_calls=tool_calls,
            text="\n".join(texts).strip(),
            blocks=blocks,
            raw=[self._echoable(i) for i in getattr(resp, "output", []) or []],
            usage=usage,
            stop_reason=getattr(resp, "status", None),
            thinking="\n".join(thinking).strip(),
        )

    # -- judge ---------------------------------------------------------------
    def judge(self, rubric: str, images: list[dict], context: str = "") -> JudgeVerdict:
        content: list[dict] = []
        for i, img in enumerate(images):
            content.append({"type": "input_text", "text": img.get("label") or f"image {i + 1}"})
            content.append(self._img(img))
        content.append({"type": "input_text",
                        "text": (f"{context}\n\nRubric: {rubric}" if context else f"Rubric: {rubric}")})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "instructions": _JUDGE_SYSTEM,
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": self.max_tokens,
            "store": False,
            "text_format": JudgeVerdict,
        }
        if self.judge_effort:
            kwargs["reasoning"] = {"effort": self.judge_effort}

        resp = self._judge_create(**kwargs)
        self._account(_usage_dict(getattr(resp, "usage", None)))
        parsed = getattr(resp, "output_parsed", None)
        if parsed is None:
            return JudgeVerdict(passed=False, score=0.0,
                                detail="judge returned no parseable verdict")
        return parsed

    def _judge_create(self, **kwargs) -> Any:
        # Structured outputs: validated against JudgeVerdict, so a malformed score
        # can't silently become part of a benchmark result.
        return self.client.responses.parse(**kwargs)


def _usage_dict(usage: Any) -> dict:
    if usage is None:
        return {}
    out: dict[str, int] = {}
    for src, dst in (("input_tokens", "input_tokens"), ("output_tokens", "output_tokens")):
        v = getattr(usage, src, None)
        if isinstance(v, int):
            out[dst] = v
    cached = getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", None)
    if isinstance(cached, int) and cached:
        # input_tokens includes the cached span; split them so cost maths can
        # price the cheap part separately, as the Anthropic path does.
        out["cache_read_input_tokens"] = cached
        out["input_tokens"] = max(0, out.get("input_tokens", 0) - cached)
    return out
