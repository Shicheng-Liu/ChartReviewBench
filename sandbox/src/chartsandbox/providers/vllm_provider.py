"""Self-hosted open-weight backend — vLLM's OpenAI-compatible server.

Reuses the `openai` SDK pointed at your own endpoint, but deliberately targets
**Chat Completions**, not the Responses API: vLLM's `/v1/chat/completions` is the
surface that has been stable and fully featured across releases.

Differences from the hosted backends, handled here:

1. No `reasoning_effort`. Depth is a property of the model you loaded, so `--effort`
   is ignored on this backend rather than sent and rejected.
2. `temperature` defaults to 0 — a self-hosted benchmark should be reproducible, and
   unlike the frontier APIs vLLM still accepts the parameter.
3. Structured output comes from guided decoding. We try `response_format` with a JSON
   schema and fall back to vLLM's older `guided_json` extra-body field, so the judge
   works across server versions.
4. No pricing: tokens are reported, `cost_usd` stays null. Your cost is GPU time,
   which this harness has no way to observe.
5. A `role: "tool"` message is text-only here too, so an image returned by a tool
   follows as a separate user message.

The server must be started with tool calling enabled, e.g.:

    vllm serve Qwen/Qwen2.5-VL-72B-Instruct \\
        --enable-auto-tool-choice --tool-call-parser hermes \\
        --limit-mm-per-prompt image=8

Point the harness at it with `--base-url` or `VLLM_BASE_URL`. If the loaded model
cannot emit tool calls at all, episodes end early via the agent's no-tool-call
guard — that is a real capability result, not a harness failure.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

import openai

from .base import (
    JudgeVerdict,
    Provider,
    ProviderTurn,
    text_block,
    tool_call_block,
)

#: Left in the text-only tool message to point at the images that follow it.
IMAGE_HANDOFF = "image(s) attached in the next message"

DEFAULT_BASE_URL = "http://localhost:8000/v1"

_JUDGE_SYSTEM = (
    "You grade chart images for a benchmark. Judge only what is visible in the image "
    "you are given; do not assume anything about the code that produced it. When a "
    "reference image is supplied, the candidate does not need to match it pixel for "
    "pixel — it needs to convey the same data and the same chart type.\n\n"
    "Reply with JSON only: passed (boolean, is the rubric satisfied), score (number "
    "from 0.0 to 1.0 for how fully it is satisfied — 1.0 holds completely, 0.0 does "
    "not hold at all, in between holds in part; this is not your confidence in your "
    "own verdict, so a rubric you are certain is violated scores near 0.0), detail "
    "(one or two sentences naming the specific visual evidence you based that on)."
)


def _judge_schema() -> dict:
    schema = JudgeVerdict.model_json_schema()
    schema["additionalProperties"] = False  # required by strict guided decoding
    return schema


class VLLMProvider(Provider):
    name = "vllm"

    def __init__(self, model: str, *, base_url: str | None = None, api_key: str | None = None,
                 max_tokens: int = 4096, temperature: float = 0.0,
                 effort: str | None = None, judge_effort: str | None = None,
                 client: Any = None):
        super().__init__(model)
        # effort/judge_effort accepted and ignored so the CLI can stay uniform.
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.base_url = base_url or os.environ.get("VLLM_BASE_URL", DEFAULT_BASE_URL)
        self.client = client if client is not None else openai.OpenAI(
            base_url=self.base_url,
            # vLLM ignores the key unless started with --api-key; the SDK still
            # insists on a non-empty one.
            api_key=api_key or os.environ.get("VLLM_API_KEY", "EMPTY"),
        )

    def stats(self) -> dict:
        return {**super().stats(), "base_url": self.base_url}

    # -- wire translation ----------------------------------------------------
    @staticmethod
    def _img(block: dict) -> dict:
        return {"type": "image_url",
                "image_url": {"url": f"data:{block['mime']};base64,{block['b64']}"}}

    @staticmethod
    def _tools(tools: list[dict]) -> list[dict]:
        return [{"type": "function",
                 "function": {"name": t["name"], "description": t["description"],
                              "parameters": t["input_schema"]}}
                for t in tools]

    def _content(self, blocks: list[dict]) -> list[dict]:
        out: list[dict] = []
        for b in blocks:
            if b["type"] == "text":
                out.append({"type": "text", "text": b["text"]})
            elif b["type"] == "image":
                out.append(self._img(b))
        return out

    def _to_native(self, messages: list[dict]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            if m["role"] == "assistant" and m.get("raw") is not None and m.get("provider") == self.name:
                out.append(m["raw"])
                continue

            if m["role"] == "assistant":
                text = " ".join(b["text"] for b in m["content"] if b["type"] == "text")
                calls = [{"id": b["id"], "type": "function",
                          "function": {"name": b["name"], "arguments": json.dumps(b["args"])}}
                         for b in m["content"] if b["type"] == "tool_call"]
                msg: dict = {"role": "assistant", "content": text or None}
                if calls:
                    msg["tool_calls"] = calls
                out.append(msg)
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
                out.append({"role": "tool", "tool_call_id": b["id"],
                            "content": "\n".join(texts) or "(no output)"})
                trailing_images.extend(images)
            if plain:
                out.append({"role": "user", "content": self._content(plain)})
            if trailing_images:
                out.append({"role": "user", "content": self._content(trailing_images)})
        return out

    # -- the one seam the dry run overrides ----------------------------------
    def _create(self, **kwargs) -> Any:
        try:
            return self.client.chat.completions.create(**kwargs)
        except openai.APIConnectionError as exc:
            # The default SDK message is just "Connection error." — say which
            # endpoint we tried, since that is the whole diagnosis.
            raise RuntimeError(
                f"no vLLM server answering at {self.base_url}. Start one, e.g.\n"
                f"    vllm serve {self.model} --enable-auto-tool-choice "
                f"--tool-call-parser hermes\n"
                f"or point --base-url (or $VLLM_BASE_URL) at the right endpoint."
            ) from exc

    # -- agent loop ----------------------------------------------------------
    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> ProviderTurn:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *self._to_native(messages)],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        # No tools under the tagged-text protocol — which is the point of that
        # protocol here: the server no longer needs --enable-auto-tool-choice, so
        # a model whose tool-call parser is weak or missing is still evaluable.
        if tools:
            kwargs["tools"] = self._tools(tools)
            kwargs["tool_choice"] = "auto"
        return self._parse(self._create(**kwargs))

    def _parse(self, resp: Any) -> ProviderTurn:
        choice = resp.choices[0]
        message = choice.message
        text = message.content or ""

        blocks: list[dict] = []
        if text:
            blocks.append(text_block(text))
        tool_calls: list[dict] = []
        for i, call in enumerate(getattr(message, "tool_calls", None) or []):
            try:
                args = json.loads(call.function.arguments or "{}")
            except (json.JSONDecodeError, TypeError):
                args = {}
            if not isinstance(args, dict):
                args = {}
            # Some tool-call parsers omit the id; synthesise a stable one so the
            # tool_result can still be matched back to its call.
            call_id = getattr(call, "id", None) or f"call_{self.calls}_{i}"
            tool_calls.append({"id": call_id, "name": call.function.name, "args": args})
            blocks.append(tool_call_block(call_id, call.function.name, args))

        usage = _usage_dict(getattr(resp, "usage", None))
        self._account(usage)
        return ProviderTurn(
            tool_calls=tool_calls,
            text=text.strip(),
            blocks=blocks,
            raw=_as_dict(message),
            usage=usage,
            stop_reason=getattr(choice, "finish_reason", None),
        )

    # -- judge ---------------------------------------------------------------
    def judge(self, rubric: str, images: list[dict], context: str = "") -> JudgeVerdict:
        content: list[dict] = []
        for i, img in enumerate(images):
            content.append({"type": "text", "text": img.get("label") or f"image {i + 1}"})
            content.append(self._img(img))
        content.append({"type": "text",
                        "text": (f"{context}\n\nRubric: {rubric}" if context else f"Rubric: {rubric}")})

        messages = [{"role": "system", "content": _JUDGE_SYSTEM},
                    {"role": "user", "content": content}]
        base = {"model": self.model, "messages": messages,
                "max_tokens": self.max_tokens, "temperature": self.temperature}
        schema = _judge_schema()

        try:
            resp = self._create(**base, response_format={
                "type": "json_schema",
                "json_schema": {"name": "verdict", "schema": schema, "strict": True},
            })
        except openai.BadRequestError:
            # Older vLLM builds take guided decoding as an extra body field instead.
            resp = self._create(**base, extra_body={"guided_json": schema})

        self._account(_usage_dict(getattr(resp, "usage", None)))
        return _parse_verdict(resp.choices[0].message.content or "")


def _parse_verdict(raw: str) -> JudgeVerdict:
    """Guided decoding should make this exact; keep a salvage path for the case
    where a model wraps the JSON in prose or a code fence anyway."""
    for candidate in (raw, *_json_spans(raw)):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            try:
                return JudgeVerdict(**data)
            except Exception:  # noqa: BLE001 - wrong keys / types; try the next span
                continue
    return JudgeVerdict(passed=False, score=0.0,
                        detail=f"judge returned unparseable output: {raw[:200]!r}")


def _json_spans(raw: str) -> list[str]:
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.S)
    braces = re.findall(r"\{.*\}", raw, re.S)
    return [*fenced, *braces]


def _as_dict(message: Any) -> Any:
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(message, attr, None)
        if callable(fn):
            try:
                return fn(exclude_none=True) if attr == "model_dump" else fn()
            except TypeError:
                return fn()
    return message


def _sub(obj: Any, *path: str) -> Any:
    """Walk attributes or dict keys, whichever this SDK object happens to use."""
    for key in path:
        if obj is None:
            return None
        obj = getattr(obj, key, None) if not isinstance(obj, dict) else obj.get(key)
    return obj


def _usage_dict(usage: Any) -> dict:
    """Token counts, including the two spans that are priced differently.

    `prompt_tokens`/`completion_tokens` alone look complete and are not: reasoning
    tokens are billed as output but arrive nested under a details object, and
    cache-hit input is billed at a fraction of the normal rate. Dropping either
    does not lose the run, it loses the ability to cost it afterwards — so both are
    carried alongside the totals they are part of.
    """
    if usage is None:
        return {}
    out: dict[str, int] = {}
    if isinstance(_sub(usage, "prompt_tokens"), int):
        out["input_tokens"] = _sub(usage, "prompt_tokens")
    if isinstance(_sub(usage, "completion_tokens"), int):
        out["output_tokens"] = _sub(usage, "completion_tokens")

    reasoning = _sub(usage, "completion_tokens_details", "reasoning_tokens")
    if isinstance(reasoning, int) and reasoning:
        out["reasoning_tokens"] = reasoning

    # DeepSeek spells the cached span twice; either is the same number.
    cached = _sub(usage, "prompt_tokens_details", "cached_tokens")
    if not isinstance(cached, int) or not cached:
        cached = _sub(usage, "prompt_cache_hit_tokens")
    if isinstance(cached, int) and cached:
        out["cache_read_input_tokens"] = cached
    return out
