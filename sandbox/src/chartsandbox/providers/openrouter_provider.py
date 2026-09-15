"""OpenRouter Chat Completions, shared by the XML/tool agent and image judge.

See https://openrouter.ai/docs/quickstart and the reasoning/structured-output guides.
Reasoning details are replayed intact; token counts include reasoning and caching.
"""
from __future__ import annotations

import os
from typing import Any

import openai

from .vllm_provider import (
    VLLMProvider, _JUDGE_SYSTEM, _judge_schema, _parse_verdict, _as_dict, _sub,
    _usage_dict,
)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider(VLLMProvider):
    name = "openrouter"

    def __init__(self, model: str, *, api_key: str | None = None,
                 base_url: str | None = None, max_tokens: int = 16000,
                 effort: str | None = None, judge_effort: str | None = None,
                 client: Any = None, **kwargs):
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key and client is None:
            raise RuntimeError("set OPENROUTER_API_KEY before using openrouter:<model-id>")
        endpoint = base_url or os.environ.get("OPENROUTER_BASE_URL", DEFAULT_BASE_URL)
        if client is None:
            client = openai.OpenAI(api_key=key, base_url=endpoint, timeout=120, max_retries=2)
        super().__init__(model, client=client, base_url=endpoint,
                         max_tokens=max_tokens, **kwargs)
        self.effort = effort
        self.judge_effort = judge_effort
        self.responses: list[dict] = []

    def _create(self, **kwargs):
        try:
            response = self.client.chat.completions.create(**kwargs)
        except openai.APIConnectionError as exc:
            raise RuntimeError(f"OpenRouter connection failed at {self.base_url}") from exc
        if not getattr(response, "choices", None):
            raise RuntimeError("OpenRouter returned no completion choices")
        self.responses.append({
            "id": getattr(response, "id", None),
            "model": getattr(response, "model", None),
            "provider": getattr(response, "provider", None),
            "usage": _as_dict(getattr(response, "usage", None)),
        })
        return response

    def _options(self, effort):
        # Reject providers that cannot honor tools/schema/reasoning, rather than
        # silently dropping requested benchmark parameters. No model fallbacks.
        extra = {"provider": {"require_parameters": True}}
        if effort is not None:
            extra["reasoning"] = {"effort": effort}
        return extra

    def complete(self, system, messages, tools):
        kwargs = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *self._to_native(messages)],
            "max_tokens": self.max_tokens,
            "extra_body": self._options(self.effort),
        }
        if tools:
            kwargs.update(tools=self._tools(tools), tool_choice="auto", parallel_tool_calls=False)
        return self._parse(self._create(**kwargs))

    def _parse(self, response):
        turn = super()._parse(response)
        message = response.choices[0].message
        turn.thinking = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None) or ""
        # Retain signed/encrypted reasoning exactly as returned, but do not echo
        # response-only fields such as annotations or refusal metadata.
        raw = _as_dict(message)
        turn.raw = {k: v for k, v in raw.items() if k in {
            "role", "content", "tool_calls", "reasoning", "reasoning_details",
            "reasoning_content",
        }}
        return turn

    def _account(self, usage):
        if self.responses:
            cache_write = _sub(self.responses[-1]["usage"], "prompt_tokens_details", "cache_write_tokens")
            if isinstance(cache_write, int):
                usage["cache_write_input_tokens"] = cache_write
        super()._account(usage)

    def stats(self):
        return {**super().stats(), "responses": self.responses,
                "reasoning_effort": self.effort}

    def episode_stats(self, mark):
        return {**super().episode_stats(mark), "base_url": self.base_url,
                "responses": self.responses[mark:], "reasoning_effort": self.judge_effort}

    def judge(self, rubric, images, context=""):
        content = []
        for i, img in enumerate(images):
            content.extend([{"type": "text", "text": img.get("label") or f"image {i + 1}"}, self._img(img)])
        content.append({"type": "text", "text": f"{context}\n\nRubric: {rubric}"})
        response = self._create(
            model=self.model,
            messages=[{"role": "system", "content": _JUDGE_SYSTEM}, {"role": "user", "content": content}],
            max_tokens=self.max_tokens,
            extra_body=self._options(self.judge_effort),
            response_format={"type": "json_schema", "json_schema": {
                "name": "verdict", "schema": _judge_schema(), "strict": True}},
        )
        # No vLLM-specific guided_json fallback: unsupported schema is an explicit
        # infrastructure error; choose a model supporting image input + schema.
        self._account(_usage_dict(getattr(response, "usage", None)))
        return _parse_verdict(response.choices[0].message.content or "")
