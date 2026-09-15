"""DeepSeek backend — OpenAI-compatible Chat Completions at api.deepseek.com.

Reuses the vLLM translation because the wire format is the same surface, and
differs from it in three ways that all matter in practice:

1. **Reasoning arrives in its own field.** These are reasoning models: the visible
   answer is in `message.content` and the chain of thought in
   `message.reasoning_content`. The two are separate, so a turn whose whole budget
   went to reasoning comes back with an *empty* content and a normal
   `finish_reason` — which looks exactly like a model that refused to answer. It is
   captured into `thinking` for the trajectory, and never echoed back on the next
   turn: the API rejects a replayed `reasoning_content`.

2. **`max_tokens` covers the reasoning too**, so the default is raised well above
   the vLLM one. Too small a budget does not truncate the answer, it deletes it.

3. **No pricing here.** Tokens are reported and `cost_usd` stays null until list
   prices for these ids are on file.

Credentials come from `DEEPSEEK_API_KEY`.
"""
from __future__ import annotations

import os
from typing import Any

from .base import ProviderTurn, text_block
from .vllm_provider import VLLMProvider, _as_dict, _usage_dict

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"


class DeepSeekProvider(VLLMProvider):
    name = "deepseek"

    def __init__(self, model: str, *, base_url: str | None = None,
                 api_key: str | None = None, max_tokens: int = 16000, **kwargs):
        key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError(
                "no DeepSeek credentials: set DEEPSEEK_API_KEY (or pass api_key=)")
        super().__init__(
            model,
            base_url=base_url or os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
            api_key=key,
            max_tokens=max_tokens,
            **kwargs,
        )

    def _parse(self, resp: Any) -> ProviderTurn:
        choice = resp.choices[0]
        message = choice.message
        text = message.content or ""
        reasoning = getattr(message, "reasoning_content", None) or ""

        blocks = [text_block(text)] if text else []
        usage = _usage_dict(getattr(resp, "usage", None))
        self._account(usage)

        # The echoed assistant turn keeps the visible answer only. Replaying
        # reasoning_content is rejected by the API, and it is already preserved in
        # `thinking` for anyone reading the trajectory.
        raw = _as_dict(message)
        if isinstance(raw, dict):
            raw = {k: v for k, v in raw.items() if k != "reasoning_content"}

        return ProviderTurn(
            tool_calls=[],          # tagged-text protocol; no tool declarations sent
            text=text.strip(),
            blocks=blocks,
            raw=raw,
            usage=usage,
            stop_reason=getattr(choice, "finish_reason", None),
            thinking=reasoning.strip(),
        )
