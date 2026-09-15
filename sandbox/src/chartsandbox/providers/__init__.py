"""Model backends. `get_provider("<model>")` is the only entry point you need.

Routing is by model-ID prefix, so `--model claude-opus-5` and `--model gpt-5.4` both
just work. Self-hosted weights have arbitrary names, so name the backend explicitly:
`vllm:Qwen/Qwen2.5-VL-72B-Instruct`. The same form forces a backend for any hosted ID
the prefixes don't cover — `anthropic:<id>`, `openai:<id>`.

SDK imports are lazy — you only need the SDK for the backend you actually use.
"""
from __future__ import annotations

from .base import (
    JudgeVerdict,
    PRICE_PER_MTOK,
    Provider,
    ProviderTurn,
    assistant,
    image_block,
    text_block,
    token_summary,
    tool_call_block,
    tool_result_block,
    usage_cost,
    user,
)

DEFAULT_MODEL = "claude-opus-5"

_ANTHROPIC_PREFIXES = ("claude", "fable", "mythos")
_OPENAI_PREFIXES = ("gpt", "chatgpt", "o1", "o3", "o4")
_DEEPSEEK_PREFIXES = ("deepseek",)


BACKENDS = ("anthropic", "openai", "vllm", "deepseek", "openrouter")


def resolve(model: str) -> tuple[str, str]:
    """('anthropic'|'openai'|'vllm', model_id) — raises if it can't be inferred.

    Split on the *first* colon only: `vllm:Qwen/Qwen2.5-VL-72B-Instruct` keeps its
    slashes, and a bare HuggingFace repo id never looks like a backend prefix.
    """
    if ":" in model:
        backend, _, ident = model.partition(":")
        backend = backend.lower()
        if backend in BACKENDS:
            return backend, ident
        raise ValueError(f"unknown backend {backend!r} (expected one of {', '.join(BACKENDS)})")
    lowered = model.lower()
    if lowered.startswith(_ANTHROPIC_PREFIXES):
        return "anthropic", model
    if lowered.startswith(_OPENAI_PREFIXES):
        return "openai", model
    if lowered.startswith(_DEEPSEEK_PREFIXES):
        return "deepseek", model
    raise ValueError(
        f"cannot infer a backend for model {model!r}. Prefix it explicitly, e.g. "
        f"'vllm:{model}' for a locally served model, or 'anthropic:{model}' / "
        f"'openai:{model}' for a hosted one."
    )


def get_provider(model: str = DEFAULT_MODEL, **kwargs) -> Provider:
    backend, ident = resolve(model)
    if backend == "openrouter":
        from .openrouter_provider import OpenRouterProvider

        return OpenRouterProvider(ident, **kwargs)
    if backend == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(ident, **_drop(kwargs, "base_url"))
    if backend == "vllm":
        from .vllm_provider import VLLMProvider

        return VLLMProvider(ident, **kwargs)
    if backend == "deepseek":
        from .deepseek_provider import DeepSeekProvider

        return DeepSeekProvider(ident, **_drop(kwargs, "effort"))
    from .openai_provider import OpenAIProvider

    return OpenAIProvider(ident, **_drop(kwargs, "base_url"))


def _drop(kwargs: dict, *keys: str) -> dict:
    """Strip backend-specific options so one CLI can feed every backend."""
    return {k: v for k, v in kwargs.items() if k not in keys}


__all__ = [
    "DEFAULT_MODEL",
    "JudgeVerdict",
    "PRICE_PER_MTOK",
    "Provider",
    "ProviderTurn",
    "assistant",
    "get_provider",
    "image_block",
    "resolve",
    "text_block",
    "token_summary",
    "tool_call_block",
    "tool_result_block",
    "usage_cost",
    "user",
]
