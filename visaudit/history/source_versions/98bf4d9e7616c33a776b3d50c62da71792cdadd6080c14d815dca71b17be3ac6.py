"""Hosted models, made to work with the sandbox's providers.

Two things a hosted run needs that the sandbox does not do on its own. Both were
measured against the real API before they were written down here.

  * `output_config={"effort": ...}` goes out on every Anthropic request. Models
    below the Claude 5 line reject it: `claude-haiku-4-5` answers HTTP 400
    ("This model does not support the effort ...") with the parameter and 200 OK
    without it. It is therefore dropped for every model not on EFFORT_MODELS.
  * A text-only agent: `view_image` is never offered, and any image already on
    the wire is replaced by a short note. `sandbox_run.endpoint_provider` needs the
    same two moves for a different reason (a text-only server rejects image
    blocks outright), so the pieces here are provider-neutral and that path can
    call them instead of keeping its own copy.

The sandbox is imported lazily, the way `sandbox_run.load_sandbox` does it, so
importing this module never requires the clone to exist — only `make_provider`
does.
"""
import os
import sys
from pathlib import Path

# The sibling clone, resolved like sandbox_run does but recomputed here so this
# module does not import that one (it pulls in the whole capture stack).
SANDBOX = Path(__file__).resolve().parents[2].parent / "ChartReviewBench" / "sandbox"

# Models known to accept `output_config.effort`. Deliberately an allowlist and
# not a blocklist: a model missing from it only loses a non-default effort
# setting (omitting the parameter is the same as "high", which is also the
# sandbox's default), while a model missing from a blocklist costs the whole
# episode an HTTP 400 on its first call. Claude Opus 4.5 does accept a reduced
# effort set but is left out on purpose — 4-5 and older is the group we measured
# rejections in, and being one setting too conservative there is free.
EFFORT_MODELS = frozenset({
    "claude-fable-5", "claude-mythos-5", "claude-opus-5",
    "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6",
    "claude-sonnet-5", "claude-sonnet-4-6",
})

#: What the model reads in place of an image it cannot be shown. Same wording the
#: ENDPOINT path already writes into its trajectories, so runs stay comparable.
TEXT_ONLY_NOTE = "[image omitted: this agent cannot view images]"

# One image block shape per wire format — Anthropic messages, vLLM/OpenAI chat
# completions, OpenAI Responses — mapped to the text block that replaces it. The
# text block is named differently on the Responses API, hence a map and not a set.
_IMAGE_TO_TEXT = {"image": "text", "image_url": "text", "input_image": "input_text"}


def supports_effort(model: str) -> bool:
    """Does this model accept `output_config={"effort": ...}`?

    Tolerates the sandbox's explicit backend prefix (`anthropic:claude-opus-5`)
    and a dated or versioned snapshot of a listed id (`claude-sonnet-5-2026...`,
    `claude-opus-4-6@2026...`), which behaves like the id it extends.
    """
    ident = model.partition(":")[2] or model
    ident = ident.strip().lower().partition("@")[0]
    return ident in EFFORT_MODELS or any(ident.startswith(m + "-") for m in EFFORT_MODELS)


def strip_view_image(tools: list) -> list:
    """The tool list without `view_image` — the only tool that returns pixels."""
    return [t for t in tools if t.get("name") != "view_image"]


def redact_images(native: list, note: str = TEXT_ONLY_NOTE) -> list:
    """Replace every image block in provider-native messages with a text note.

    Provider-neutral on purpose: the three wire formats spell an image
    differently, and Anthropic additionally nests one inside a `tool_result`, so
    the walk recurses and rewrites by block type. Anything that is not a plain
    dict is passed through untouched — an echoed assistant turn carries the SDK's
    own block objects, and those must survive the round trip unchanged.

    Nothing is mutated: a message is copied only when one of its blocks changed,
    so an unchanged history stays the exact objects the provider built.
    """
    return [_redact_message(m, note) for m in native]


def _redact_message(msg, note):
    if not isinstance(msg, dict):
        return msg
    content = msg.get("content")
    redacted = _redact_blocks(content, note)
    return msg if redacted is content else {**msg, "content": redacted}


def _redact_blocks(content, note):
    if not isinstance(content, list):
        return content
    out, changed = [], False
    for block in content:
        if not isinstance(block, dict):
            out.append(block)
            continue
        kind = block.get("type")
        if kind in _IMAGE_TO_TEXT:
            out.append({"type": _IMAGE_TO_TEXT[kind], "text": note})
            changed = True
            continue
        nested = _redact_blocks(block.get("content"), note)   # a tool_result carries its own
        if nested is not block.get("content"):
            out.append({**block, "content": nested})
            changed = True
            continue
        out.append(block)
    return out if changed else content


def compat_provider(base_cls, *, text_only: bool = False, drop_effort: bool = False):
    """Subclass a sandbox provider with the two hosted-model workarounds.

    Only the seams the sandbox already has are used: `complete` decides which
    tools are offered, `_to_native` is the single place a wire format is built,
    and `_create` / `_judge_create` are the only calls that carry `output_config`.
    Both switches are off by default, so the subclass is a no-op unless asked.
    """

    class Compat(base_cls):
        # Read back in stats(): which workaround ran is part of the result, not a
        # detail of the launch command.
        _text_only = bool(text_only)
        _drop_effort = bool(drop_effort)

        def complete(self, system, messages, tools):
            if self._text_only:
                tools = strip_view_image(tools)
            return super().complete(system, messages, tools)

        def _to_native(self, messages):
            native = super()._to_native(messages)
            return redact_images(native) if self._text_only else native

        def _create(self, **kwargs):
            if self._drop_effort:
                kwargs.pop("output_config", None)
            return super()._create(**kwargs)

        def _judge_create(self, **kwargs):
            # judge() sends the effort of the same model on a second endpoint;
            # it would fail exactly the same way.
            if self._drop_effort:
                kwargs.pop("output_config", None)
            return super()._judge_create(**kwargs)

        def stats(self):
            return {**super().stats(),
                    "text_only": self._text_only, "effort_sent": not self._drop_effort}

    Compat.__name__ = "%sCompat" % base_cls.__name__
    Compat.__qualname__ = Compat.__name__
    return Compat


def load_providers():
    """Import `chartsandbox.providers` from the sibling clone without installing it."""
    sandbox_path = Path(os.environ.get("CHARTREPAIR_SANDBOX", SANDBOX))
    src = sandbox_path / "src"
    if not src.exists():
        raise SystemExit("sandbox clone not found at %s" % sandbox_path)
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    for extra in sorted((sandbox_path / ".venv/lib").glob("python*/site-packages")):
        # appended, never inserted: our own pinned environment must win
        if str(extra) not in sys.path:
            sys.path.append(str(extra))
    from chartsandbox import providers  # noqa: E402
    return providers


def _provider_class(backend: str):
    """The class `get_provider` would instantiate — we need the class, to subclass it."""
    if backend == "anthropic":
        from chartsandbox.providers.anthropic_provider import AnthropicProvider  # noqa: E402
        return AnthropicProvider
    if backend == "openai":
        from chartsandbox.providers.openai_provider import OpenAIProvider  # noqa: E402
        return OpenAIProvider
    from chartsandbox.providers.vllm_provider import VLLMProvider  # noqa: E402
    return VLLMProvider


def make_provider(model: str, *, text_only: bool = False, max_tokens: int = 8000, **kw):
    """A sandbox provider for `model`, ready to drive an episode of this benchmark.

    Routing is the sandbox's own (`claude-*` / `gpt-*` by prefix, `vllm:<id>` and
    `anthropic:<id>` explicit), so this is a drop-in for `get_provider`. Extra
    keyword arguments reach the provider's constructor unchanged (`client=` for a
    test double, `effort=`, `judge_effort=`, `base_url=` on vLLM).

    text_only=True runs the agent without eyes: `view_image` is not offered and
    any image already on the wire becomes TEXT_ONLY_NOTE. The effort parameter is
    dropped for every Anthropic model outside EFFORT_MODELS; `stats()` records
    both decisions so a result never has to be reconstructed from the command.
    """
    providers = load_providers()
    backend, ident = providers.resolve(model)
    drop_effort = backend == "anthropic" and not supports_effort(ident)
    cls = compat_provider(_provider_class(backend), text_only=text_only, drop_effort=drop_effort)
    if backend != "vllm":
        kw.pop("base_url", None)   # as get_provider does: the hosted SDKs reject it
    return cls(ident, max_tokens=max_tokens, **kw)
