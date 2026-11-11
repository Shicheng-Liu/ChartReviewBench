"""Hosted models, made to work with the sandbox's providers.

Hosted-model compatibility and explicit request/accounting settings.

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
import base64
import hashlib
import io
import os
import sys
from pathlib import Path

# The sibling clone, resolved like sandbox_run does but recomputed here so this
# module does not import that one (it pulls in the whole capture stack).
SANDBOX = Path(__file__).resolve().parents[2] / "runtime"

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

# Explicit snapshot mapping: never guess the price of a new model generation.
HAIKU_PRICE_MODELS = {
    "claude-haiku-4-5": "claude-haiku-4-5",
    "claude-haiku-4-5-20251001": "claude-haiku-4-5",
}

# Haiku rejected these initial and generated chart images above 8,000 pixels.
# Apply this policy only to the explicitly supported Haiku deployments.
HAIKU_IMAGE_TRANSPORT_POLICY = {
    "version": "anthropic-haiku-max-dimension-8000-v1",
    "max_dimension": 8000,
    "resampling": "LANCZOS",
    "oversize_encoding": "PNG",
    "scope": "request_payload_only",
}

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


def _fit_anthropic_images(messages, max_dimension, cache, resized):
    """Downscale oversized request images without mutating the source history.

    Keep a small per-episode payload cache because every model call replays the
    same images. Only compact provenance, never pixels, enters episode stats.
    """
    from PIL import Image

    def fit(block):
        if not isinstance(block, dict):
            return block
        source = block.get("source")
        if (block.get("type") == "image" and isinstance(source, dict)
                and source.get("type") == "base64"):
            encoded = source["data"]
            digest = hashlib.sha256(encoded.encode("ascii")).hexdigest()
            if digest in cache:
                replacement = cache.pop(digest)
            else:
                raw = base64.b64decode(encoded, validate=True)
                replacement = None
                with Image.open(io.BytesIO(raw)) as original:
                    width, height = original.size
                    if max(width, height) > max_dimension:
                        scale = max_dimension / max(width, height)
                        size = (max(1, round(width * scale)), max(1, round(height * scale)))
                        image = original.convert("RGBA" if "A" in original.getbands()
                                                 or "transparency" in original.info else "RGB")
                        image = image.resize(size, Image.Resampling.LANCZOS)
                        output = io.BytesIO()
                        image.save(output, format="PNG")
                        payload = output.getvalue()
                        replacement = {**source, "media_type": "image/png",
                                       "data": base64.b64encode(payload).decode("ascii")}
                        resized[digest] = {
                            "source_sha256": hashlib.sha256(raw).hexdigest(),
                            "source_dimensions": [width, height],
                            "request_sha256": hashlib.sha256(payload).hexdigest(),
                            "request_dimensions": list(size),
                        }
            cache[digest] = replacement
            if len(cache) > 16:
                cache.pop(next(iter(cache)))
            return block if replacement is None else {**block, "source": replacement}

        content = block.get("content")
        if not isinstance(content, list):
            return block
        changed = [fit(child) for child in content]
        return (block if all(a is b for a, b in zip(content, changed))
                else {**block, "content": changed})

    return [fit(message) for message in messages]


def compat_provider(base_cls, *, text_only: bool = False, drop_effort: bool = False,
                    pricing_model=None, temperature=None, image_transport_policy=None):
    """Subclass a sandbox provider with hosted-model compatibility settings.

    Only the seams the sandbox already has are used: `complete` decides which
    tools are offered, `_to_native` is the single place a wire format is built,
    and `_create` / `_judge_create` are the only calls that carry `output_config`.
    Compatibility settings are disabled unless explicitly selected.
    """

    class Compat(base_cls):
        # Read back in stats(): which workaround ran is part of the result, not a
        # detail of the launch command.
        _text_only = bool(text_only)
        _drop_effort = bool(drop_effort)
        _pricing_model = pricing_model
        _temperature = temperature
        _image_transport_policy = image_transport_policy

        @property
        def cost_usd(self):
            if self._pricing_model is None:
                return super().cost_usd
            from chartsandbox.providers.base import usage_cost
            return usage_cost(self._pricing_model, self.usage)

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
            if self._temperature is not None:
                kwargs["temperature"] = self._temperature
            kwargs = self._prepare_images(kwargs)
            response = super()._create(**kwargs)
            model = getattr(response, "model", None)
            if isinstance(model, str):
                self._response_models = getattr(self, "_response_models", set()) | {model}
            return response

        def _judge_create(self, **kwargs):
            # judge() sends the effort of the same model on a second endpoint;
            # it would fail exactly the same way.
            if self._drop_effort:
                kwargs.pop("output_config", None)
            kwargs = self._prepare_images(kwargs)
            return super()._judge_create(**kwargs)

        def _prepare_images(self, kwargs):
            if self._image_transport_policy is not None and not self._text_only:
                if not hasattr(self, "_image_transport_cache"):
                    self._image_transport_cache = {}
                    self._image_transport_resizes = {}
                kwargs["messages"] = _fit_anthropic_images(
                    kwargs["messages"], self._image_transport_policy["max_dimension"],
                    self._image_transport_cache, self._image_transport_resizes)
            return kwargs

        def stats(self):
            stats = {**super().stats(),
                     "text_only": self._text_only, "effort_sent": not self._drop_effort}
            if self._pricing_model is not None:
                stats.update(pricing_model=self._pricing_model,
                             request_temperature=self._temperature,
                             response_models=sorted(getattr(self, "_response_models", set())))
            if self._image_transport_policy is not None and not self._text_only:
                stats["image_transport_policy"] = dict(self._image_transport_policy)
                stats["image_transport_resizes"] = list(
                    getattr(self, "_image_transport_resizes", {}).values())
            return stats

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
    pricing_model = HAIKU_PRICE_MODELS.get(ident) if backend == "anthropic" else None
    cls = compat_provider(_provider_class(backend), text_only=text_only, drop_effort=drop_effort,
                          pricing_model=pricing_model,
                          temperature=0 if pricing_model else None,
                          image_transport_policy=HAIKU_IMAGE_TRANSPORT_POLICY if pricing_model else None)
    if backend != "vllm":
        kw.pop("base_url", None)   # as get_provider does: the hosted SDKs reject it
    return cls(ident, max_tokens=max_tokens, **kw)
