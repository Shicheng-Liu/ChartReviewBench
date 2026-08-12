"""Real multimodal judge for the `vlm_judge` verifier.

`vlm_judge` ships a labelled mock; this installs a backend that actually looks at
the rendered chart and scores it against the subgoal's rubric, with the verdict
constrained to a schema so a malformed score can't reach a result file.

    from chartsandbox.judge import install_judge
    install_judge("claude-opus-5")

The judge may read the oracle — it is the verifier, not the agent. The agent's
tools stay path-sandboxed to the workspace and can never see these files.
Attaching the reference image is opt-in per subgoal, via a `reference:` param in
task.yaml, because for an *edit* subgoal the reference shows the pre-edit chart
and would argue against the very change being graded.
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from .providers import Provider, get_provider
from .verifiers import set_judge_backend

_MIME = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
         "gif": "image/gif", "webp": "image/webp"}


def _load(path: Path, label: str) -> dict | None:
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if not raw:
        return None
    mime = _MIME.get(path.suffix.lstrip(".").lower())
    if not mime:
        return None  # judging a PDF/SVG needs a raster step we don't do yet
    return {"mime": mime, "b64": base64.b64encode(raw).decode(), "label": label}


def make_judge_backend(provider: Provider):
    """Build a `set_judge_backend`-compatible callable from a provider."""

    def backend(rubric: str, image_path: Path | None, workspace: Path,
                oracle: Path | None = None, params: dict | None = None) -> dict:
        params = params or {}
        images: list[dict] = []

        if image_path is not None:
            candidate = _load(image_path, f"candidate ({image_path.name})")
            if candidate is None:
                return {"passed": False, "score": 0.0,
                        "detail": f"candidate {image_path.name!r} missing, empty or not a raster image"}
            images.append(candidate)

        reference = params.get("reference")
        if reference and oracle is not None:
            ref = _load(oracle / reference, f"reference ({reference})")
            if ref is not None:
                images.append(ref)

        if not images:
            return {"passed": False, "score": 0.0, "detail": "no image to judge"}

        context = ("The first image is the agent's output. The second is the reference "
                   "the task was derived from." if len(images) > 1
                   else "The image is the agent's output.")
        verdict = provider.judge(rubric=rubric, images=images, context=context)
        return {"passed": bool(verdict.passed),
                "score": min(1.0, max(0.0, float(verdict.score))),
                "detail": verdict.detail}

    return backend


def install_judge(model: str, **provider_kwargs: Any) -> Provider:
    """Point `vlm_judge` at a real model. Returns the provider, for cost accounting."""
    provider = get_provider(model, **provider_kwargs)
    set_judge_backend(make_judge_backend(provider))
    return provider
