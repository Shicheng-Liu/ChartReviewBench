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

A rubric that names a file gets that file. Rubrics are written against the task's
ground truth — "every plotted value matches the corresponding number in table.csv",
"the colors match the mapping described in summary.txt" — and a judge holding only
the image cannot act on them: it answers that it was not given table.csv and scores
zero, which reads as a failed repair when it is a failed question. So any text file
the rubric names by filename is inlined into the judge's context. An explicit
`context_files:` param overrides the guess.
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from .providers import Provider, get_provider
from .verifiers import set_judge_backend

_MIME = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
         "gif": "image/gif", "webp": "image/webp"}


#: Enough of a data file to adjudicate against, bounded so a large table cannot
#: crowd the rubric out of the judge's context.
_MAX_CONTEXT_CHARS = 6000
_TEXT_SUFFIXES = {".csv", ".txt", ".json", ".md", ".tsv"}


def _context_files(rubric: str, workspace: Path, oracle: Path | None,
                   params: dict) -> list[tuple[str, str]]:
    """(name, content) for each ground-truth file the rubric relies on.

    Explicit `context_files` wins; otherwise any text file in the workspace or the
    oracle whose filename appears in the rubric is taken to be one the judge needs.
    """
    named = params.get("context_files")
    roots = [r for r in (workspace, oracle) if r is not None]
    out: list[tuple[str, str]] = []
    seen: set[str] = set()

    if named:
        wanted = [(root / n, n) for n in named for root in roots]
    else:
        wanted = []
        for root in roots:
            try:
                entries = sorted(root.iterdir())
            except OSError:
                continue
            for f in entries:
                if (f.is_file() and f.suffix.lower() in _TEXT_SUFFIXES
                        and f.name in rubric):
                    wanted.append((f, f.name))

    for path, name in wanted:
        if name in seen or not path.is_file():
            continue
        try:
            body = path.read_text(errors="replace")
        except OSError:
            continue
        seen.add(name)
        if len(body) > _MAX_CONTEXT_CHARS:
            body = body[:_MAX_CONTEXT_CHARS] + f"\n... [clipped, {len(body)} chars total]"
        out.append((name, body))
    return out


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


def _reconcile(passed: bool, score: float, detail: str) -> dict:
    """Keep the verdict and the score from contradicting each other.

    `passed` is the field that survives misreading — it is a boolean about the
    rubric. `score` does not: a judge asked for a number between 0 and 1 will
    sometimes answer with confidence in its own verdict rather than with how far
    the rubric is met, and then a chart it is *certain* is broken comes back as
    `passed: false, score: 1.0`. That single flipped meaning is enough to hand a
    failed subgoal a perfect weighted score, so the boolean arbitrates and the
    number is pulled into the half it belongs in. The adjustment is written into
    the detail rather than applied quietly, because a clamped score is evidence
    that the judge prompt still needs work.
    """
    if not passed and score > 0.5:
        return {"passed": False, "score": 0.5,
                "detail": f"[score {score:.2f} contradicted a failing verdict; "
                          f"clamped to 0.50] {detail}"}
    if passed and score < 0.5:
        return {"passed": True, "score": 0.5,
                "detail": f"[score {score:.2f} contradicted a passing verdict; "
                          f"raised to 0.50] {detail}"}
    return {"passed": passed, "score": score, "detail": detail}


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

        parts = ["The first image is the agent's output. The second is the reference "
                 "the task was derived from." if len(images) > 1
                 else "The image is the agent's output."]
        for name, body in _context_files(rubric, workspace, oracle, params):
            parts.append(f"--- {name} ---\n{body}")
        verdict = provider.judge(rubric=rubric, images=images, context="\n\n".join(parts))
        return _reconcile(bool(verdict.passed),
                          min(1.0, max(0.0, float(verdict.score))),
                          verdict.detail)

    return backend


def install_judge(model: str, **provider_kwargs: Any) -> Provider:
    """Point `vlm_judge` at a real model. Returns the provider, for cost accounting."""
    provider = get_provider(model, **provider_kwargs)
    set_judge_backend(make_judge_backend(provider))
    return provider
