"""vlm_judge — rubric scoring by a (multimodal) LLM.

The real API isn't wired up yet, so this ships a MOCK that is deterministic and
clearly labelled. Swapping in a real judge later is a one-function change:

    from chartsandbox.verifiers.vlm_judge import set_judge_backend

    def my_backend(rubric, image_path, workspace) -> dict:
        # call Claude / a local vLLM here
        return {"passed": True, "score": 0.87, "detail": "..."}

    set_judge_backend(my_backend)

params:
    rubric:    natural-language description of what "pass" means
    candidate: optional workspace-relative image/file the judge should inspect
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from .base import VerifierResult, VerifyContext, register

# backend signature: (rubric: str, image_path: Path | None, workspace: Path) -> dict
_BACKEND: Callable[..., dict] | None = None


def set_judge_backend(fn: Callable[..., dict] | None) -> None:
    """Install the real judge. Pass None to fall back to the mock."""
    global _BACKEND
    _BACKEND = fn


@register("vlm_judge")
def vlm_judge(ctx: VerifyContext) -> VerifierResult:
    rubric = ctx.params.get("rubric", "")
    candidate = ctx.params.get("candidate")
    cand_path: Path | None = (ctx.workspace / candidate) if candidate else None
    cand_present = bool(cand_path and cand_path.exists() and cand_path.stat().st_size > 0)

    if _BACKEND is not None:
        v = _BACKEND(rubric=rubric, image_path=cand_path, workspace=ctx.workspace)
        return VerifierResult("vlm_judge", bool(v["passed"]), float(v["score"]),
                              "REAL: " + v.get("detail", ""))

    # --- MOCK ---------------------------------------------------------------
    # Deterministic stand-in: if the rubric references a candidate artifact, the
    # artifact must actually exist; otherwise pass. Score is a fixed placeholder.
    if candidate and not cand_present:
        return VerifierResult("vlm_judge", False, 0.0,
                              f"SIMULATED: candidate {candidate!r} absent -> fail")
    return VerifierResult("vlm_judge", True, 0.9,
                          f"SIMULATED judge (no API wired). rubric={rubric!r}")
