"""vlm_judge — rubric scoring by a (multimodal) LLM.

Ships a deterministic, clearly-labelled MOCK. Installing a real judge is a
one-function change — `chartsandbox.judge.install_judge` does it for you, or
supply your own:

    from chartsandbox.verifiers.vlm_judge import set_judge_backend

    def my_backend(rubric, image_path, workspace) -> dict:
        # call Claude / GPT / a local vLLM here
        return {"passed": True, "score": 0.87, "detail": "..."}

    set_judge_backend(my_backend)

A backend may also accept `oracle` (the task's read-only reference directory) and
`params` (the subgoal's verifier params); both are passed only if its signature
accepts them, so the three-argument form above keeps working.

params:
    rubric:    natural-language description of what "pass" means
    candidate: optional workspace-relative image/file the judge should inspect
    reference: optional oracle-relative image to show the judge alongside the
               candidate. Off by default — on an *edit* subgoal the reference is
               the pre-edit chart and argues against the change being graded.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Callable

from .base import VerifierResult, VerifyContext, register

# backend signature: (rubric: str, image_path: Path | None, workspace: Path,
#                     oracle: Path | None = ..., params: dict = ...) -> dict
_BACKEND: Callable[..., dict] | None = None


def set_judge_backend(fn: Callable[..., dict] | None) -> None:
    """Install the real judge. Pass None to fall back to the mock."""
    global _BACKEND
    _BACKEND = fn


def _call_backend(fn: Callable[..., dict], **kwargs) -> dict:
    """Call `fn` with the arguments it actually declares.

    Lets us hand richer context (oracle, params) to backends that want it without
    breaking the documented three-argument signature.
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # builtins / C callables
        return fn(rubric=kwargs["rubric"], image_path=kwargs["image_path"],
                  workspace=kwargs["workspace"])
    if any(p.kind is p.VAR_KEYWORD for p in params.values()):
        return fn(**kwargs)
    return fn(**{k: v for k, v in kwargs.items() if k in params})


@register("vlm_judge")
def vlm_judge(ctx: VerifyContext) -> VerifierResult:
    rubric = ctx.params.get("rubric", "")
    candidate = ctx.params.get("candidate")
    cand_path: Path | None = (ctx.workspace / candidate) if candidate else None
    cand_present = bool(cand_path and cand_path.exists() and cand_path.stat().st_size > 0)

    if _BACKEND is not None:
        v = _call_backend(_BACKEND, rubric=rubric, image_path=cand_path,
                          workspace=ctx.workspace, oracle=ctx.oracle, params=ctx.params)
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
