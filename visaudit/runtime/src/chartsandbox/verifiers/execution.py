"""execution — did the code run and produce the expected artifact?

params:
    produces: optional workspace-relative path that must exist and be non-empty
              (if an image, it must be openable).
If `produces` is omitted, passes iff the last execute_python had no error.
"""
from __future__ import annotations

from .base import VerifierResult, VerifyContext, register

_IMG = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


@register("execution", cheap=True)
def execution(ctx: VerifyContext) -> VerifierResult:
    produces = ctx.params.get("produces")
    if produces:
        p = ctx.workspace / produces
        if not p.exists() or p.stat().st_size == 0:
            return VerifierResult("execution", False, 0.0,
                                  f"expected artifact {produces!r} is missing or empty")
        if p.suffix.lower() in _IMG:
            try:
                from PIL import Image

                with Image.open(p) as im:
                    im.verify()
            except Exception as e:
                return VerifierResult("execution", False, 0.0,
                                      f"artifact {produces!r} is not a valid image: {e}")
        return VerifierResult("execution", True, 1.0,
                              f"artifact {produces!r} present ({p.stat().st_size} bytes)")

    le = ctx.last_exec
    if le and le.get("error"):
        return VerifierResult("execution", False, 0.0, "last execution raised an error")
    return VerifierResult("execution", True, 1.0, "no execution error")
