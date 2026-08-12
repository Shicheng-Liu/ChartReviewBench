"""data — compare numeric/tabular output against an oracle reference.

params:
    candidate: workspace-relative CSV the agent produced (e.g. extracted values)
    reference: oracle-relative CSV to compare against
    tolerance: absolute tolerance for numeric cells (default 1e-6)

Score = fraction of aligned cells that match (numeric within tolerance, text exact).
Robust to shape mismatch: missing/extra cells count as misses.
"""
from __future__ import annotations

from .base import VerifierResult, VerifyContext, register


@register("data", cheap=True)
def data_match(ctx: VerifyContext) -> VerifierResult:
    import numpy as np
    import pandas as pd

    cand_p = ctx.workspace / ctx.params["candidate"]
    ref_p = ctx.oracle / ctx.params["reference"]
    tol = float(ctx.params.get("tolerance", 1e-6))

    if not cand_p.exists():
        return VerifierResult("data", False, 0.0, f"candidate {ctx.params['candidate']!r} not found")
    if not ref_p.exists():
        return VerifierResult("data", False, 0.0, f"oracle reference {ctx.params['reference']!r} not found")

    cand = pd.read_csv(cand_p)
    ref = pd.read_csv(ref_p)

    total = int(ref.shape[0] * ref.shape[1])
    if total == 0:
        return VerifierResult("data", True, 1.0, "empty reference")

    matched = 0
    for col in ref.columns:
        if col not in cand.columns:
            continue
        r = ref[col].reset_index(drop=True)
        c = cand[col].reset_index(drop=True)
        n = min(len(r), len(c))
        for i in range(n):
            rv, cv = r.iloc[i], c.iloc[i]
            try:
                if abs(float(rv) - float(cv)) <= tol:
                    matched += 1
            except (TypeError, ValueError):
                if str(rv) == str(cv):
                    matched += 1

    score = matched / total
    return VerifierResult("data", score >= 0.999, score,
                          f"{matched}/{total} cells matched (tol={tol})")
