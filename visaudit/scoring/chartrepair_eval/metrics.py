"""The three rule-based metrics, computed on captured figure objects.

Data Fidelity   do the plotted numbers match the correct chart?
Recovery        are the properties the flaw moved back on their true value?
Preservation    is everything that was already correct still correct?

Recovery and Preservation split the chart in two: the flaw's footprint (where
the flawed chart differs from the truth) and its complement. The split is taken
under the same tolerance mode that judges the repair, so the two halves stay
complementary and no property can be counted as both flawed and already right.
Reported as a vector, never aggregated.

Every metric is reported twice: as a continuous score and as a binary verdict.
The binary form is all-or-nothing, so one wrong property disqualifies the whole
repair. Both are kept because they answer different questions: the score says
how far an agent got, the binary says whether it actually finished. For a
single-attribute flaw the two coincide by construction, which is a property of
the footprint, not a special case in the code.
"""
from . import compare
from .compare import TableIndex, equal, value_verdict

# Flaw classes that legitimately move data: matplotlib recomputes limits and
# ticks on its own, so those must not count as collateral damage.
DATA_MOVING = {
    "value-perturbation", "dropped-series", "swapped-series",
    "wrong-chart-type", "axis-truncation",
}
AUTO_PROPS = ("xlim", "ylim", "xticklabels", "yticklabels")


def category_components(category: str) -> set[str]:
    """Logical flaw categories carried by a physical task category.

    MultiFlow folders use ``multiN__a+b`` so several combinations can coexist
    under one source chart.  Metric policy belongs to ``a`` and ``b``, not to
    the storage name ``multiN``.
    """
    if "__" in category and category.split("__", 1)[0].startswith("multi"):
        return {part for part in category.split("__", 1)[1].split("+") if part}
    return {category} if category else set()


def binary_of(value):
    """All-or-nothing form of a score: 1 only when nothing at all is left wrong.

    Every score here is a fraction of integer counts, so a complete result is
    literally the float 1.0 and `== 1.0` is an identity, not a threshold. The
    rounding to four places the scores carry cannot fake one either: to round up
    to 1.0 from below, a repair would have to leave exactly one property wrong out
    of 20000, and a flat property map holds tens.

    None means the metric is undefined for this instance (an empty footprint, no
    readable data points). It stays None and must never be averaged as a zero.
    """
    if value is None:
        return None
    return 1 if value == 1.0 else 0


# --------------------------------------------------------------------------- data fidelity

def _pairs(series: dict) -> list:
    out = []
    for item in series.get("values") or []:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            out.append((item[0], item[1]))
    return out


def _match_series(gt_series: list, sub_series: list) -> list:
    """Pair series up: by (axis, label) first, then greedily by kind and size."""
    pairs, used = [], set()
    for gi, g in enumerate(gt_series):
        key = (g.get("axis"), (g.get("label") or "").strip())
        for si, s in enumerate(sub_series):
            if si in used:
                continue
            if (s.get("axis"), (s.get("label") or "").strip()) == key:
                pairs.append((gi, si))
                used.add(si)
                break
    for gi, g in enumerate(gt_series):
        if any(p[0] == gi for p in pairs):
            continue
        best, best_score = None, -1
        for si, s in enumerate(sub_series):
            if si in used:
                continue
            score = (2 if s.get("kind") == g.get("kind") else 0) + \
                    (1 if len(_pairs(s)) == len(_pairs(g)) else 0)
            if score > best_score:
                best, best_score = si, score
        if best is not None and best_score > 0:
            pairs.append((gi, best))
            used.add(best)
        else:
            pairs.append((gi, None))
    for si in range(len(sub_series)):
        if si not in used:
            pairs.append((None, si))
    return pairs


def _point_matching(edges: list[list[int]]) -> dict[int, int]:
    """Maximum one-to-one match; traversal order must not change the score."""
    from collections import deque

    matched = {}  # submitted point -> reference point
    for start in range(len(edges)):
        queue = deque([start])
        parents = {start: None}
        end = None
        while queue and end is None:
            gi = queue.popleft()
            for si in edges[gi]:
                if si not in matched:
                    end = (gi, si)
                    break
                other = matched[si]
                if other not in parents:
                    parents[other] = (gi, si)
                    queue.append(other)
        while end is not None:
            gi, si = end
            matched[si] = gi
            end = parents[gi]
    return matched


def _compare_points(gt: list, sub: list, table: TableIndex) -> dict:
    """Compare point multisets, checking BOTH coordinates in each mode.

    Data fidelity measures the data, independent of artist insertion order.
    Geometry/order changes remain visible to recovery and preservation. A
    submitted point may satisfy only one reference point, including duplicates.
    """
    from collections import Counter
    from functools import lru_cache

    # Most captures preserve the data exactly, possibly in a different order.
    # Only a complete multiset equality can bypass tolerant maximum matching.
    def exact_key(point):
        return tuple((isinstance(value, bool), value) for value in point)

    if Counter(map(exact_key, gt)) == Counter(map(exact_key, sub)):
        return {"strict": len(gt), "lenient": len(gt), "wide": len(gt),
                "n_gt": len(gt), "n_sub": len(sub), "hallucinations": []}

    grounded = lru_cache(maxsize=None, typed=True)(table.has_number)

    @lru_cache(maxsize=None, typed=True)
    def verdict(g, s):
        return value_verdict(g, s, table, grounded=grounded(g))

    modes = ("strict", "lenient", "wide")
    edges = {m: [[] for _ in gt] for m in modes}
    for gi, (gx, gy) in enumerate(gt):
        for si, (sx, sy) in enumerate(sub):
            x = verdict(gx, sx)
            if not any(x[m] for m in modes):
                continue
            y = verdict(gy, sy)
            for m in modes:
                if x[m] and y[m]:
                    edges[m][gi].append(si)
    matches = {m: _point_matching(edges[m]) for m in modes}
    res = {m: len(matches[m]) for m in modes}

    # Diagnose a changed table cell only after exact points have been removed,
    # and only where the other coordinate identifies the point. Diagnostics
    # never determine the matching or give partial credit for one coordinate.
    remaining_gt = set(range(len(gt))) - set(matches["strict"].values())
    remaining_sub = set(range(len(sub))) - set(matches["strict"])
    halluc = []
    for coordinate, anchor in ((1, 0), (0, 1)):
        for gi in sorted(remaining_gt):
            for si in sorted(remaining_sub):
                if (equal(gt[gi][anchor], sub[si][anchor], "strict")
                        and verdict(gt[gi][coordinate], sub[si][coordinate])["hallucination"]):
                    halluc.append({"coordinate": "xy"[coordinate],
                                   "expected": gt[gi][coordinate],
                                   "got": sub[si][coordinate]})
                    remaining_gt.remove(gi)
                    remaining_sub.remove(si)
                    break
    res.update(n_gt=len(gt), n_sub=len(sub), hallucinations=halluc)
    return res


def data_fidelity(gt_spec: dict, sub_spec: dict, table: TableIndex) -> dict:
    from .capture import series_of
    gt_series, sub_series = series_of(gt_spec), series_of(sub_spec)
    totals = {m: 0 for m in ("strict", "lenient", "wide")}
    n_gt = n_sub = 0
    halluc = []
    for gi, si in _match_series(gt_series, sub_series):
        gt_pts = _pairs(gt_series[gi]) if gi is not None else []
        sub_pts = _pairs(sub_series[si]) if si is not None else []
        if gi is None:
            n_sub += len(sub_pts)
            continue
        if si is None:
            n_gt += len(gt_pts)
            continue
        r = _compare_points(gt_pts, sub_pts, table)
        for m in totals:
            totals[m] += r[m]
        n_gt += r["n_gt"]
        n_sub += r["n_sub"]
        halluc.extend(r["hallucinations"])

    out = {"n_gt_points": n_gt, "n_sub_points": n_sub, "hallucinations": halluc}
    if n_gt == 0:
        # Nothing was read out of the reference chart, so there is nothing to be
        # faithful to. Reporting 1.0 here would hand a perfect score to any
        # submission, including an empty canvas — say "not applicable" instead.
        out["undefined_reason"] = ("no data points extracted from the reference chart"
                                   if n_sub == 0 else
                                   "reference has no readable data but the submission plots some")
        for m in ("strict", "lenient", "wide"):
            # Only the empty-against-empty case is undefined. A submission that
            # plots numbers the reference does not have is a real failure, so its
            # f1 of 0.0 gets a binary 0, not a None that would exempt it.
            f1 = None if n_sub == 0 else 0.0
            out[m] = {"precision": None, "recall": None, "f1": f1, "binary": binary_of(f1)}
        return out
    for m in ("strict", "lenient", "wide"):
        hit = totals[m]
        prec = hit / n_sub if n_sub else 0.0
        rec = hit / n_gt
        f1 = round(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec), 4)
        out[m] = {"precision": round(prec, 4), "recall": round(rec, 4),
                  "f1": f1, "binary": binary_of(f1)}
    return out


# --------------------------------------------------------------------------- footprint

def footprint(gt_flat: dict, flawed_flat: dict, mode: str = "strict") -> list:
    """Properties where the flawed chart differs from the truth: what the flaw did.

    Mode-aware, because the footprint and the restoration check have to use ONE
    notion of equality. When the footprint was fixed at strict while `_restored`
    asked the mode, a perturbation smaller than the mode's tolerance sat in both
    sets at once: strict-different enough to count as the flaw, lenient-equal
    enough to count as already repaired. That paid an agent that changed nothing
    a lenient recovery of 1.0. Under one mode "differs" and "is equal" partition
    the paths, and no path may be in both halves.

    The default keeps the strict reading for callers that want the flaw's full
    extent regardless of tolerance.
    """
    paths = set(gt_flat) | set(flawed_flat)
    return sorted(p for p in paths if not equal(gt_flat.get(p), flawed_flat.get(p), mode))


def _restored(gt_flat, sub_flat, path, mode, special=None) -> bool:
    if special and path in special:
        return bool(special[path](gt_flat, sub_flat))
    return equal(gt_flat.get(path), sub_flat.get(path), mode)


def recovery(gt_flat: dict, sub_flat: dict, fp: list, mode: str = "strict",
             special: dict = None) -> dict:
    """Of the properties the flaw moved, how many sit on their true value again?

    `fp` must be the footprint taken under `mode`. Handing in a stricter one
    re-opens the bug this file was fixed for: paths that the mode already calls
    equal would be counted as repaired without the agent touching them.
    An empty footprint is undefined, never a zero.

    `special` maps a footprint path to its own comparator and overrides numeric
    equality there — the two-case axis rule (axis_rule.py) rides in on it.
    """
    if not fp:
        return {"score": None, "binary": None, "fixed": 0, "injected": 0, "missed": []}
    fixed = [p for p in fp if _restored(gt_flat, sub_flat, p, mode, special)]
    sc = round(len(fixed) / len(fp), 4)
    return {
        "score": sc,
        "binary": binary_of(sc),
        "fixed": len(fixed),
        "injected": len(fp),
        "missed": [p for p in fp if p not in fixed],
    }


def preservation_scope(gt_flat: dict, sub_flat: dict, fp, category: str = "") -> tuple:
    """What preservation is allowed to ask about: (kept paths, invented paths).

    Split out so that "is preservation defined for this instance?" is answered by
    exactly one piece of code, whether or not the submission rendered.
    """
    fp = set(fp)
    keep = [p for p in gt_flat if p not in fp]
    added = [p for p in sub_flat if p not in gt_flat and p not in fp]
    if category_components(category) & DATA_MOVING:
        drop = lambda p: any(p.endswith("." + a) for a in AUTO_PROPS)  # noqa: E731
        keep = [p for p in keep if not drop(p)]
        added = [p for p in added if not drop(p)]
    return keep, added


def preservation(gt_flat: dict, flawed_flat: dict, sub_flat: dict,
                 category: str = "", mode: str = "strict") -> dict:
    """Of the aspects that were already correct, how many still are?

    Properties the submission invents (absent from the truth) count too: the
    chart then asserts something the data never said, which is damage as much
    as losing a label is.

    The footprint is taken under the SAME mode that judges the survivors, so the
    two halves stay complementary. A looser mode therefore moves a sub-tolerance
    property out of the flaw's territory and into this one: under that mode the
    flaw did not touch it, so the repair had to leave it alone, and here is where
    that is checked. Mixing the modes would either double-count such a property
    or drop it from both halves.
    """
    fp = set(footprint(gt_flat, flawed_flat, mode))
    keep, added = preservation_scope(gt_flat, sub_flat, fp, category)
    checked = keep + added
    if not checked:
        return {"score": None, "binary": None, "kept": 0, "total": 0, "damaged": [], "added": []}
    ok = [p for p in keep if _restored(gt_flat, sub_flat, p, mode)]
    sc = round(len(ok) / len(checked), 4)
    return {
        "score": sc,
        "binary": binary_of(sc),
        "kept": len(ok),
        "total": len(checked),
        "damaged": [p for p in keep if p not in ok],
        "added": added,
    }


# --------------------------------------------------------------------------- the vector

def score(gt: dict, flawed: dict, sub: dict, table: TableIndex, category: str = "",
          summary_text: str = None) -> dict:
    """Assemble the result vector for one submission. Never aggregated."""
    # The footprint reads gt and flawed only, never the submission, so it is a
    # property of the INSTANCE. Computing it up front keeps whether a metric is
    # defined independent of whether the agent's code happened to run.
    fps = {m: footprint(gt["flat"], flawed["flat"], m) for m in compare.MODES}

    # The two-case axis rule (sync 25.08): where the flaw moved an axis limit,
    # a summary-stated span or the data extent judges the repair instead of
    # float equality against autoscale padding. Instance-level, mode-free.
    special = {}
    if "axis-truncation" in category_components(category):
        from . import axis_rule
        special = axis_rule.comparators(fps["strict"], gt["flat"], flawed["flat"],
                                        gt.get("spec") or {}, summary_text)

    if sub["status"] != "ok" or not sub["spec"]:
        # A submission that does not render achieves nothing, but achieving
        # nothing is not the same as the question not applying. Where the metric
        # has a denominator the score is 0.0 and the instance stays in it
        # (ChartMimic convention); where it has none the answer is None, exactly
        # as it would be had the same code rendered. Collapsing that None to 0.0
        # let an instance join or leave the recovery denominator purely because
        # the agent crashed.
        fid = data_fidelity(gt.get("spec") or {}, {}, table)   # no submission points at all
        rec = {m: (None if not fps[m] else 0.0) for m in compare.MODES}
        pres = {}
        for m in compare.MODES:
            keep, added = preservation_scope(gt["flat"], {}, fps[m], category)
            pres[m] = None if not (keep + added) else 0.0
        return {
            "executability": 0,
            "data_fidelity": {m: fid[m]["f1"] for m in compare.MODES},
            "data_fidelity_binary": {m: fid[m]["binary"] for m in compare.MODES},
            "recovery": rec,
            "recovery_binary": {m: binary_of(rec[m]) for m in compare.MODES},
            "preservation": pres,
            "preservation_binary": {m: binary_of(pres[m]) for m in compare.MODES},
            "diagnostics": {"status": sub["status"], "footprint": fps["strict"],
                            "footprint_by_mode": fps,
                            "stderr": (sub.get("stderr") or "")[-600:]},
        }

    fp = fps["strict"]      # several callers read diagnostics["footprint"] as one
                            # list of paths; the strict reading is the flaw's full
                            # extent, so that is the one kept under the old key.
    fid = data_fidelity(gt["spec"], sub["spec"], table)
    out = {
        "executability": 1,
        "data_fidelity": {m: fid[m]["f1"] for m in compare.MODES},
        "data_fidelity_binary": {m: fid[m]["binary"] for m in compare.MODES},
        "recovery": {}, "recovery_binary": {},
        "preservation": {}, "preservation_binary": {},
        "diagnostics": {
            "status": "ok",
            "footprint": fp,
            "footprint_by_mode": fps,
            "data_fidelity_detail": {m: fid[m] for m in ("strict", "lenient", "wide")},
            "data_fidelity_wide": fid["wide"]["f1"],
            "hallucinations": fid["hallucinations"],
            "n_points": {"gt": fid["n_gt_points"], "submission": fid["n_sub_points"]},
        },
    }
    if special:
        out["diagnostics"]["axis_rule_paths"] = sorted(special)
    for m in compare.MODES:
        rec = recovery(gt["flat"], sub["flat"], fps[m], m, special)
        pres = preservation(gt["flat"], flawed["flat"], sub["flat"], category, m)
        out["recovery"][m] = rec["score"]
        out["recovery_binary"][m] = rec["binary"]
        out["preservation"][m] = pres["score"]
        out["preservation_binary"][m] = pres["binary"]
        out["diagnostics"].setdefault("recovery_detail", {})[m] = rec
        out["diagnostics"].setdefault("preservation_detail", {})[m] = pres
    return out
