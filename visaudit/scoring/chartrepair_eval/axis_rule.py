"""Axis repair follows bound text claims AND preserves reference geometry.

Only a range tied to the named axis can constrain its limits. Tick labels and
data domains are not limits; ambiguous axis claims get no semantic waiver.
Explicit limits retain the existing 10% autoscale allowance, but that allowance
never permits clipping reference data. With no bound limits, containment is the
existing two-case contract (sync 25.08), not equality to autoscaled floats.

Containment uses plotted geometry, not inferred raw observations: this preserves
the existing conservative treatment of filled baselines and KDE outlines. It
does not decide that a clipped KDE tail is scientifically equivalent. Rectangle
edges include stacked offsets and histogram bin widths. Unsupported geometry
does not create evidence for a semantic waiver.
"""
import math
import re

# "ranging from 0 to 14,000 million USD" · "ranges from 0% to 35%" ·
# "range is from 0 to 90" · "between 0 and 250 USD"
_NUM = r"[-+−–]?\$?\d[\d,]*(?:\.\d+)?"
_RANGES = (
    re.compile(rf"rang(?:e|es|ing)\s+(?:is\s+)?from\s+({_NUM})\s*%?"
               rf"[^.;\d]{{0,24}}?(?:up\s+)?to\s+({_NUM})", re.I),
    re.compile(rf"from\s+({_NUM})\s*%?\s+(?:up\s+)?to\s+({_NUM})", re.I),
    re.compile(rf"between\s+({_NUM})\s*%?\s+and\s+({_NUM})", re.I),
)


def _num(token: str) -> float:
    return float(token.replace(",", "").replace("$", "").replace("−", "-").replace("–", "-"))


def spans_from_summary(text: str) -> list:
    """Every numeric span the summary states, as (lo, hi) with lo < hi."""
    spans, seen = [], set()
    for rx in _RANGES:
        for m in rx.finditer(text or ""):
            try:
                a, b = _num(m.group(1)), _num(m.group(2))
            except ValueError:
                continue
            lo, hi = (a, b) if a <= b else (b, a)
            if hi > lo and (lo, hi) not in seen:
                seen.add((lo, hi))
                spans.append((lo, hi))
    return spans


def _iou(a, b) -> float:
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    if hi <= lo:
        return 0.0
    union = max(a[1], b[1]) - min(a[0], b[0])
    return (hi - lo) / union if union else 0.0


def pinned_span(spans: list, gt_limits) -> tuple:
    """Legacy numeric lookup for historical diagnostics; NOT a scoring rule.

    New callers must use bound_span with the original text and axis identity.
    """
    lims = _pair(gt_limits)
    if lims is None or not spans:
        return None
    best = max(spans, key=lambda s: _iou(s, lims))
    return best if _iou(best, lims) >= 0.3 else None


def _pair(value):
    if (isinstance(value, (list, tuple)) and len(value) == 2
            and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                    and math.isfinite(v) for v in value)):
        lo, hi = value
        return (lo, hi) if lo <= hi else (hi, lo)
    return None


def _contains(limits, extent):
    limits, extent = _pair(limits), _pair(extent)
    if limits is None or extent is None:
        return False
    epsilon = 1e-9 * max(*(abs(value) for value in (*limits, *extent)), 1.0)
    return limits[0] <= extent[0] + epsilon and limits[1] >= extent[1] - epsilon


_AXIS = re.compile(
    r"\b(?P<name>(?:(?:left|right|top|bottom)(?:[ -]hand)?\s+)?"
    r"(?:[xy][ -]?axis|(?:horizontal|vertical|radial)(?:\s*\([xy]\))? axis)|"
    r"(?:left|right|top|bottom)(?:[ -]hand)? axis)\b", re.I)
_UNIT = r"(?:thousands?|millions?|billions?|[kmb])\b|%|percent(?:age)?s?\b"
_RANGE = re.compile(
    rf"\b(?:from|between)\s+(?:approximately\s+|about\s+|around\s+)?(?P<lo>{_NUM})"
    rf"\s*(?P<lo_unit>{_UNIT})?"
    rf"(?:(?!\b(?:axis|from|between)\b)[^.;\d]){{0,32}}?\b(?:to|and)\s+"
    rf"(?:approximately\s+|about\s+|around\s+|slightly\s+over\s+)?(?P<hi>{_NUM})"
    rf"\s*(?P<hi_unit>{_UNIT})?", re.I)
_EXPLICIT_LIMIT = re.compile(r"\b(?:limits?|bounds?|minimum|maximum|endpoints?)\b", re.I)
_TICKS = re.compile(r"\b(?:ticks?|ticklabels?|tick labels?|marked|markings|increments|gridlines)\b", re.I)
_DOMAIN = re.compile(
    r"\b(?:on a scale|scoring system|score domain|values? (?:range|ranges|ranging|span|spans|spanning|vary)|"
    r"scores? (?:range|ranges|ranging|span|spans|spanning)|which spans|which ranges)\b", re.I)
_CURRENCY = r"(?:\b(?:USD|EUR|GBP|JPY|euros?|dollars?|pounds?|yen)\b|US\$|[$€£])"


def _unit_scale(unit):
    unit = unit.lower()
    if unit == "%" or unit.startswith("percent"):
        return "percent"
    return {"k": 1e3, "m": 1e6, "b": 1e9}.get(unit, {
        "thousand": 1e3, "million": 1e6, "billion": 1e9,
    }.get(unit.rstrip("s")))


def _label_scales(label):
    scales = {_unit_scale(match.group()) for match in re.finditer(
        r"\b(?:thousands?|millions?|billions?|percent(?:age)?s?)\b|%", label, re.I)}
    # A bare 'm' could mean metres. Currency-adjacent abbreviations identify
    # scale explicitly without guessing from the reference values' magnitude.
    for match in re.finditer(rf"{_CURRENCY}\s*[(]?\s*([kmb])\b|\b([kmb])\s*[)]?\s*{_CURRENCY}", label, re.I):
        scales.add(_unit_scale(next(group for group in match.groups() if group)))
    return scales


def _bound_units(claim, label, policy):
    """Normalize a text span only when the coordinate unit is explicit."""
    units = claim["units"]
    if claim.get("ambiguous_unit_hint"):
        return None, {"reason": "conflicting_text_scales", "axis_label": label}
    if not any(units):
        return claim["span"], {"reason": "native_axis_numbers"}
    scales = [_unit_scale(unit) if unit else None for unit in units]
    scales = [scale if scale is not None else scales[1 - index]
              for index, scale in enumerate(scales)]
    label_scales = _label_scales(label)
    detail = {"axis_label": label, "text_units": units, "reason": "ambiguous_axis_units"}
    if len(label_scales) > 1 or ("percent" in scales and set(scales) != {"percent"}):
        return None, detail
    if scales == ["percent", "percent"]:
        formatter = (policy or {}).get("formatter") or {}
        if formatter.get("class") == "matplotlib.ticker.PercentFormatter":
            xmax = (formatter.get("config") or {}).get("xmax")
            if not isinstance(xmax, (int, float)) or isinstance(xmax, bool) or not math.isfinite(xmax) or xmax <= 0:
                return None, detail
            factors = [xmax / 100] * 2
            detail["reason"] = "percent_formatter_scale"
        elif label_scales == {"percent"} and formatter.get("class") in (None, "matplotlib.ticker.ScalarFormatter"):
            factors = [1.0, 1.0]
            detail["reason"] = "native_percent_label"
        else:
            return None, detail
    else:
        if label_scales == {"percent"}:
            return None, detail
        coordinate_scale = next(iter(label_scales)) if label_scales else 1.0 if re.search(_CURRENCY, label, re.I) else None
        if coordinate_scale is None:
            return None, detail
        factors = [scale / coordinate_scale for scale in scales]
        detail.update(reason="explicit_axis_label_scale", coordinate_scale=coordinate_scale)
    detail["factors"] = factors
    return _pair([value * factor for value, factor in zip(claim["endpoints"], factors)]), detail


def _axis_claims(text):
    """Ranges stay inside their sentence and the nearest explicit axis clause."""
    claims = []
    for sentence in re.finditer(r"[^\n;.!?]+(?:\.(?=\d)[^\n;.!?]+)*", text or ""):
        # Decimal points are kept by the tokenizer; punctuation ends a clause.
        segment = sentence.group()
        axes = list(_AXIS.finditer(segment))
        for index, axis in enumerate(axes):
            end = axes[index + 1].start() if index + 1 < len(axes) else len(segment)
            clause = segment[axis.end():end]
            name = axis.group().lower()
            letter = "x" if any(word in name for word in ("xaxis", "x-axis", "x axis", "horizontal", "top", "bottom")) else "y"
            side = next((side for side in ("left", "right", "top", "bottom") if side in name), None)
            ranges = list(_RANGE.finditer(clause))
            if not ranges and _EXPLICIT_LIMIT.search(clause):
                start, stop = sentence.start() + axis.start(), sentence.start() + end
                claims.append({"axis": letter, "side": side, "span": None,
                               "kind": "unparsed", "explicit": True,
                               "text": text[start:stop], "start": start, "end": stop})
            previous_end = 0
            for match in ranges:
                endpoints = [_num(match["lo"]), _num(match["hi"])]
                span = _pair(endpoints)
                if span is None:
                    continue
                prefix = clause[previous_end:match.start()]
                units = [match["lo_unit"], match["hi_unit"]]
                stated_scales = _label_scales(prefix)
                if not any(units):
                    if len(stated_scales) == 1:
                        scale = next(iter(stated_scales))
                        unit = "percent" if scale == "percent" else {1e3: "thousand", 1e6: "million", 1e9: "billion"}[scale]
                        units = [unit, unit]
                explicit = bool(_EXPLICIT_LIMIT.search(prefix))
                kind = "limits" if explicit else "ticks" if _TICKS.search(prefix) else "data" if _DOMAIN.search(prefix) else "range"
                if previous_end and not explicit:
                    kind = "data"
                previous_end = match.end()
                start = sentence.start() + axis.start()
                stop = sentence.start() + axis.end() + match.end()
                claims.append({"axis": letter, "side": side, "span": span,
                               "endpoints": endpoints, "units": units,
                               "ambiguous_unit_hint": len(stated_scales) > 1,
                               "kind": kind, "explicit": explicit,
                               "text": text[start:stop], "start": start, "end": stop})
    return claims


def axis_constraint(summary_text, axes_index, axis_letter, gt_spec, gt_flat):
    """Auditable text binding: status, span, reason, and original claim spans."""
    axes = gt_spec.get("axes") or [{}]
    claims = []
    ambiguous = False
    path = f"axes[{axes_index}].{axis_letter}lim"
    limits = gt_flat.get(path)
    extent = data_extent(gt_spec, axes_index, axis_letter, limits)
    for claim in _axis_claims(summary_text or ""):
        if claim["axis"] != axis_letter:
            continue
        candidates = list(range(len(axes)))
        if claim["side"]:
            candidates = [index for index, axis in enumerate(axes)
                          if (axis.get("axis_sides") or {}).get(axis_letter,
                              "left" if axis_letter == "y" and len(axes) == 1 else
                              "bottom" if axis_letter == "x" and len(axes) == 1 else None) == claim["side"]]
        if axes_index not in candidates:
            continue
        claim = dict(claim)
        if claim["kind"] == "unparsed":
            claim["binding"] = "unparsed_explicit_limits"
            ambiguous = True
        elif len(candidates) != 1:
            claim["binding"] = "ambiguous_axis"
            ambiguous = ambiguous or claim["kind"] in {"limits", "range"}
        elif claim["kind"] in {"ticks", "data"}:
            claim["binding"] = claim["kind"] + "_not_limits"
        else:
            label = gt_flat.get(path[:-3] + "label") or axes[axes_index].get(axis_letter + "label") or ""
            span, unit_binding = _bound_units(claim, label, gt_flat.get(path[:-3] + "tick_policy"))
            claim.update(raw_span=claim["span"], unit_binding=unit_binding)
            if span is None or span[0] == span[1]:
                claim["binding"] = "ambiguous_axis_units" if span is None else "degenerate_axis_span"
                ambiguous = True
                claims.append(claim)
                continue
            claim["span"] = span
            visible = [tick for tick in gt_flat.get(path[:-3] + "ticks", [])
                       if _pair([tick, tick]) and _contains(limits, [tick, tick])]
            tick_span = (min(visible), max(visible)) if visible else None
            if (not claim["explicit"] and tick_span == claim["span"]
                    and extent is not None and not _contains(claim["span"], extent)):
                claim["binding"] = "visible_tick_endpoints_not_limits"
            else:
                claim["binding"] = "axis_limits"
        claims.append(claim)
    spans = {claim["span"] for claim in claims if claim["binding"] == "axis_limits"}
    if ambiguous or len(spans) > 1:
        return {"status": "ambiguous", "span": None, "reason": "ambiguous_axis_limits", "claims": claims}
    if spans:
        return {"status": "bound", "span": next(iter(spans)), "reason": "explicit_axis_binding", "claims": claims}
    return {"status": "unbound", "span": None, "reason": "no_bound_axis_limits", "claims": claims}


def bound_span(summary_text, axes_index, axis_letter, gt_spec, gt_flat):
    return axis_constraint(summary_text, axes_index, axis_letter, gt_spec, gt_flat)["span"]


def data_extent(gt_spec: dict, axes_index: int, axis_letter: str, gt_limits) -> tuple:
    """What the plotted data spans on that axis, the zero baseline included when
    the true chart shows it (a bar's foot is part of the drawing)."""
    from .capture import series_of
    vals = []
    for series in series_of(gt_spec):
        if series.get("axis") != axes_index:
            continue
        if axis_letter in (series.get("axis_bounds_unavailable") or []):
            return None
        if series.get("kind") not in {"bar", "barh", "line", "scatter", "area", "box"}:
            continue
        bounds = (series.get("axis_bounds") or {}).get(axis_letter)
        if _pair(bounds) is not None:
            vals.extend(bounds)
            continue
        # The extractor stores vertical bars as [x-position, height], but
        # horizontal bars as [y-position, width]. Treating every pair as [x,y]
        # made a free horizontal x-axis compare against category indices 0..31
        # instead of values 10.2..25.1. A correct 0..26.355 repair then failed.
        if series.get("kind") == "barh":
            coord = 1 if axis_letter == "x" else 0
        else:
            coord = 0 if axis_letter == "x" else 1
        vals.extend(
            point[coord] for point in (series.get("values") or [])
            if isinstance(point, (list, tuple)) and len(point) > coord
            and isinstance(point[coord], (int, float)) and not isinstance(point[coord], bool)
            and math.isfinite(point[coord])
        )
    if not vals:
        return None
    lo, hi = min(vals), max(vals)
    glims = _pair(gt_limits)
    if glims and glims[0] <= 0 <= glims[1]:
        lo = min(lo, 0.0)
    return (lo, hi)


_PATH = re.compile(r"axes\[(\d+)\]\.([xy])lim$")


def _automatic_label_policy(policy):
    if not isinstance(policy, dict):
        return False
    locator, formatter = policy.get("locator"), policy.get("formatter")
    if not all(isinstance(component, dict) and component.get("automatic") is True
               for component in (locator, formatter)):
        return False
    # Callable/custom formatter behavior is not captured by the policy snapshot.
    return (str(locator.get("class", "")).startswith("matplotlib.ticker.")
            and formatter.get("class") in {
                "matplotlib.ticker." + name for name in (
                    "ScalarFormatter", "LogFormatter", "LogFormatterExponent",
                    "LogFormatterMathtext", "LogFormatterSciNotation", "LogitFormatter",
                    "EngFormatter", "PercentFormatter", "FormatStrFormatter", "StrMethodFormatter",
                )
            })


def comparators(fp: list, gt_flat: dict, flawed_flat: dict, gt_spec: dict,
                summary_text: str) -> dict:
    """{footprint path: comparator(gt_flat, sub_flat) -> bool} for the paths a
    truncated axis moved. Empty when the rule has nothing to say.

    A limit path only gets a comparator when the FLAWED limits themselves fail
    the rule — the semantic reading replaces float equality exactly where the
    flaw is semantically visible. A degenerate perturbation that still satisfies
    the span or contains all the data (a 3 % widening, say) keeps the strict
    numeric reading, so an agent that changes nothing is never paid for it
    (the F1 principle: no reward for a no-op).

    Different ticklabels can follow repaired limits only when unchanged
    automatic formatter/locator policies establish that they are derived and
    every tick still has a visible label. Fixed/custom labels remain exact.
    Automatically generated tick positions follow the same rule only when
    the unchanged locator policy proves that they are derived.
    """
    from .compare import equal

    out = {}
    for path in fp:
        m = _PATH.search(path)
        if not m:
            continue
        axes_index, letter = int(m.group(1)), m.group(2)
        gt_limits = gt_flat.get(path)
        extent = data_extent(gt_spec, axes_index, letter, gt_limits)
        constraint = axis_constraint(summary_text, axes_index, letter, gt_spec, gt_flat)
        span = constraint["span"]
        if constraint["status"] == "ambiguous" or extent is None:
            continue                       # nothing to judge against: numeric equality stays

        def lim_ok(sub_val, span=span, extent=extent):
            lims = _pair(sub_val)
            if lims is None:
                return False
            if extent is not None and not _contains(lims, extent):
                return False
            slo, shi = lims
            if span is not None:           # case A: the summary pins the span
                lo, hi = span
                slack = 0.10 * (hi - lo)
                return abs(slo - lo) <= slack and abs(shi - hi) <= slack
            return extent is not None      # case B: containment checked above

        if lim_ok(flawed_flat.get(path)):
            continue                       # the flaw passes the rule: degenerate, stay numeric

        def lim_cmp(gt_f, sub_f, path=path, lim_ok=lim_ok):
            return lim_ok(sub_f.get(path))

        out[path] = lim_cmp
        ticks = "axes[%d].%sticklabels" % (axes_index, letter)
        positions = "axes[%d].%sticks" % (axes_index, letter)
        policy_path = "axes[%d].%stick_policy" % (axes_index, letter)
        policy = gt_flat.get(policy_path)
        derived_labels = (_automatic_label_policy(policy)
                          and equal(policy, flawed_flat.get(policy_path)))
        if ticks in fp:
            def ticks_cmp(gt_f, sub_f, path=path, lim_ok=lim_ok,
                          ticks=ticks, positions=positions, policy_path=policy_path,
                          policy=policy, derived_labels=derived_labels):
                labels = sub_f.get(ticks)
                if (equal(gt_f.get(ticks), labels)
                        and equal(gt_f.get(positions), sub_f.get(positions))):
                    return True
                tick_positions = sub_f.get(positions)
                return (derived_labels and lim_ok(sub_f.get(path))
                        and equal(policy, sub_f.get(policy_path))
                        and isinstance(labels, list) and bool(labels)
                        and isinstance(tick_positions, list)
                        and len(labels) == len(tick_positions)
                        and all(isinstance(label, str) and label.strip() for label in labels))
            out[ticks] = ticks_cmp
        locator = (gt_flat.get(policy_path) or {}).get("locator")
        flawed_locator = (flawed_flat.get(policy_path) or {}).get("locator")
        if (positions in fp and isinstance(locator, dict)
                and locator.get("automatic") is True and equal(locator, flawed_locator)):
            def positions_cmp(gt_f, sub_f, path=path, lim_ok=lim_ok,
                              positions=positions, policy_path=policy_path, locator=locator):
                if equal(gt_f.get(positions), sub_f.get(positions)):
                    return True
                submitted_locator = (sub_f.get(policy_path) or {}).get("locator")
                return (bool(sub_f.get(positions)) and equal(locator, submitted_locator)
                        and lim_ok(sub_f.get(path)))
            out[positions] = positions_cmp
    return out
