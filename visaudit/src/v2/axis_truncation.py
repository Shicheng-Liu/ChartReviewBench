"""Target-aware V2 axis truncation that clips real plotted data.

The V1 perturbation merely raised a zero baseline while leaving every value
visible.  V2 instead derives a limit from the captured ground-truth figure and
places at least one real data endpoint outside the visible interval.  It never
touches the data arrays themselves.
"""
from __future__ import annotations

import ast
import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AxisTruncationPlan:
    axis_index: int
    axis: str
    lower: float
    upper: float
    gt_limits: tuple[float, float]
    clipped: tuple[dict[str, Any], ...]

    @property
    def target(self) -> dict[str, Any]:
        return {
            "axis_index": self.axis_index,
            "axis": self.axis,
            "gt_limits": list(self.gt_limits),
            "flawed_limits": [self.lower, self.upper],
            "clipped": [dict(item) for item in self.clipped],
        }

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["target"] = self.target
        return result


def _finite(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _axis_points(axis_spec: dict, axis: str) -> list[dict[str, Any]]:
    points = []
    for series_index, series in enumerate(axis_spec.get("series") or []):
        kind = str(series.get("kind") or "")
        # The extractor represents bars as [category-position, height/width].
        # Horizontal bars carry their x value in the second slot as well.
        if kind == "bar" and axis != "y":
            continue
        if kind == "barh" and axis != "x":
            continue
        if kind in {"bar", "barh"}:
            coordinate_index = 1
        elif kind in {"line", "scatter", "area"}:
            coordinate_index = 0 if axis == "x" else 1
        else:
            continue
        for point_index, pair in enumerate(series.get("values") or []):
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                continue
            value = _finite(pair[coordinate_index])
            if value is None:
                continue
            points.append({
                "series_label": str(series.get("label") or f"series{series_index}"),
                "point_index": point_index,
                "value": value,
            })
    return points


def _candidate(axis_spec: dict, axis_index: int, axis: str):
    if str(axis_spec.get(f"{axis}scale") or "linear") == "log":
        return None
    raw_limits = axis_spec.get(f"{axis}lim")
    if not isinstance(raw_limits, (list, tuple)) or len(raw_limits) != 2:
        return None
    limits = tuple(_finite(v) for v in raw_limits)
    if any(v is None for v in limits) or limits[0] >= limits[1]:
        return None
    points = [p for p in _axis_points(axis_spec, axis)
              if limits[0] <= p["value"] <= limits[1]]
    distinct = sorted({p["value"] for p in points})
    if len(distinct) < 2:
        return None

    candidates = []
    lower_cut = (distinct[0] + distinct[1]) / 2.0
    clipped_low = tuple(p for p in points if p["value"] < lower_cut)
    if clipped_low and lower_cut < limits[1]:
        candidates.append((len(clipped_low), lower_cut, limits[1], clipped_low))

    upper_cut = (distinct[-2] + distinct[-1]) / 2.0
    clipped_high = tuple(p for p in points if p["value"] > upper_cut)
    if clipped_high and upper_cut > limits[0]:
        candidates.append((len(clipped_high), limits[0], upper_cut, clipped_high))
    if not candidates:
        return None

    # Clip the fewest endpoints; prefer the upper side on a tie because its
    # visual interpretation is direct for ordinary vertical bars.
    count, lower, upper, clipped = min(
        candidates,
        key=lambda item: (item[0], 0 if item[1] == limits[0] else 1),
    )
    return count, AxisTruncationPlan(
        axis_index=axis_index,
        axis=axis,
        lower=float(lower),
        upper=float(upper),
        gt_limits=(float(limits[0]), float(limits[1])),
        clipped=clipped,
    )


def plan_axis_truncation(spec: dict) -> AxisTruncationPlan | None:
    """Choose a single-axis data-clipping plan from a GT figure capture.

    Multi-panel figures are declined for now: a one-line global limit can alter
    several panels and would violate the edit-locality contract.  This is an
    applicability restriction, not a weakening of the postcondition.
    """
    axes = (spec or {}).get("axes") or []
    if len(axes) != 1:
        return None
    options = []
    for axis in ("y", "x"):
        candidate = _candidate(axes[0], 0, axis)
        if candidate:
            options.append(candidate)
    if not options:
        return None
    return min(options, key=lambda item: (item[0], 0 if item[1].axis == "y" else 1))[1]


def _call_name(node: ast.Call) -> str:
    parts = []
    current = node.func
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _save_call(tree: ast.AST) -> ast.Call | None:
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and _call_name(node).split(".")[-1] in {"savefig", "write_image"}]
    return max(calls, key=lambda node: (node.lineno, node.col_offset), default=None)


def _plotly_figure_expression(code: str, call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Attribute) and call.func.attr == "write_image":
        return ast.get_source_segment(code, call.func.value)
    if call.args:
        return ast.get_source_segment(code, call.args[0])
    return None


def inject_axis_truncation(
    code: str,
    backend: str,
    plan: AxisTruncationPlan,
) -> str:
    """Insert the one limit statement immediately before image materialization."""
    if plan.axis_index != 0:
        raise ValueError("V2 axis injection currently accepts one-axis figures only")
    tree = ast.parse(code)
    call = _save_call(tree)
    if call is None:
        raise ValueError("script has no savefig/write_image call")
    lines = code.splitlines(keepends=True)
    insertion = sum(len(line) for line in lines[: call.lineno - 1])
    line = lines[call.lineno - 1]
    indent = line[: len(line) - len(line.lstrip(" \t"))]
    lower, upper = f"{plan.lower:.10g}", f"{plan.upper:.10g}"

    if backend == "plotly":
        figure = _plotly_figure_expression(code, call)
        if not figure:
            raise ValueError("cannot identify the Plotly figure written to disk")
        layout_axis = plan.axis + "axis"
        statement = f"{figure}.layout.{layout_axis}.range = [{lower}, {upper}]"
    elif backend == "matplotlib":
        setter = "set_xlim" if plan.axis == "x" else "set_ylim"
        statement = f"plt.gca().{setter}({lower}, {upper})"
    else:
        raise ValueError(f"unsupported backend {backend!r}")
    return code[:insertion] + indent + statement + "\n" + code[insertion:]
