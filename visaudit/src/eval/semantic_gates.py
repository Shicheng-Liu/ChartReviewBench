"""Fail-closed semantic postconditions for every V2 perturbation category.

A non-empty Figure-Object footprint only proves that *something* changed. The
gates here prove the stronger category claim and reject ambiguity, near-misses,
and unrelated semantic changes. Inputs may be nested Figure specs, complete
capture records, or legacy flattened property maps.

``target`` is the generator's structured statement of intent; ``evidence`` is
the untouched table/summary witness. Targets are inferred only where the
Figure Object makes the answer unique. Color contradictions, in particular,
require an explicit summary color mapping.
"""

from __future__ import annotations

import ast
import colorsys
import csv
import hashlib
import io
import itertools
import json
import math
import re
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

CATEGORIES = (
    "axis-truncation",
    "color-contradicts-summary",
    "dropped-series",
    "missing-legend",
    "swapped-legend-labels",
    "swapped-series",
    "value-perturbation",
    "wrong-axis-label",
    "wrong-chart-type",
    "wrong-title",
)

_SERIES_RE = re.compile(r"^axes\[(\d+)\]\.series\[(.*)\]\.([^.]+)$")
_MISSING = object()
_SERIES_FIELDS = (
    "kind", "values", "color", "edgecolor", "colors", "sizes", "clim",
    "color_alpha", "marker", "linestyle", "linewidth", "markersize",
    "markerfacecolor", "markerfacecolor_alpha", "markeredgecolor",
    "markeredgecolor_alpha", "markeredgewidth", "fillstyle", "drawstyle",
    "alpha", "zorder",
)
MIN_VALUE_VISUAL_FRACTION = 0.10
MIN_HEATMAP_COLOR_DISTANCE = 0.10
VALUE_SOURCE_BINDING_SCHEMA = "chartrepairbench.v2.value-source-binding/1"
SERIES_TABLE_BINDING_SCHEMA = "chartrepairbench.v2.series-table-binding/1"
REPAIR_INPUT_RECOVERABILITY_SCHEMA = (
    "chartrepairbench.v2.repair-input-recoverability/3"
)
_SUMMARY_COLOR_WORDS = frozenset({
    "black", "blue", "brown", "coral", "crimson", "cyan", "gold", "gray",
    "green", "grey", "indigo", "magenta", "maroon", "navy", "olive",
    "orange", "pink", "purple", "red", "salmon", "teal", "turquoise",
    "violet", "white", "yellow",
})
_COLOR_VISUAL_NOUNS = frozenset({
    "area", "areas", "bar", "bars", "dot", "dots", "line", "lines",
    "marker", "markers", "trace", "traces",
})
_COLOR_VISUAL_STYLES = frozenset({
    "circular", "connecting", "dashed", "diamond", "dotted", "horizontal",
    "shaped", "solid", "square", "vertical",
})
_COLOR_RELATION_BRIDGES = frozenset({
    "a", "across", "an", "and", "at", "chart", "circles", "circular",
    "connecting", "diamond", "drawn", "graph", "line", "mark", "marked",
    "marker", "markers", "overlaying", "overlays", "plotted", "runs",
    "shaped", "square", "the", "through", "with",
})
_COLOR_ENTITY_BRIDGES = frozenset({
    "a", "an", "average", "baseline", "current", "level", "number",
    "overall", "target", "the", "value",
})
_COLOR_ACTIVE_RELATIONS = frozenset({
    "denote", "denotes", "denoting", "depict", "depicts", "depicting",
    "display", "displays", "displaying", "indicate", "indicates", "indicating",
    "mark", "marks", "marking", "reflect", "reflects", "reflecting",
    "represent", "represents", "representing", "show", "shows", "showing",
    "track", "tracks", "tracking",
})
_COLOR_PASSIVE_RELATIONS = frozenset({
    "denoted", "depicted", "displayed", "indicated", "marked", "reflected",
    "represented", "shown", "tracked",
})
_SEMANTIC_SUFFIXES = (
    ".title", ".xlabel", ".ylabel", ".title_color", ".xlabel_color",
    ".ylabel_color", ".legend.labels", ".legend.title", ".kind", ".values",
    ".color", ".edgecolor", ".colors", ".sizes", ".xscale", ".yscale",
    ".color_alpha", ".marker", ".linestyle", ".linewidth", ".markersize",
    ".markerfacecolor", ".markerfacecolor_alpha", ".markeredgecolor",
    ".markeredgecolor_alpha", ".markeredgewidth", ".fillstyle", ".drawstyle",
    ".alpha", ".zorder",
    ".clim", ".rendered_colors",
    ".xlim", ".ylim", ".xticks", ".yticks", ".xticklabels", ".yticklabels",
    ".xtick_policy", ".ytick_policy",
    ".annotations",
    ".arrow_colors",
)


@dataclass(frozen=True)
class _State:
    flat: dict[str, Any]
    status: str

    @property
    def valid(self) -> bool:
        return self.status == "ok"


def _mapping(value: Any) -> dict:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    for method in ("model_dump", "dict"):
        fn = getattr(value, method, None)
        if callable(fn):
            dumped = fn()
            if isinstance(dumped, Mapping):
                return dict(dumped)
    return {}


def _state(value: dict | None) -> _State:
    raw = _mapping(value)
    is_capture = "flat" in raw or "spec" in raw or (
        "status" in raw and not any("." in str(key) for key in raw)
    )
    status = str(raw.get("status", "ok")) if is_capture else "ok"
    if is_capture and isinstance(raw.get("flat"), Mapping):
        flat = dict(raw["flat"])
    else:
        spec = raw.get("spec") if is_capture else raw
        if isinstance(spec, Mapping) and "axes" in spec:
            from .capture import flatten

            flat = flatten(dict(spec))
        elif is_capture:
            flat = {}
        else:
            flat = raw
    return _State(flat=flat, status=status)


def _metadata(target: Any, evidence: Any) -> tuple[dict, dict]:
    target = _mapping(target)
    if not evidence and isinstance(target.get("evidence"), Mapping):
        evidence = target["evidence"]
    if isinstance(target.get("target"), Mapping):
        outer = target
        target = dict(target["target"])
        evidence = evidence or outer.get("evidence")
    return target, _mapping(evidence)


def _equal(left: Any, right: Any) -> bool:
    if left is _MISSING or right is _MISSING:
        return left is right
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not (math.isfinite(float(left)) and math.isfinite(float(right))):
            return False
        scale = max(abs(float(left)), abs(float(right)), 1.0)
        return abs(float(left) - float(right)) <= scale * 1e-9
    if isinstance(left, str) and isinstance(right, str):
        return _normalized_visible_text(left) == _normalized_visible_text(right)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_equal(left[key], right[key]) for key in left)
    return left == right


def _changed(before: _State, after: _State) -> list[str]:
    return sorted(
        path for path in set(before.flat) | set(after.flat)
        if not _equal(before.flat.get(path, _MISSING), after.flat.get(path, _MISSING))
    )


def _semantic_path(path: str) -> bool:
    return (
        path in {
            "suptitle", "figlegend.labels", "figure_size_inches",
            "figure_dpi", "canvas_size_px",
        }
        or path.endswith(_SEMANTIC_SUFFIXES)
    )


def _collateral(
    changed: list[str], *, exact: tuple[str, ...] = (), prefixes: tuple[str, ...] = (),
    ignored_suffixes: tuple[str, ...] = (), predicate: Callable[[str], bool] | None = None,
    target: Mapping | None = None,
) -> list[str]:
    target = _mapping(target)

    def paths(name):
        value = target.get(name, ())
        value = [value] if isinstance(value, str) else value
        return tuple(str(path) for path in value)

    external_exact = paths("allowed_collateral_paths")
    external_prefixes = paths("allowed_collateral_prefixes")
    collateral = []
    for path in changed:
        if not _semantic_path(path):
            continue
        allowed = path in exact or path in external_exact
        allowed = allowed or any(path.startswith(prefix) for prefix in prefixes + external_prefixes)
        allowed = allowed or any(path.endswith(suffix) for suffix in ignored_suffixes)
        allowed = allowed or bool(predicate and predicate(path))
        if not allowed:
            collateral.append(path)
    return collateral


def _series(state: _State) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path, value in state.flat.items():
        match = _SERIES_RE.match(path)
        if not match:
            continue
        axis, token, field = match.groups()
        root = f"axes[{axis}].series[{token}]"
        occurrence_match = re.fullmatch(r"(.*)#(\d+)", token)
        encoded_label = occurrence_match.group(1) if occurrence_match else token
        occurrence = int(occurrence_match.group(2)) if occurrence_match else None
        label = re.sub(
            r"%(25|23|5[Bb]|5[Dd])",
            lambda found: {
                "25": "%", "23": "#", "5b": "[", "5d": "]",
            }[found.group(1).lower()],
            encoded_label,
        )
        rec = out.setdefault(root, {
            "axis": int(axis), "label": label, "occurrence": occurrence,
            "root": root,
        })
        rec[field] = value
    return out


def _identity(series: dict, *, values: bool = True) -> dict:
    fields = _SERIES_FIELDS if values else tuple(f for f in _SERIES_FIELDS if f != "values")
    return {field: series.get(field, _MISSING) for field in fields}


def _series_key(series: dict[str, dict], label: Any, axis_index: Any = None) -> str | None:
    if label is None:
        return None
    text = str(label).strip()
    matches = [root for root, rec in series.items() if _equal(rec["label"], text)]
    if axis_index is not None:
        axis = _target_index(axis_index)
        if axis is None:
            return None
        matches = [root for root in matches if series[root]["axis"] == axis]
    return matches[0] if len(matches) == 1 else None


def _labels(target: dict) -> list[str]:
    value = target.get("labels") or target.get("series_labels") or target.get("targets")
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    left = target.get("label_a") or target.get("series_a")
    right = target.get("label_b") or target.get("series_b")
    if left is not None and right is not None:
        return [str(left), str(right)]
    single = target.get("series_label") or target.get("label")
    return [str(single)] if single is not None else []


def _legend_groups(state: _State) -> dict[str, tuple[str, ...]]:
    groups = {}
    for path, value in state.flat.items():
        if not path.endswith("legend.labels") or not value:
            continue
        labels = tuple(str(label).strip() for label in value if str(label).strip())
        if labels:
            groups[path] = labels
    return groups


def legend_groups(flat: dict) -> dict[str, tuple[str, ...]]:
    """Legacy public helper: non-empty legend groups keyed by capture path."""
    return _legend_groups(_state(flat))


def _base(
    category: str, before: _State, after: _State, *, checks: dict[str, Any],
    success_reason: str, failures: list[str] | None = None,
    missing_metadata: list[str] | None = None, collateral_paths: list[str] | None = None,
    **detail: Any,
) -> dict:
    failures = list(dict.fromkeys(failures or []))
    missing_metadata = list(dict.fromkeys(missing_metadata or []))
    collateral_paths = sorted(set(collateral_paths or []))
    if not before.valid:
        failures.insert(0, "before_capture_not_ok")
    if not after.valid:
        failures.insert(0, "after_capture_not_ok")
    if missing_metadata:
        reason = "missing_required_metadata"
    elif failures:
        reason = failures[0]
    elif collateral_paths:
        reason = "collateral_semantic_change"
    else:
        reason = success_reason
    result = {
        "name": category.replace("-", "_") + "_v2_postcondition",
        "category": category,
        "schema_version": "semantic-postcondition-v2",
        "ok": not missing_metadata and not failures and not collateral_paths
              and before.valid and after.valid,
        "reason": reason,
        "checks": checks,
        "missing_metadata": missing_metadata,
        "near_misses": failures,
        "changed_paths": _changed(before, after),
        "collateral_paths": collateral_paths,
    }
    result.update(detail)
    return result


def _expected(value: Any, expected: Any) -> bool:
    return expected is None or _equal(value, expected)


def _target_index(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str) and re.fullmatch(r"\+?\d+", value.strip()):
        return int(value)
    return None


def _number_matches(value: Any, expected: Any) -> bool:
    if not isinstance(value, str):
        return _equal(value, expected)
    try:
        parsed = float(value.strip().replace(",", ""))
    except ValueError:
        return False
    return _equal(parsed, expected)


def _table_number(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = float(value.strip().replace(",", ""))
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _capture_sha256(state: _State) -> str | None:
    try:
        payload = json.dumps(
            state.flat,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(payload).hexdigest()


def _normalized_table_text(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return " ".join(value.split()).casefold()


def _full_table_relation(
    rows: list[list[str]], locator_column: int, value_column: int,
    locators: tuple[str, ...], values: tuple[int | float, ...],
) -> bool:
    """Prove a complete, unique locator-to-value relation independent of row order."""
    if len(rows) != len(values) or len(set(map(_normalized_table_text, locators))) != \
            len(locators):
        return False
    row_locators = [_normalized_table_text(row[locator_column]) for row in rows]
    if None in row_locators or len(set(row_locators)) != len(row_locators):
        return False
    expected = {_normalized_table_text(locator) for locator in locators}
    if set(row_locators) != expected:
        return False
    for locator, value in zip(locators, values):
        locator_key = _normalized_table_text(locator)
        matches = [row for row in rows
                   if _normalized_table_text(row[locator_column]) == locator_key]
        if len(matches) != 1 or not _number_matches(matches[0][value_column], value):
            return False
    return True


def _canonical_heatmap(series: Mapping[str, Any], x_count: int, y_count: int) -> bool:
    values = series.get("values")
    if not isinstance(values, (list, tuple)) or len(values) != x_count * y_count:
        return False
    for expected_index, point in enumerate(values):
        if not isinstance(point, (list, tuple)) or len(point) != 2 or \
                _target_index(point[0]) != expected_index or \
                isinstance(point[1], bool) or not isinstance(point[1], (int, float)) or \
                not math.isfinite(float(point[1])):
            return False
    return True


def _walk_ast_paths(node: ast.AST, path: tuple[str | int, ...] = ()):
    """Yield every AST node with its stable field/index path from the root."""
    yield node, path
    for field, value in ast.iter_fields(node):
        if isinstance(value, ast.AST):
            yield from _walk_ast_paths(value, (*path, field))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                if isinstance(child, ast.AST):
                    yield from _walk_ast_paths(child, (*path, field, index))


def value_perturbation_source_binding(
    source_code: str, table_csv: str, gt_capture: Mapping[str, Any] | None,
    target: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Bind one immutable source-list point through the table and GT heatmap."""
    target = _mapping(target)
    label = target.get("series_label") or target.get("label")
    point_index = _target_index(target.get("point_index"))
    truth = target.get("truth_value", target.get("truth"))
    injected = target.get("injected_value", target.get("injected"))
    gt_state = _state(gt_capture)
    base = {
        "schema": VALUE_SOURCE_BINDING_SCHEMA,
        "ok": False,
        "series_label": label,
        "point_index": point_index,
        "truth_value": truth,
        "injected_value": injected,
        "source_code_sha256": (
            hashlib.sha256(source_code.encode()).hexdigest()
            if isinstance(source_code, str) else None
        ),
        "table_sha256": (
            hashlib.sha256(table_csv.encode()).hexdigest()
            if isinstance(table_csv, str) else None
        ),
        "gt_capture_sha256": _capture_sha256(gt_state),
    }

    def failed(reason: str) -> dict[str, Any]:
        return {
            **base,
            "reason": reason,
            "source_series_length": None,
            "locator_key": None,
            "locator_label": None,
            "locator_labels": [],
            "axis_index": None,
            "requested_dimension": None,
            "row": None,
            "column": None,
            "flat_index": None,
            "series_root": None,
            "render_kind": None,
            "source_series_ast_path": None,
            "source_value_ast_path": None,
            "source_value_literal": None,
        }

    if not isinstance(source_code, str) or not isinstance(table_csv, str) or \
            not gt_state.valid or base["gt_capture_sha256"] is None or \
            not isinstance(label, str) or not label.strip() or point_index is None or \
            isinstance(truth, bool) or not isinstance(truth, (int, float)) or \
            not math.isfinite(float(truth)) or isinstance(injected, bool) or \
            not isinstance(injected, (int, float)) or \
            not math.isfinite(float(injected)) or _equal(injected, truth):
        return failed("source_target_metadata_invalid")
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return failed("source_code_not_parseable")

    code_candidates: set[tuple[
        str,
        tuple[str, ...],
        tuple[int | float, ...],
        tuple[str | int, ...],
        tuple[str | int, ...],
    ]] = set()
    for node, node_path in _walk_ast_paths(tree):
        if not isinstance(node, ast.Dict):
            continue
        try:
            mapping = ast.literal_eval(node)
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            continue
        if not isinstance(mapping, dict):
            continue
        matching_key_positions = []
        for position, key_node in enumerate(node.keys):
            try:
                key = ast.literal_eval(key_node)
            except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                continue
            if isinstance(key, str) and _equal(key, label):
                matching_key_positions.append((position, key))
        if len(matching_key_positions) != 1:
            continue
        series_position, series_key = matching_key_positions[0]
        series_node = node.values[series_position]
        series = mapping[series_key]
        if not isinstance(series_node, (ast.List, ast.Tuple)) or \
                not isinstance(series, (list, tuple)) or not series or \
                point_index >= len(series) or not _equal(series[point_index], truth) or any(
                    isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(float(value)) for value in series
                ):
            continue
        for key, locator_values in mapping.items():
            if key == series_key or not isinstance(key, str) or \
                    not isinstance(locator_values, (list, tuple)) or \
                    len(locator_values) != len(series) or any(
                        not isinstance(value, str) or not value.strip()
                        for value in locator_values
                    ):
                continue
            code_candidates.add((
                key,
                tuple(value.strip() for value in locator_values),
                tuple(series),
                (*node_path, "values", series_position),
                (*node_path, "values", series_position, "elts", point_index),
            ))
    if not code_candidates:
        return failed("source_series_binding_not_unique")

    declared_value_headers = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func.attr if isinstance(node.func, ast.Attribute) else \
            node.func.id if isinstance(node.func, ast.Name) else ""
        if function not in {"melt", "pivot", "pivot_table"}:
            continue
        accepted_keywords = {"value_name"} if function == "melt" else {"values"}
        for keyword in node.keywords:
            if keyword.arg in accepted_keywords and isinstance(keyword.value, ast.Constant) \
                    and isinstance(keyword.value.value, str) and keyword.value.value.strip():
                declared_value_headers.add(keyword.value.value.strip())
    if len(declared_value_headers) > 1:
        return failed("source_value_column_not_unique")

    try:
        rows = list(csv.reader(io.StringIO(table_csv)))
    except csv.Error:
        return failed("truth_table_not_parseable")
    if len(rows) < 2 or not rows[0]:
        return failed("truth_table_empty")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        return failed("truth_table_ragged")
    header, data_rows = rows[0], rows[1:]
    normalized_header = [_normalized_table_text(value) for value in header]
    if None in normalized_header or len(set(normalized_header)) != len(header):
        return failed("truth_table_header_ambiguous")
    table_candidates: set[tuple[Any, ...]] = set()

    wide_value_columns = [index for index, value in enumerate(header) if _equal(value, label)]
    if len(wide_value_columns) == 1:
        value_column = wide_value_columns[0]
        for (locator_key, locators, values, series_ast_path,
             value_ast_path) in code_candidates:
            for locator_column in range(width):
                if locator_column == value_column or not _equal(
                    header[locator_column], locator_key
                ):
                    continue
                if _full_table_relation(
                    data_rows, locator_column, value_column, locators, values,
                ):
                    target_rows = [position for position, row in enumerate(data_rows)
                                   if _equal(row[locator_column], locators[point_index])]
                    if len(target_rows) == 1:
                        table_candidates.add((
                            "wide", locator_key, locators, values,
                            series_ast_path, value_ast_path, None,
                            locator_column, value_column, target_rows[0],
                        ))

    for series_column in range(width):
        matching_rows = [row for row in data_rows if _equal(row[series_column], label)]
        if not matching_rows:
            continue
        for (locator_key, locators, values, series_ast_path,
             value_ast_path) in code_candidates:
            for locator_column in range(width):
                if locator_column == series_column or not _equal(
                    header[locator_column], locator_key
                ):
                    continue
                eligible_value_columns = [
                    column for column in range(width)
                    if column not in (series_column, locator_column)
                    and all(_table_number(row[column]) is not None for row in matching_rows)
                    and (
                        not declared_value_headers
                        or any(_equal(header[column], name)
                               for name in declared_value_headers)
                    )
                ]
                if len(eligible_value_columns) != 1:
                    continue
                for value_column in eligible_value_columns:
                    if not \
                            _full_table_relation(
                                matching_rows, locator_column, value_column,
                                locators, values,
                            ):
                        continue
                    target_rows = [
                        position for position, row in enumerate(data_rows)
                        if _equal(row[series_column], label)
                        and _equal(row[locator_column], locators[point_index])
                    ]
                    if len(target_rows) == 1:
                        table_candidates.add((
                            "long", locator_key, locators, values,
                            series_ast_path, value_ast_path, series_column,
                            locator_column, value_column, target_rows[0],
                        ))

    render_candidates: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for projection in table_candidates:
        (layout, locator_key, locators, source_values, source_series_ast_path,
         source_value_ast_path, series_column, locator_column, value_column,
         table_row) = projection
        locator_label = locators[point_index]
        for root, heatmap in _series(gt_state).items():
            if heatmap.get("kind") != "heatmap":
                continue
            axis_root = f"axes[{heatmap['axis']}]"
            xlabels = gt_state.flat.get(f"{axis_root}.xticklabels")
            ylabels = gt_state.flat.get(f"{axis_root}.yticklabels")
            if not isinstance(xlabels, (list, tuple)) or not xlabels or \
                    not isinstance(ylabels, (list, tuple)) or not ylabels or \
                    any(_normalized_table_text(value) is None
                        for value in (*xlabels, *ylabels)) or \
                    len({_normalized_table_text(value) for value in xlabels}) != len(xlabels) or \
                    len({_normalized_table_text(value) for value in ylabels}) != len(ylabels) or \
                    not _canonical_heatmap(heatmap, len(xlabels), len(ylabels)):
                continue
            requested_positions = [
                ("x", position) for position, value in enumerate(xlabels)
                if _equal(value, label)
            ] + [
                ("y", position) for position, value in enumerate(ylabels)
                if _equal(value, label)
            ]
            if len(requested_positions) != 1:
                continue
            dimension, requested_position = requested_positions[0]
            opposite = ylabels if dimension == "x" else xlabels
            if len(opposite) != len(locators):
                continue
            locator_positions = []
            for locator in locators:
                positions = [position for position, value in enumerate(opposite)
                             if _equal(value, locator)]
                if len(positions) != 1:
                    break
                locator_positions.append(positions[0])
            if len(locator_positions) != len(locators) or \
                    len(set(locator_positions)) != len(locator_positions):
                continue
            rendered_values = heatmap["values"]
            coordinates = []
            full_projection_ok = True
            for source_position, locator_position in enumerate(locator_positions):
                row = locator_position if dimension == "x" else requested_position
                column = requested_position if dimension == "x" else locator_position
                flat_index = row * len(xlabels) + column
                if not _equal(rendered_values[flat_index][1], source_values[source_position]):
                    full_projection_ok = False
                    break
                coordinates.append((row, column, flat_index))
            if not full_projection_ok:
                continue
            row, column, flat_index = coordinates[point_index]
            render_candidates.append((projection, {
                "render_kind": "heatmap",
                "axis_index": heatmap["axis"],
                "requested_dimension": dimension,
                "row": row,
                "column": column,
                "flat_index": flat_index,
                "series_root": root,
                "values_path": f"{root}.values",
                "locator_label": locator_label,
                "table_layout": layout,
                "table_row": table_row,
                "series_column": series_column,
                "locator_column": locator_column,
                "value_column": value_column,
            }))

        for root, cartesian in _series(gt_state).items():
            if cartesian.get("kind") not in {"area", "bar", "barh", "line", "scatter"} \
                    or not _equal(cartesian.get("label"), label):
                continue
            values = cartesian.get("values")
            if not isinstance(values, (list, tuple)) or len(values) != len(source_values) \
                    or any(
                        not isinstance(point, (list, tuple)) or len(point) != 2
                        or not _equal(point[1], source_values[position])
                        for position, point in enumerate(values)
                    ):
                continue
            direct_locators = [point[0] for point in values]
            locator_match = all(
                _equal(actual, expected)
                for actual, expected in zip(direct_locators, locators)
            )
            if not locator_match:
                axis_root = f"axes[{cartesian['axis']}]"
                locator_axis = "y" if cartesian.get("kind") == "barh" else "x"
                ticklabels = gt_state.flat.get(f"{axis_root}.{locator_axis}ticklabels")
                locator_match = (
                    isinstance(ticklabels, (list, tuple))
                    and len(ticklabels) == len(locators)
                    and all(_equal(actual, expected)
                            for actual, expected in zip(ticklabels, locators))
                )
            if not locator_match:
                continue
            render_candidates.append((projection, {
                "render_kind": "cartesian",
                "axis_index": cartesian["axis"],
                "requested_dimension": "series",
                "row": point_index,
                "column": 1,
                "flat_index": point_index,
                "series_root": root,
                "values_path": f"{root}.values",
                "locator_label": locator_label,
                "table_layout": layout,
                "table_row": table_row,
                "series_column": series_column,
                "locator_column": locator_column,
                "value_column": value_column,
            }))

    unique_render_candidates = {
        (projection, tuple(sorted(render.items())))
        for projection, render in render_candidates
    }
    if len(unique_render_candidates) != 1:
        return failed("truth_table_binding_not_unique")
    projection, render_items = next(iter(unique_render_candidates))
    render = dict(render_items)
    (layout, locator_key, locators, source_values, source_series_ast_path,
     source_value_ast_path, series_column, locator_column, value_column,
     table_row) = projection
    locator_label = locators[point_index]
    return {
        **base,
        "ok": True,
        "reason": "source_point_bound_to_truth_table_and_gt_render",
        "source_series_length": len(source_values),
        "source_series_ast_path": list(source_series_ast_path),
        "source_value_ast_path": list(source_value_ast_path),
        "source_value_literal": source_values[point_index],
        "locator_key": locator_key,
        "locator_label": locator_label,
        "locator_labels": [locator_label],
        "table_layout": layout,
        "table_coordinate": {
            "row_index": table_row,
            "series_column": (
                header[series_column] if series_column is not None else None
            ),
            "locator_column": header[locator_column],
            "value_column": header[value_column],
        },
        "axis_index": render["axis_index"],
        "render_kind": render["render_kind"],
        "requested_dimension": render["requested_dimension"],
        "row": render["row"],
        "column": render["column"],
        "flat_index": render["flat_index"],
        "series_root": render["series_root"],
        "values_path": render["values_path"],
        "render": {
            key: render[key] for key in (
                "render_kind", "series_root", "row", "column", "flat_index",
                "values_path",
            )
        },
    }


def _axis_label_path(target: Mapping[str, Any]) -> str | None:
    """Map backend-neutral axis identity to the canonical capture path."""
    if target.get("axis") not in ("x", "y") or "axis_index" not in target:
        return None
    raw_axis_index = target["axis_index"]
    if isinstance(raw_axis_index, bool):
        return None
    try:
        axis_index = int(raw_axis_index)
    except (TypeError, ValueError):
        return None
    if isinstance(raw_axis_index, float) and raw_axis_index != axis_index:
        return None
    if axis_index < 0:
        return None
    return f"axes[{axis_index}].{target['axis']}label"


def _normalized_visible_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    normalized = "".join(
        " " if char.isspace() else ""
        if unicodedata.category(char).startswith("C") else char
        for char in normalized
    )
    return " ".join(normalized.split()).casefold()


def _has_renderable_text(value: Any) -> bool:
    return isinstance(value, str) and bool(_normalized_visible_text(value))


def _contains_exact_visible_text(source: Any, truth: Any) -> bool:
    if not isinstance(source, str) or not source.strip() \
            or not _has_renderable_text(truth):
        return False
    needle = _normalized_visible_text(truth)
    if not needle:
        return False
    prefix = r"(?<!\w)" if needle[0].isalnum() or needle[0] == "_" else ""
    suffix = r"(?!\w)" if needle[-1].isalnum() or needle[-1] == "_" else ""
    return re.search(prefix + re.escape(needle) + suffix,
                     _normalized_visible_text(source)) is not None


_CLAIM_BLOCKERS = frozenset({
    "ain", "apparently", "aren", "arguably", "but", "can", "cannot", "cant",
    "could", "couldn", "decoy", "didn", "doesn", "don", "either", "except",
    "false", "fake", "hadn", "hasn", "haven", "isn",
    "former", "formerly", "however", "if", "incorrect", "instead", "may", "might",
    "likely", "maybe", "neither", "never", "no", "not", "or", "perhaps",
    "possibly", "potentially", "previously", "probably", "rather", "should",
    "shouldn", "supposedly", "t", "unless", "unrelated", "used", "was",
    "wasn", "were", "weren", "without", "won", "would", "wouldn", "wrong",
})
_COLOR_RELATION_BLOCKERS = _CLAIM_BLOCKERS | frozenset({
    "alleged", "allegedly", "contrary", "falsely", "historic", "historical",
    "historically", "incorrectly", "old", "originally", "purported",
    "purportedly", "reportedly", "traditional", "traditionally", "untrue",
})
_COLOR_SHADE_MODIFIERS = frozenset({
    "bright", "dark", "deep", "light", "pale",
})
_NEGATION_BLOCKERS = frozenset({
    "ain", "aren", "arguably", "cannot", "cant", "couldn", "decoy", "didn",
    "doesn", "don", "either", "false", "fake", "hadn", "hasn", "haven",
    "incorrect", "isn", "likely", "maybe", "neither",
    "never", "no", "not", "or", "perhaps", "possibly", "potentially",
    "probably", "shouldn", "t", "unrelated", "wasn", "weren", "without",
    "won", "wouldn", "wrong",
})


def _claim_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(re.sub(
        r"[^\w]+", " ", _normalized_visible_text(value), flags=re.UNICODE,
    ).split())


def _summary_segments(summary: Any) -> list[str]:
    if not isinstance(summary, str):
        return []
    return [
        segment.strip()
        for segment in re.split(
            r"[;!?\n]+|(?<=\w)\.(?=\s|$)|"
            r",\s*(?:while|whereas)\s+|\b(?:while|whereas)\b",
            summary,
            flags=re.IGNORECASE,
        )
        if _has_renderable_text(segment)
    ]


def _phrase_spans(tokens: list[str], phrase: list[str]) -> list[tuple[int, int]]:
    if not phrase or len(phrase) > len(tokens):
        return []
    return [
        (index, index + len(phrase))
        for index in range(len(tokens) - len(phrase) + 1)
        if tokens[index:index + len(phrase)] == phrase
    ]


def _direct_color_relation(segment: Any, entity: Any, color: Any) -> bool:
    """Recognize one bounded, current entity-to-color visual relation."""
    text = _claim_text(segment)
    entity_text = _claim_text(entity)
    color_text = _claim_text(color)
    if not text or not entity_text or color_text not in _SUMMARY_COLOR_WORDS:
        return False
    tokens = text.split()
    if set(tokens) & _COLOR_RELATION_BLOCKERS:
        return False
    mentioned_colors = set(tokens) & _SUMMARY_COLOR_WORDS
    if mentioned_colors != {color_text}:
        return False
    entity_tokens = entity_text.split()
    entity_spans = _phrase_spans(tokens, entity_tokens)
    color_positions = [
        index for index, token in enumerate(tokens) if token == color_text
    ]
    if not entity_spans or not color_positions:
        return False
    if any(
        index > 0 and tokens[index - 1] in _COLOR_SHADE_MODIFIERS
        for index in color_positions
    ):
        return False

    escaped_entity = re.escape(entity_text)
    escaped_color = re.escape(color_text)
    mark = r"(?:area|areas|bar|bars|dot|dots|line|lines|marker|markers|trace|traces)"
    simple_patterns = (
        rf"(?:the )?{escaped_entity}(?: {mark})? "
        rf"(?:is|are|appears|appear)(?: shown| displayed| represented| rendered| "
        rf"colou?red)?(?: in| as| by| with)? {escaped_color}",
        rf"(?:the )?{escaped_entity}(?: {mark})? (?:uses|use|has|have) "
        rf"{escaped_color}(?: colou?r)?",
    )
    if any(re.fullmatch(pattern, text) for pattern in simple_patterns):
        return True

    visual_spans: list[tuple[int, int]] = []
    for color_index in color_positions:
        for noun_index, token in enumerate(tokens):
            if token not in _COLOR_VISUAL_NOUNS:
                continue
            start, end = sorted((color_index, noun_index))
            between = tokens[start + 1:end]
            if len(between) <= 4 and all(
                word in _COLOR_VISUAL_STYLES for word in between
            ):
                style_count = len(between)
                while start > 0 and style_count < 4 \
                        and tokens[start - 1] in _COLOR_VISUAL_STYLES:
                    start -= 1
                    style_count += 1
                while end + 1 < len(tokens) and style_count < 4 \
                        and tokens[end + 1] in _COLOR_VISUAL_STYLES:
                    end += 1
                    style_count += 1
                visual_spans.append((start, end + 1))

    for entity_start, entity_end in entity_spans:
        for visual_start, visual_end in visual_spans:
            if entity_end <= visual_start:
                bridge = tokens[entity_end:visual_start]
                bridge_text = " ".join(bridge)
                passive = re.fullmatch(
                    r"(?:is|are) (?:denoted|depicted|displayed|indicated|marked|"
                    r"reflected|represented|shown|tracked) (?:as|by|in|with) "
                    r"(?:a |an |the )?",
                    bridge_text + (" " if bridge_text else ""),
                )
                if passive:
                    return True
            elif visual_end <= entity_start:
                bridge = tokens[visual_end:entity_start]
                relation_positions = [
                    index for index, word in enumerate(bridge)
                    if word in _COLOR_ACTIVE_RELATIONS
                ]
                for relation_index in relation_positions:
                    before = bridge[:relation_index]
                    after = bridge[relation_index + 1:]
                    if len(before) <= 6 \
                            and all(word in _COLOR_RELATION_BRIDGES for word in before) \
                            and len(after) <= 3 \
                            and all(
                                word in _COLOR_ENTITY_BRIDGES or word.isdigit()
                                for word in after
                            ):
                        return True
    return False


def _claim_phrase_pattern(value: Any) -> str | None:
    phrase = _claim_text(value)
    if not phrase:
        return None
    prefix = r"(?<!\w)" if phrase[0].isalnum() or phrase[0] == "_" else ""
    suffix = r"(?!\w)" if phrase[-1].isalnum() or phrase[-1] == "_" else ""
    return prefix + re.escape(phrase) + suffix


def _mask_claim_phrases(text: str, phrases: Any, marker: str) -> str:
    masked = _claim_text(text)
    normalized = sorted(
        {_claim_text(phrase) for phrase in phrases if _claim_text(phrase)},
        key=len,
        reverse=True,
    )
    for phrase in normalized:
        pattern = _claim_phrase_pattern(phrase)
        if pattern:
            masked = re.sub(pattern, marker, masked)
    return masked


def _summary_claim_binding(
    summary: Any,
    truth_phrases: Any,
    *,
    injected_phrases: Any = (),
    subject_phrases: Any = (),
    relation_patterns: tuple[str, ...] = (),
    blockers: frozenset[str] = _CLAIM_BLOCKERS,
    segments: list[str] | None = None,
) -> dict[str, Any]:
    """Bind exact truth text to one clean, whitelisted summary relation."""
    truths = [phrase for phrase in truth_phrases if _has_renderable_text(phrase)]
    injected = [
        phrase for phrase in injected_phrases if _has_renderable_text(phrase)
    ]
    segments = _summary_segments(summary) if segments is None else segments
    truth_segments = [
        segment for segment in segments
        if any(_contains_exact_visible_text(segment, phrase) for phrase in truths)
    ]
    injected_segments = [
        segment for segment in segments
        if any(_contains_exact_visible_text(segment, phrase) for phrase in injected)
    ]
    clean_segments = []
    affirmative_segments = []
    for segment in truth_segments:
        masked = _mask_claim_phrases(segment, truths, "truthclaimtoken")
        masked = _mask_claim_phrases(masked, injected, "injectedclaimtoken")
        masked = _mask_claim_phrases(
            masked, subject_phrases, "subjectclaimtoken",
        )
        words = set(re.findall(r"\b\w+\b", masked))
        if words & blockers:
            continue
        clean_segments.append(segment)
        if not relation_patterns or any(
            re.fullmatch(pattern, masked) for pattern in relation_patterns
        ):
            affirmative_segments.append(segment)
    return {
        "truth_mentioned": bool(truth_segments),
        "injected_absent": not injected_segments,
        "clean_truth_mention": bool(clean_segments),
        "direct_affirmative_relation": bool(affirmative_segments),
        "truth_segments": truth_segments,
        "blocked_truth_segments": [
            segment for segment in truth_segments if segment not in clean_segments
        ],
        "injected_segments": injected_segments,
        "affirmative_segments": affirmative_segments,
    }


_CLAIM_DIMENSION = (
    r"(?:categor(?:y|ies)|channel(?:s)?|countr(?:y|ies)|group(?:s)?|"
    r"month(?:s)?|period(?:s)?|quarter(?:s)?|region(?:s)?|state(?:s)?|"
    r"time|x|y|year(?:s)?)"
)
_CLAIM_DIMENSION_TAIL = (
    rf"(?: (?:across|by|for|over) (?:the )?{_CLAIM_DIMENSION})?"
)
_CHART_KIND_PHRASES = {
    "barh": ("barh", "horizontal bar", "horizontal bar chart",
             "horizontal bar graph"),
    "box": ("box", "box plot", "boxplot"),
    "scatter": ("scatter", "scatter plot", "scatterplot"),
    "line": ("line", "line chart", "line graph"),
    "bar": ("bar", "bar chart", "bar graph", "column chart"),
    "area": ("area", "area chart"),
    "pie": ("pie", "pie chart"),
    "heatmap": ("heatmap", "heat map"),
}


_TITLE_SUBJECT_PATTERN = (
    r"(?:the |this )?(?:chart|image|plot|graph|visualization)"
)
_TITLE_DESCRIPTOR_PATTERN = r"(?:\w+ ){0,7}\w+"
_TITLE_TITLED_SCOPE_PATTERN = (
    rf"{_TITLE_SUBJECT_PATTERN}(?:"
    r" (?:is )?titled"
    rf"| (?:is|presents|shows|displays|depicts) (?:a|an|the) "
    rf"{_TITLE_DESCRIPTOR_PATTERN} titled"
    rf"| presented (?:is|as) (?:a|an|the) "
    rf"{_TITLE_DESCRIPTOR_PATTERN} titled"
    r")"
)
_QUOTED_TITLED_PATTERN = (
    r"\btitled\s+(?:[*_`]+\s*)?[\"“]"
    r"(?P<claim>[^\"”]+)[\"”](?:\s*[*_`]+)?"
)


def _summary_title_segments(summary: Any) -> list[str]:
    if not isinstance(summary, str):
        return []
    sentinels = {".": "\ue000", "!": "\ue001", "?": "\ue002"}

    def protect(match: re.Match[str]) -> str:
        return "".join(sentinels.get(char, char) for char in match.group(0))

    protected = re.sub(r'[\"“][^\"”]*[\"”]', protect, summary)
    restore = {value: key for key, value in sentinels.items()}
    return [
        "".join(restore.get(char, char) for char in segment)
        for segment in _summary_segments(protected)
    ]


def _summary_title_binding_segments(summary: Any) -> list[str]:
    assertions = []
    for segment in _summary_title_segments(summary):
        quoted = re.search(_QUOTED_TITLED_PATTERN, segment, re.IGNORECASE)
        if not quoted:
            assertions.append(segment)
            continue
        tail = segment[quoted.end():].lstrip(" *_`,.")
        competing = bool(
            re.match(
                r"^(?:although|but|despite|instead|or|rather|yet)\b",
                tail,
                re.IGNORECASE,
            )
            or re.search(r"\btitle(?:d)?\b", tail, re.IGNORECASE)
            or re.search(r'[\"“][^\"”]+[\"”]', tail)
        )
        assertions.append(segment if competing else segment[:quoted.end()].strip())
    return assertions


def _summary_title_claims(summary: Any) -> list[dict[str, str]]:
    patterns = (
        r"(?:the |this )?(?:(?:chart|image|plot|graph|visualization) )?title "
        r"(?:is|reads|says|states) (?P<claim>.+)",
        rf"{_TITLE_TITLED_SCOPE_PATTERN} (?P<claim>.+)",
        r"(?P<claim>.+) (?:is|appears as) (?:the |this )?"
        r"(?:(?:chart|image|plot|graph|visualization) )?title",
    )
    claims = []
    for segment in _summary_title_segments(summary):
        visible = _normalized_visible_text(segment)
        quoted = re.search(_QUOTED_TITLED_PATTERN, visible)
        if quoted:
            scope = _claim_text(f"{visible[:quoted.start()]} titled")
            claim = _claim_text(quoted.group("claim"))
            if claim and re.fullmatch(_TITLE_TITLED_SCOPE_PATTERN, scope):
                claims.append({"segment": segment, "claim": claim})
                continue
        text = _claim_text(segment)
        match = next(
            (match for pattern in patterns
             if (match := re.fullmatch(pattern, text))),
            None,
        )
        if match:
            claims.append({"segment": segment, "claim": match.group("claim")})
    return claims


def _legacy_summary_title_claims(summary: Any) -> list[dict[str, str]]:
    """Parse the frozen title-recoverability/1 claim grammar."""
    patterns = (
        r"(?:the |this )?(?:(?:chart|plot|graph|visualization) )?title "
        r"(?:is|reads|says|states) (?P<claim>.+)",
        r"(?:the |this )?(?:chart|plot|graph|visualization) "
        r"(?:is )?titled (?P<claim>.+)",
        r"(?P<claim>.+) (?:is|appears as) (?:the |this )?"
        r"(?:(?:chart|plot|graph|visualization) )?title",
    )
    claims = []
    for segment in _summary_segments(summary):
        text = _claim_text(segment)
        match = next(
            (
                match for pattern in patterns
                if (match := re.fullmatch(pattern, text))
            ),
            None,
        )
        if match:
            claims.append({"segment": segment, "claim": match.group("claim")})
    return claims


def _summary_axis_claims(
    summary: Any, axis: Any, series_labels: list[str] | tuple[str, ...] = (),
) -> list[dict[str, str]]:
    if axis not in {"x", "y"}:
        return []
    patterns = (
        ("explicit_axis", rf"(?:the )?{axis} axis (?:label|title) "
         r"(?:is|reads|says|states) (?P<claim>.+)"),
        ("explicit_axis", rf"(?P<claim>.+) (?:is|appears as) "
         rf"(?:the )?{axis} axis (?:label|title)"),
        ("chart_quantity",
         r"(?:the |this )?(?:chart|plot|graph|visualization) "
         r"(?:compares|displays|measures|plots|presents|reports|shows|tracks) "
         rf"(?P<claim>.+?){_CLAIM_DIMENSION_TAIL}"),
    )
    subjects = sorted({
        _claim_text(label) for label in series_labels if _claim_text(label)
    })
    if subjects:
        subject = "(?:" + "|".join(map(re.escape, subjects)) + ")"
        patterns = (*patterns,
            ("series_quantity",
             rf"{subject}(?: (?:and|versus|vs) {subject})* "
             r"(?:is|are) (?:displayed|measured|plotted|reported|shown) "
             r"(?:across|by|for|on|over|using|with) (?P<claim>.+)"),
        )
    claims = []
    for segment in _summary_segments(summary):
        text = _claim_text(segment)
        matched = next(
            ((scope, match) for scope, pattern in patterns
             if (match := re.fullmatch(pattern, text))),
            None,
        )
        if matched:
            scope, match = matched
            claims.append({
                "segment": segment,
                "claim": match.group("claim"),
                "scope": scope,
            })
    return claims


def _summary_series_claim_groups(summary: Any) -> list[dict[str, Any]]:
    patterns = (
        r"(?:the |this )?(?:chart|plot|graph|visualization) "
        r"(?:compares|displays|includes|plots|presents|reports|shows|tracks) "
        rf"(?P<claims>.+?){_CLAIM_DIMENSION_TAIL}",
        rf"(?P<claims>.+?) (?:is|are) "
        r"(?:displayed|included|plotted|reported|shown|tracked)"
        rf"{_CLAIM_DIMENSION_TAIL}",
        r"(?:legend|key) (?:identifies|lists|maps|names|shows) "
        r"(?P<claims>.+)",
    )
    groups = []
    for segment in _summary_segments(summary):
        text = _claim_text(segment)
        match = next(
            (match for pattern in patterns
             if (match := re.fullmatch(pattern, text))),
            None,
        )
        if not match:
            continue
        claims = [
            claim.strip() for claim in re.split(r"\s+and\s+", match.group("claims"))
            if claim.strip()
        ]
        groups.append({"segment": segment, "claims": claims})
    return groups


def _axis_label_table_binding(
    before: _State,
    target: Mapping[str, Any],
    table: Any,
    label: Any,
) -> dict[str, Any]:
    """Bind one exact axis label to the CSV relation rendered on that axis."""
    base = {
        "schema_version": "chartrepairbench.v2.axis-label-table-binding/1",
        "ok": False,
        "label": label if _has_renderable_text(label) else None,
        "mode": None,
    }
    if not before.valid or not isinstance(table, str) \
            or not _has_renderable_text(label):
        return {**base, "reason": "axis_table_input_invalid"}
    try:
        parsed = list(csv.reader(io.StringIO(table)))
    except csv.Error:
        return {**base, "reason": "axis_table_not_parseable"}
    if len(parsed) < 2 or not parsed[0] or any(
        len(row) != len(parsed[0]) for row in parsed
    ):
        return {**base, "reason": "axis_table_not_rectangular"}
    header, rows = parsed[0], parsed[1:]
    normalized_header = [_normalized_table_text(cell) for cell in header]
    label_key = _normalized_table_text(label)
    columns = [
        index for index, value in enumerate(normalized_header)
        if value == label_key
    ]
    if label_key is None or len(columns) != 1 \
            or None in normalized_header \
            or len(set(normalized_header)) != len(normalized_header):
        return {**base, "reason": "axis_header_not_unique"}
    column = columns[0]
    axis_index = _target_index(target.get("axis_index"))
    axis = target.get("axis")
    if axis_index is None or axis not in {"x", "y"}:
        return {**base, "reason": "axis_identity_invalid", "column": column}

    axis_series = [
        rec for rec in _series(before).values() if rec["axis"] == axis_index
    ]
    cartesian = [
        rec for rec in axis_series
        if rec.get("kind") in {"area", "bar", "barh", "line", "scatter"}
    ]
    value_series = [
        rec for rec in cartesian
        if axis == ("x" if rec.get("kind") == "barh" else "y")
    ]
    locator_series = [rec for rec in cartesian if rec not in value_series]

    def cell_matches(cell: str, expected: Any) -> bool:
        return _number_matches(cell, expected) \
            if isinstance(expected, (int, float)) and not isinstance(expected, bool) \
            else _equal(cell, expected)

    def column_matches_at(
        candidate: int, sequence: list[Any] | tuple[Any, ...],
    ) -> bool:
        return bool(sequence) and len(rows) == len(sequence) and all(
            cell_matches(row[candidate], expected)
            for row, expected in zip(rows, sequence)
        )

    modes = []
    relation_columns: dict[str, list[int]] = {}
    ambiguous_modes = []
    if value_series and len(value_series) == len(cartesian):
        sequences = [_series_value_sequence(rec) for rec in value_series]
        if len(sequences) == 1 and sequences[0] is not None:
            matches = [
                candidate for candidate in range(len(header))
                if column_matches_at(candidate, sequences[0])
            ]
            relation_columns["single_value_series"] = matches
            if matches == [column]:
                modes.append("single_value_series")
            elif column in matches:
                ambiguous_modes.append("single_value_series")
        labels = [rec["label"] for rec in value_series]
        relation = table_capture_series_binding(
            before.flat, table, labels, axis_index=axis_index,
        )
        bindings = relation.get("bindings", [])
        if relation.get("ok") is True and bindings and all(
            binding.get("mode") == "long"
            and _equal(binding.get("value_header"), label)
            for binding in bindings
        ):
            modes.append("complete_long_value_relation")

    if locator_series and len(locator_series) == len(cartesian):
        ticklabels = before.flat.get(
            f"axes[{axis_index}].{axis}ticklabels", _MISSING,
        )
        if isinstance(ticklabels, (list, tuple)):
            matches = [
                candidate for candidate in range(len(header))
                if column_matches_at(candidate, ticklabels)
            ]
            relation_columns["captured_ticklabels"] = matches
            if matches == [column]:
                modes.append("captured_ticklabels")
            elif column in matches:
                ambiguous_modes.append("captured_ticklabels")
        locators = []
        for rec in locator_series:
            values = rec.get("values")
            if not isinstance(values, (list, tuple)) or not values or any(
                not isinstance(point, (list, tuple)) or len(point) != 2
                for point in values
            ):
                locators = []
                break
            locators.append([point[0] for point in values])
        if locators and all(_equal(sequence, locators[0]) for sequence in locators):
            matches = [
                candidate for candidate in range(len(header))
                if column_matches_at(candidate, locators[0])
            ]
            relation_columns["captured_locators"] = matches
            if matches == [column]:
                modes.append("captured_locators")
            elif column in matches:
                ambiguous_modes.append("captured_locators")
            positional = list(range(len(locators[0])))
            positional_columns = []
            if _equal(locators[0], positional):
                for candidate in range(len(header)):
                    cells = [
                        _normalized_table_text(row[candidate]) for row in rows
                    ]
                    if None not in cells \
                            and all(_table_number(row[candidate]) is None for row in rows) \
                            and len(set(cells)) == len(cells) == len(positional):
                        positional_columns.append(candidate)
            relation_columns["positional_categorical_locators"] = \
                positional_columns
            if positional_columns == [column]:
                modes.append("positional_categorical_locators")
            elif column in positional_columns:
                ambiguous_modes.append("positional_categorical_locators")

    if not modes:
        return {
            **base,
            "reason": "axis_table_relation_ambiguous" if ambiguous_modes
            else "axis_table_relation_missing",
            "column": column,
            "candidate_modes": ambiguous_modes,
            "relation_columns": relation_columns,
        }
    return {
        **base,
        "ok": True,
        "reason": "axis_table_relation_bound",
        "mode": modes[0],
        "corroborating_modes": modes,
        "relation_columns": relation_columns,
        "column": column,
        "header": header[column],
    }


def wrong_axis_label_text_recoverability(
    before: dict,
    target=None,
    *,
    summary: str,
    table: str,
    flawed_code: str,
) -> dict[str, Any]:
    """Bind the exact GT label to a clean summary claim or CSV header."""
    before = _state(before)
    target, _ = _metadata(target, None)
    canonical_path = _axis_label_path(target)
    truth_label = before.flat.get(canonical_path, _MISSING) if canonical_path else _MISSING
    target_truth = target.get("truth", target.get("truth_label", _MISSING))
    gt_label_present = _has_renderable_text(truth_label)
    target_truth_present = _has_renderable_text(target_truth)
    truth_matches_target = (
        gt_label_present and target_truth_present and _equal(truth_label, target_truth)
    )
    injected_label = target.get("injected", target.get("injected_label"))
    series_labels = sorted({
        rec["label"] for rec in _series(before).values()
        if _has_renderable_text(rec.get("label"))
    })
    axis_relation_patterns = (
        r"(?:the )?(?:x|y) axis (?:label|title) "
        r"(?:is|reads|says|states) truthclaimtoken",
        r"truthclaimtoken (?:is|appears as) (?:the )?(?:x|y) axis "
        r"(?:label|title)",
        r"(?:the |this )?(?:chart|plot|graph|visualization) "
        r"(?:compares|displays|measures|plots|presents|reports|shows|tracks) "
        rf"truthclaimtoken{_CLAIM_DIMENSION_TAIL}",
        r"subjectclaimtoken(?: (?:and|versus|vs) subjectclaimtoken)* "
        r"(?:is|are) (?:displayed|measured|plotted|reported|shown) "
        r"(?:across|by|for|on|over|using|with) truthclaimtoken",
    )
    summary_binding = _summary_claim_binding(
        summary,
        [truth_label] if gt_label_present else [],
        injected_phrases=[injected_label],
        subject_phrases=series_labels,
        relation_patterns=axis_relation_patterns,
    )
    all_axis_claims = _summary_axis_claims(
        summary, target.get("axis"), series_labels,
    )
    parsed_axis_segments = {claim["segment"] for claim in all_axis_claims}
    unclassified_axis_claims = []
    for segment in _summary_segments(summary):
        text = _claim_text(segment)
        explicit_scope = bool(re.search(
            rf"\b{re.escape(str(target.get('axis')))} axis "
            r"(?:label|title)\b",
            text,
        )) if target.get("axis") in {"x", "y"} else False
        chart_quantity_scope = bool(re.search(
            r"\b(?:chart|graph|plot|visualization)\b.*\b(?:compares|displays|"
            r"measures|plots|presents|reports|shows|tracks)\b",
            text,
        ))
        series_quantity_scope = bool(
            any(_contains_exact_visible_text(segment, label)
                for label in series_labels)
            and re.search(
                r"\b(?:displayed|measured|plotted|reported|shown)\b.*\b"
                r"(?:across|by|for|on|over|using|with)\b",
                text,
            )
        )
        coreferent_axis_scope = bool(
            all_axis_claims
            and re.match(
                r"^(?:it|that axis|the former|the latter|this axis) "
                r"(?:(?:also )?(?:displays|is|measures|plots|presents|reads|"
                r"reports|says|shows|states|tracks)(?: also)?\b)",
                text,
            )
        )
        if (
            explicit_scope or chart_quantity_scope or series_quantity_scope
            or coreferent_axis_scope
        ) and segment not in parsed_axis_segments:
            unclassified_axis_claims.append(segment)
    truth_claim_text = _claim_text(truth_label) if gt_label_present else ""
    active_axis_scopes = {"explicit_axis"}
    active_axis_scopes.update(
        claim["scope"] for claim in all_axis_claims
        if truth_claim_text and claim["claim"] == truth_claim_text
    )
    axis_claims = [
        claim for claim in all_axis_claims
        if claim["scope"] in active_axis_scopes
    ]
    distinct_axis_claims = sorted({claim["claim"] for claim in axis_claims})
    scoped_axis_claims_consistent = bool(
        not distinct_axis_claims
        or (
            gt_label_present
            and distinct_axis_claims == [_claim_text(truth_label)]
        )
    )
    summary_relation = summary_binding["direct_affirmative_relation"]
    truth_table_binding = _axis_label_table_binding(
        before, target, table, truth_label,
    )
    injected_table_binding = _axis_label_table_binding(
        before, target, table, injected_label,
    )
    table_relation = truth_table_binding["ok"] is True
    injected_table_conflict = injected_table_binding["ok"] is True
    summary_conflict_free = bool(
        summary_binding["injected_absent"]
        and (
            not summary_binding["truth_mentioned"]
            or summary_binding["clean_truth_mention"]
        )
    )
    visible_matches = {
        "summary": summary_relation,
        "table": table_relation,
        "flawed_code": False,
    }
    input_conflict_free = bool(
        summary_conflict_free
        and scoped_axis_claims_consistent
        and not injected_table_conflict
    )
    truth_agent_visible = input_conflict_free and any(visible_matches.values())
    checks = {
        "before_capture_ok": before.valid,
        "canonical_axis_identity": canonical_path is not None,
        "gt_label_present": gt_label_present,
        "target_truth_present": target_truth_present,
        "target_truth_matches_gt": truth_matches_target,
        "summary_truth_claim_not_conflicting": summary_conflict_free,
        "axis_claim_segments_classified": not unclassified_axis_claims,
        "scoped_axis_claims_consistent": scoped_axis_claims_consistent,
        "injected_table_relation_not_competing": not injected_table_conflict,
        "truth_label_agent_visible": truth_agent_visible,
    }
    ok = all(checks.values())
    if not before.valid:
        reason = "before_capture_not_ok"
    elif canonical_path is None:
        reason = "canonical_axis_identity_missing"
    elif not gt_label_present:
        reason = "truth_label_missing_from_gt"
    elif not target_truth_present:
        reason = "target_truth_missing"
    elif not truth_matches_target:
        reason = "target_truth_mismatch"
    elif not summary_conflict_free:
        reason = "summary_truth_claim_conflicting"
    elif unclassified_axis_claims:
        reason = "axis_claim_segments_unclassified"
    elif not scoped_axis_claims_consistent:
        reason = "scoped_axis_claims_competing"
    elif injected_table_conflict:
        reason = "injected_table_relation_competing"
    elif not truth_agent_visible:
        reason = "truth_label_not_agent_visible"
    else:
        reason = "truth_label_agent_visible"
    return {
        "name": "wrong_axis_label_text_recoverability",
        "category": "wrong-axis-label",
        "schema_version": "text-recoverability-v1",
        "ok": ok,
        "reason": reason,
        "checks": checks,
        "canonical_path": canonical_path,
        "reported_path": target.get("path"),
        "truth_label": truth_label if gt_label_present else None,
        "summary_claim_binding": summary_binding,
        "summary_axis_claims": axis_claims,
        "all_summary_axis_claims": all_axis_claims,
        "unclassified_summary_axis_claims": unclassified_axis_claims,
        "truth_table_binding": truth_table_binding,
        "injected_table_binding": injected_table_binding,
        "visible_source_matches": visible_matches,
        "visible_sources": [
            name for name, matched in visible_matches.items() if matched
        ],
    }


def _color_summary_evidence_binding(
    summary: str,
    target: Mapping[str, Any] | None,
    evidence: Mapping[str, Any] | None,
    *,
    before: Mapping[str, Any] | None = None,
    legacy_relation: bool,
) -> dict[str, Any]:
    target = _mapping(target)
    evidence = _mapping(evidence)
    entity = target.get("series_label") or target.get("entity") or target.get("label")
    truth_color = target.get("summary_color", target.get("truth_color"))
    quote = evidence.get("summary_quote")
    mapping = evidence.get("summary_color_mapping")

    def normalized(value: Any) -> str:
        return " ".join(
            re.sub(r"[^\w]+", " ", str(value).casefold().replace("_", " ")).split()
        )

    quote_text = quote.strip() if isinstance(quote, str) else ""
    quote_normalized = normalized(quote_text)
    entity_normalized = normalized(entity) if isinstance(entity, str) else ""
    color_normalized = normalized(truth_color) \
        if isinstance(truth_color, str) else ""

    def names_phrase(needle: str) -> bool:
        return bool(
            needle
            and f" {needle} " in f" {quote_normalized} "
        )

    exact_mapping = bool(
        isinstance(mapping, Mapping)
        and isinstance(entity, str)
        and isinstance(truth_color, str)
        and dict(mapping) == {entity: truth_color}
    )
    before_state = _state(before) if before is not None else None
    before_series = _series(before_state) if before_state is not None else {}
    known_entities = {
        rec["label"] for rec in before_series.values()
        if _has_renderable_text(rec.get("label"))
    }
    if before_state is not None:
        known_entities.update(
            label for labels in _legend_groups(before_state).values()
            for label in labels if _has_renderable_text(label)
        )
    mentioned_entities = sorted(
        item for item in known_entities
        if _contains_exact_visible_text(quote_text, item)
    )
    quote_words = set(re.findall(r"\b[\w]+\b", quote_normalized))
    mentioned_colors = sorted(quote_words & _SUMMARY_COLOR_WORDS)
    truth_color_word = color_normalized if color_normalized in _SUMMARY_COLOR_WORDS \
        else None
    target_root = _series_key(before_series, entity) if before_state is not None else None
    gt_color_matches = None
    if before_state is not None:
        color_values = [
            before_series[target_root].get(field)
            for field in ("color", "colors", "edgecolor")
            if target_root and field in before_series[target_root]
        ]
        gt_color_matches = bool(
            color_values
            and all(_color_matches_claim(value, truth_color) for value in color_values)
        )
    if legacy_relation:
        relation_blockers = {
            "but", "except", "formerly", "however", "instead", "never", "no",
            "not", "now", "previously", "used", "was", "were", "without",
        }
        entity_pattern = re.escape(entity_normalized)
        color_pattern = re.escape(color_normalized)
        mark = r"(?:series|line|lines|bar|bars|marker|markers|trace|traces)"
        direct_patterns = (
            rf"(?:the )?{entity_pattern}(?: {mark})? "
            rf"(?:is|are|appears|appear)(?: shown| displayed| represented| rendered| "
            rf"colou?red)?(?: in| as| by| with)? {color_pattern}",
            rf"(?:the )?{entity_pattern}(?: {mark})? (?:uses|use|has|have) "
            rf"{color_pattern}(?: colou?r)?",
            rf"(?:the )?{color_pattern} {mark} "
            rf"(?:shows|show|represents|represent|denotes|denote|corresponds to) "
            rf"(?:the )?{entity_pattern}",
        )
        direct_relation = bool(
            not (quote_words & relation_blockers)
            and any(re.fullmatch(pattern, quote_normalized) for pattern in direct_patterns)
        )
    else:
        direct_relation = _direct_color_relation(
            quote_text, entity_normalized, color_normalized,
        )
    summary_color_claims = []
    blocked_summary_color_claims = []
    unclassified_summary_color_claims = []
    if before_state is not None and isinstance(entity, str):
        legacy_full_summary_patterns = (
            r"(?:the )?subjectclaimtoken"
            r"(?: series| line| lines| bar| bars| marker| markers| trace| traces)? "
            r"(?:appears|appear|is|are)(?: shown| displayed| represented| rendered|"
            r" coloured| colored)?(?: in| as| by| with)? truthclaimtoken",
            r"(?:the )?subjectclaimtoken"
            r"(?: series| line| lines| bar| bars| marker| markers| trace| traces)? "
            r"(?:uses|use|has|have) truthclaimtoken(?: colour| color)?",
            r"(?:the )?truthclaimtoken "
            r"(?:series|line|lines|bar|bars|marker|markers|trace|traces) "
            r"(?:shows|show|represents|represent|denotes|denote) "
            r"(?:the )?subjectclaimtoken",
        )
        for segment in _summary_segments(summary):
            segment_text = _claim_text(segment)
            coreferent_scope = bool(re.match(
                r"^(?:it|that series|the former|the latter|this series)\b",
                segment_text,
            ))
            if not (
                _contains_exact_visible_text(segment, entity)
                or coreferent_scope
            ):
                continue
            for color in sorted(_SUMMARY_COLOR_WORDS):
                if not _contains_exact_visible_text(segment, color):
                    continue
                if legacy_relation:
                    binding = _summary_claim_binding(
                        segment,
                        [color],
                        subject_phrases=[entity],
                        relation_patterns=legacy_full_summary_patterns,
                    )
                    direct_claim = binding["direct_affirmative_relation"]
                    blocked_claim = bool(binding["blocked_truth_segments"])
                else:
                    direct_claim = _direct_color_relation(segment, entity, color)
                    blocked_claim = bool(
                        set(segment_text.split()) & _COLOR_RELATION_BLOCKERS
                    )
                if direct_claim:
                    summary_color_claims.append({
                        "segment": segment,
                        "color": color,
                    })
                elif blocked_claim:
                    blocked_summary_color_claims.append({
                        "segment": segment,
                        "color": color,
                    })
                else:
                    unclassified_summary_color_claims.append({
                        "segment": segment,
                        "color": color,
                    })
    distinct_summary_colors = sorted({
        claim["color"] for claim in summary_color_claims
    })
    checks = {
        "summary_present": isinstance(summary, str) and bool(summary.strip()),
        "entity_present": bool(entity_normalized),
        "truth_color_present": bool(color_normalized),
        "mapping_exact": exact_mapping,
        "quote_present": bool(quote_text) and len(quote_text) <= 500,
        "quote_verbatim": bool(
            quote_text
            and isinstance(summary, str)
            and quote_text in summary
        ),
        "quote_names_entity": bool(
            names_phrase(entity_normalized)
        ),
        "quote_names_truth_color": bool(
            names_phrase(color_normalized)
        ),
    }
    if before_state is not None:
        checks.update({
            "before_capture_ok": before_state.valid,
            "target_entity_unique_in_gt": target_root is not None,
            "gt_color_matches_truth": gt_color_matches is True,
            "quote_names_only_target_entity": mentioned_entities == [entity],
            "quote_names_only_truth_color": bool(
                truth_color_word
                and mentioned_colors == [truth_color_word]
            ),
            "quote_is_direct_affirmative_relation": direct_relation,
            "full_summary_has_one_target_truth_color": bool(
                truth_color_word
                and distinct_summary_colors == [truth_color_word]
            ),
            "full_summary_target_color_claims_unblocked": not (
                blocked_summary_color_claims
            ),
            "full_summary_target_color_claims_classified": not (
                unclassified_summary_color_claims
            ),
        })
    ok = all(checks.values())
    reason = next(
        (
            name for name, passed in checks.items() if not passed
        ),
        "summary_color_claim_bound",
    )
    return {
        "schema_version": (
            "chartrepairbench.v2.color-summary-binding/1"
            if legacy_relation
            else "chartrepairbench.v2.color-summary-binding/2"
        ),
        "category": "color-contradicts-summary",
        "ok": ok,
        "reason": reason,
        "checks": checks,
        "entity": entity,
        "truth_color": truth_color,
        "summary_quote": quote_text or None,
        "mentioned_entities": mentioned_entities,
        "mentioned_colors": mentioned_colors,
        "full_summary_target_color_claims": summary_color_claims,
        "blocked_summary_target_color_claims": blocked_summary_color_claims,
        "unclassified_summary_target_color_claims": (
            unclassified_summary_color_claims
        ),
        "summary_sha256": (
            hashlib.sha256(summary.encode("utf-8")).hexdigest()
            if isinstance(summary, str) else None
        ),
    }


def color_summary_evidence_binding(
    summary: str,
    target: Mapping[str, Any] | None,
    evidence: Mapping[str, Any] | None,
    *,
    before: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind one exact summary quote to the claimed entity-to-color mapping."""
    return _color_summary_evidence_binding(
        summary, target, evidence, before=before, legacy_relation=False,
    )


def _legacy_color_summary_evidence_binding(
    summary: str,
    target: Mapping[str, Any] | None,
    evidence: Mapping[str, Any] | None,
    *,
    before: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute the frozen color-summary-binding/1 contract."""
    return _color_summary_evidence_binding(
        summary, target, evidence, before=before, legacy_relation=True,
    )


_ACTIVE_SERIES_CALLS = frozenset({
    "Bar", "Barpolar", "Box", "Funnel", "Heatmap", "Histogram", "Pie",
    "Scatter", "Scattergl", "Scatterpolar", "Violin", "Waterfall",
    "add_bar", "add_box", "add_scatter", "add_trace", "area", "bar",
    "barh", "barplot", "box", "boxplot", "errorbar", "fill_between",
    "hist", "line", "lineplot", "plot", "scatter", "scatterplot", "stem",
    "step", "stripplot", "swarmplot", "violin", "violinplot",
})
_ACTIVE_CALL_KINDS = {
    "Bar": "bar", "Barpolar": "bar", "Box": "box", "Heatmap": "heatmap",
    "Histogram": "bar", "Pie": "pie", "Violin": "box", "Waterfall": "bar",
    "add_bar": "bar", "add_box": "box", "area": "area", "bar": "bar",
    "barh": "barh", "barplot": "bar", "box": "box", "boxplot": "box",
    "errorbar": "line", "fill_between": "area", "hist": "bar",
    "line": "line", "lineplot": "line", "plot": "line", "scatter": "scatter",
    "scatterplot": "scatter", "stem": "line", "step": "line",
    "stripplot": "scatter", "swarmplot": "scatter", "violin": "box",
    "violinplot": "box",
}


def _definitely_executed_top_level_calls(code: Any) -> list[ast.Call]:
    """Conservatively collect calls in unconditional module-level statements."""
    if not isinstance(code, str) or not code.strip():
        return []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    roots = []
    for statement in tree.body:
        if isinstance(statement, ast.Expr):
            roots.append(statement.value)
        elif isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            roots.append(statement.value)

    calls = []

    def visit(node: ast.AST | None) -> None:
        if node is None or isinstance(node, (ast.Lambda, ast.FunctionDef,
                                             ast.AsyncFunctionDef, ast.ClassDef,
                                             ast.BoolOp, ast.IfExp, ast.ListComp,
                                             ast.SetComp, ast.DictComp,
                                             ast.GeneratorExp)):
            return
        if isinstance(node, ast.Call):
            calls.append(node)
        for child in ast.iter_child_nodes(node):
            visit(child)

    for root in roots:
        visit(root)
    return calls


def _active_series_label_literals(code: Any) -> list[str]:
    """Return literal label/name values attached to actual draw calls."""
    labels = []
    for node in _definitely_executed_top_level_calls(code):
        function = node.func.attr if isinstance(node.func, ast.Attribute) else \
            node.func.id if isinstance(node.func, ast.Name) else ""
        if function not in _ACTIVE_SERIES_CALLS:
            continue
        for keyword in node.keywords:
            if keyword.arg not in {"label", "name"}:
                continue
            if isinstance(keyword.value, ast.Constant) \
                    and isinstance(keyword.value.value, str) \
                    and _has_renderable_text(keyword.value.value):
                labels.append(keyword.value.value)
    return labels


def _active_labeled_kind_calls(code: Any) -> list[dict[str, str]]:
    calls = []
    for node in _definitely_executed_top_level_calls(code):
        function = node.func.attr if isinstance(node.func, ast.Attribute) else \
            node.func.id if isinstance(node.func, ast.Name) else ""
        kind = _ACTIVE_CALL_KINDS.get(function)
        if function == "plot":
            declared = next((
                keyword.value.value for keyword in node.keywords
                if keyword.arg == "kind"
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
            ), None)
            if declared is not None:
                kind = {
                    "area": "area", "bar": "bar", "barh": "barh",
                    "box": "box", "line": "line", "pie": "pie",
                    "scatter": "scatter",
                }.get(declared.casefold())
        if function in {"Scatter", "Scattergl", "Scatterpolar", "add_scatter"}:
            mode = next((
                keyword.value.value for keyword in node.keywords
                if keyword.arg == "mode"
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
            ), "")
            kind = "line" if "lines" in mode.casefold() else "scatter"
        if kind is None:
            continue
        for keyword in node.keywords:
            if keyword.arg not in {"label", "name"} \
                    or not isinstance(keyword.value, ast.Constant) \
                    or not isinstance(keyword.value.value, str) \
                    or not _has_renderable_text(keyword.value.value):
                continue
            calls.append({
                "function": function,
                "kind": kind,
                "label": keyword.value.value,
            })
    return calls


def _summary_series_claim(
    summary: Any, label: str, all_labels: list[str],
) -> dict[str, Any]:
    entity = r"(?:truthclaimtoken|subjectclaimtoken)"
    patterns = (
        r"(?:the |this )?(?:chart|plot|graph|visualization) "
        r"(?:compares|displays|includes|plots|presents|reports|shows|tracks) "
        rf"{entity}(?: and {entity})*{_CLAIM_DIMENSION_TAIL}",
        r"(?:legend|key) (?:identifies|lists|maps|names|shows) "
        rf"{entity}(?: and {entity})*",
        rf"{entity}(?: and {entity})* (?:is|are) "
        r"(?:displayed|included|plotted|reported|shown|tracked)"
        rf"{_CLAIM_DIMENSION_TAIL}",
    )
    return _summary_claim_binding(
        summary,
        [label],
        subject_phrases=[other for other in all_labels if not _equal(other, label)],
        relation_patterns=patterns,
        blockers=_NEGATION_BLOCKERS,
    )


def missing_legend_text_recoverability(
    before: Mapping[str, Any] | None,
    target: Mapping[str, Any] | None,
    *,
    summary: str,
    table: str,
    flawed_code: str,
) -> dict[str, Any]:
    """Bind every removed legend label to a surviving structural source."""
    state = _state(before)
    target = _mapping(target)
    path = target.get("path")
    valid_path = bool(
        isinstance(path, str)
        and (path == "figlegend.labels" or re.fullmatch(
            r"axes\[\d+\]\.legend\.labels", path,
        ))
    )
    truth_labels = (
        list(_legend_groups(state).get(path, ())) if valid_path else []
    )
    target_labels = target.get("labels")
    target_labels = (
        [str(label).strip() for label in target_labels]
        if isinstance(target_labels, list)
        and all(isinstance(label, str) and label.strip() for label in target_labels)
        else []
    )
    code_labels = _active_series_label_literals(flawed_code)
    legend_axis_match = re.fullmatch(
        r"axes\[(\d+)\]\.legend\.labels", path,
    ) if isinstance(path, str) else None
    legend_axis = int(legend_axis_match.group(1)) if legend_axis_match else None
    captured_series = _series(state)
    captured_axis_labels = [
        record["label"] for record in captured_series.values()
        if legend_axis is None or record["axis"] == legend_axis
    ]
    captured_label_roots = {
        label: _series_key(captured_series, label, legend_axis)
        for label in truth_labels
    }
    table_binding = table_capture_series_binding(
        before, table, truth_labels, axis_index=target.get("axis_index"),
    )
    table_bindings = table_binding.get("bindings", [])
    table_positions = [
        binding.get("value_column") for binding in table_bindings
    ]
    ordered_truth_table_bound = bool(
        table_binding.get("ok") is True
        and len(table_bindings) == len(truth_labels)
        and all(binding.get("mode") == "wide" for binding in table_bindings)
        and all(isinstance(position, int) for position in table_positions)
        and table_positions == sorted(table_positions)
    )
    table_bindable_axis_labels = [
        label for label in captured_axis_labels
        if table_capture_series_binding(
            before, table, [label], axis_index=target.get("axis_index"),
        ).get("ok") is True
    ]
    table_group_exhaustive = bool(
        len(captured_axis_labels) == len(truth_labels)
        and len(table_bindable_axis_labels) == len(truth_labels)
        and sorted(map(_normalized_visible_text, table_bindable_axis_labels))
        == sorted(map(_normalized_visible_text, truth_labels))
    )
    table_bound = ordered_truth_table_bound and table_group_exhaustive
    active_code_label_roots = [
        _series_key(captured_series, value) for value in code_labels
    ]
    all_active_code_labels_captured = all(
        root is not None for root in active_code_label_roots
    )
    target_axis_code_labels = [
        value for value, root in zip(code_labels, active_code_label_roots)
        if root is not None
        and (legend_axis is None or captured_series[root]["axis"] == legend_axis)
    ]
    code_group_order_bound = bool(
        all_active_code_labels_captured
        and len(target_axis_code_labels) == len(truth_labels)
        and all(_equal(actual, expected) for actual, expected in zip(
            target_axis_code_labels, truth_labels,
        ))
    )
    summary_bindings = {
        label: _summary_series_claim(summary, label, truth_labels)
        for label in truth_labels
    }
    visible_source_matches = {
        label: {
            # Names alone do not bind legend order/geometry. Summary claims are
            # conflict checks only; reconstruction needs code or table structure.
            "summary": False,
            "table": table_bound,
            "flawed_code": bool(
                code_group_order_bound
                and captured_label_roots[label] is not None
                and any(_equal(value, label) for value in code_labels)
            ),
        }
        for label in truth_labels
    }
    invisible_labels = [
        label for label, matches in visible_source_matches.items()
        if not any(matches.values())
    ]
    summary_conflicts = [
        label for label, binding in summary_bindings.items()
        if binding["blocked_truth_segments"]
    ]
    summary_groups = _summary_series_claim_groups(summary)
    checks = {
        "before_capture_ok": state.valid,
        "canonical_legend_path": valid_path,
        "truth_group_useful": len(truth_labels) >= 2,
        "target_labels_present": len(target_labels) >= 2,
        "target_labels_match_truth_group": bool(
            len(target_labels) >= 2
            and set(target_labels) <= set(truth_labels)
            and len(target_labels) == len(set(target_labels))
        ),
        "all_truth_labels_agent_visible": bool(truth_labels)
        and not invisible_labels,
        "summary_series_claims_not_conflicting": not summary_conflicts,
    }
    ok = all(checks.values())
    reason = next(
        (name for name, passed in checks.items() if not passed),
        "legend_labels_agent_visible",
    )
    return {
        "schema_version": "chartrepairbench.v2.legend-recoverability/1",
        "category": "missing-legend",
        "ok": ok,
        "reason": reason,
        "checks": checks,
        "canonical_path": path if valid_path else None,
        "truth_labels": truth_labels,
        "target_labels": target_labels,
        "visible_source_matches": visible_source_matches,
        "invisible_labels": invisible_labels,
        "summary_conflicts": summary_conflicts,
        "summary_claim_bindings": summary_bindings,
        "summary_claim_groups": summary_groups,
        "active_code_labels": code_labels,
        "active_code_label_roots": active_code_label_roots,
        "all_active_code_labels_captured": all_active_code_labels_captured,
        "relevant_active_code_labels": target_axis_code_labels,
        "captured_axis_labels": captured_axis_labels,
        "code_group_order_bound": code_group_order_bound,
        "captured_code_label_roots": captured_label_roots,
        "table_group_order_bound": table_bound,
        "table_group_ordered": ordered_truth_table_bound,
        "table_group_exhaustive": table_group_exhaustive,
        "captured_axis_series_count_matches": bool(
            len(captured_axis_labels) == len(truth_labels)
        ),
        "table_bindable_axis_labels": table_bindable_axis_labels,
        "series_table_binding": table_binding,
    }


def _series_value_sequence(series: Mapping[str, Any]) -> list[int | float] | None:
    values = series.get("values")
    if not isinstance(values, (list, tuple)) or not values:
        return None
    component = 1
    result = []
    for point in values:
        if not isinstance(point, (list, tuple)) or len(point) <= component:
            return None
        value = point[component]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not \
                math.isfinite(float(value)):
            return None
        result.append(value)
    return result


def table_capture_series_binding(
    before: Mapping[str, Any] | None,
    table: str,
    labels: list[str] | tuple[str, ...],
    *,
    axis_index: int | None = None,
) -> dict[str, Any]:
    """Bind complete captured series to unique physical CSV relations."""
    state = _state(before)
    requested = [str(label).strip() for label in labels]
    base = {
        "schema_version": SERIES_TABLE_BINDING_SCHEMA,
        "ok": False,
        "labels": requested,
        "table_sha256": hashlib.sha256(table.encode()).hexdigest()
        if isinstance(table, str) else None,
        "gt_capture_sha256": _capture_sha256(state),
        "bindings": [],
    }

    def failed(reason: str, bindings=None) -> dict[str, Any]:
        return {**base, "reason": reason, "bindings": bindings or []}

    if not state.valid or not isinstance(table, str) or not requested or any(
        not label for label in requested
    ) or len({_normalized_visible_text(label) for label in requested}) != len(requested):
        return failed("series_table_target_invalid")
    try:
        parsed = list(csv.reader(io.StringIO(table)))
    except csv.Error:
        return failed("table_not_parseable")
    if len(parsed) < 2 or not parsed[0] or any(
        len(row) != len(parsed[0]) for row in parsed
    ):
        return failed("table_not_rectangular")
    header, rows = parsed[0], parsed[1:]
    normalized_header = [_normalized_table_text(cell) for cell in header]
    if None in normalized_header or len(set(normalized_header)) != len(header):
        return failed("table_headers_not_unique")

    series = _series(state)
    bindings = []
    for label in requested:
        root = _series_key(series, label, axis_index)
        if root is None:
            return failed("target_series_not_unique_in_gt", bindings)
        sequence = _series_value_sequence(series[root])
        if sequence is None:
            return failed("target_series_values_not_canonical", bindings)
        candidates = []
        label_key = _normalized_table_text(label)
        for value_column, column_name in enumerate(normalized_header):
            if column_name == label_key and len(rows) == len(sequence) and all(
                _number_matches(row[value_column], value)
                for row, value in zip(rows, sequence)
            ):
                candidates.append({
                    "mode": "wide",
                    "series_root": root,
                    "label": label,
                    "label_column": None,
                    "value_column": value_column,
                    "value_header": header[value_column],
                    "row_indices": list(range(len(rows))),
                    "values": sequence,
                })
        for label_column in range(len(header)):
            selected = [
                (index, row) for index, row in enumerate(rows)
                if _normalized_table_text(row[label_column]) == label_key
            ]
            if len(selected) != len(sequence):
                continue
            for value_column in range(len(header)):
                if value_column == label_column or not all(
                    _number_matches(row[value_column], value)
                    for (_, row), value in zip(selected, sequence)
                ):
                    continue
                candidates.append({
                    "mode": "long",
                    "series_root": root,
                    "label": label,
                    "label_column": label_column,
                    "label_header": header[label_column],
                    "value_column": value_column,
                    "value_header": header[value_column],
                    "row_indices": [index for index, _ in selected],
                    "values": sequence,
                })
        unique = {
            json.dumps(candidate, sort_keys=True, separators=(",", ":")):
            candidate for candidate in candidates
        }
        if len(unique) != 1:
            return failed(
                "series_table_relation_missing" if not unique
                else "series_table_relation_ambiguous",
                bindings,
            )
        bindings.append(next(iter(unique.values())))
    return {**base, "ok": True, "reason": "series_table_relations_bound",
            "bindings": bindings}


def _title_recoverability(
    before: Mapping[str, Any] | None,
    target: Mapping[str, Any],
    summary: str,
) -> dict[str, Any]:
    state = _state(before)
    path = target.get("path")
    valid_path = isinstance(path, str) and bool(
        path == "suptitle" or re.fullmatch(r"axes\[\d+\]\.title", path)
    )
    truth = state.flat.get(path, _MISSING) if valid_path else _MISSING
    target_truth = target.get("truth", target.get("truth_title", _MISSING))
    injected = target.get("injected", target.get("injected_title", _MISSING))
    truth_present = _has_renderable_text(truth)
    title_binding = _summary_claim_binding(
        summary,
        [truth] if truth_present else [],
        injected_phrases=[injected],
        relation_patterns=(
            r"(?:the |this )?"
            r"(?:(?:chart|image|plot|graph|visualization) )?title "
            r"(?:is|reads|says|states) truthclaimtoken(?: |$)",
            rf"{_TITLE_TITLED_SCOPE_PATTERN} "
            r"truthclaimtoken(?: .+)?",
            r"truthclaimtoken (?:is|appears as) (?:the |this )?"
            r"(?:(?:chart|image|plot|graph|visualization) )?title(?: |$)",
        ),
        segments=_summary_title_binding_segments(summary),
    )
    title_claims = _summary_title_claims(summary)
    parsed_title_counts = {
        segment: sum(
            claim["segment"] == segment for claim in title_claims
        )
        for segment in {claim["segment"] for claim in title_claims}
    }
    unclassified_title_claims = [
        segment for segment in _summary_title_segments(summary)
        if (
            (
                (title_mentions := len(re.findall(
                    r"\btitle(?:d)?\b", _claim_text(segment),
                )))
                and parsed_title_counts.get(segment, 0) != title_mentions
            )
            or (
                title_claims
                and re.match(
                    r"^(?:it|that|the former|the latter|this) "
                    r"(?:(?:also )?(?:is|reads|says|states)(?: also)?\b)",
                    _claim_text(segment),
                )
                and parsed_title_counts.get(segment, 0) == 0
            )
        )
    ]
    distinct_claims = sorted({claim["claim"] for claim in title_claims})
    one_truth_claim = bool(
        truth_present
        and distinct_claims == [_claim_text(truth)]
    )
    checks = {
        "before_capture_ok": state.valid,
        "canonical_title_path": valid_path,
        "gt_title_present": truth_present,
        "target_truth_matches_gt": _has_renderable_text(target_truth)
        and _equal(truth, target_truth),
        "injected_title_present": _has_renderable_text(injected),
        "truth_title_verbatim_in_summary": truth_present
        and _contains_exact_visible_text(summary, truth),
        "truth_title_affirmatively_bound": title_binding[
            "direct_affirmative_relation"
        ],
        "title_claim_segments_classified": not unclassified_title_claims,
        "one_distinct_truth_title_claim": one_truth_claim,
        "injected_title_absent_from_summary": title_binding["injected_absent"],
    }
    return {
        "schema_version": "chartrepairbench.v2.title-recoverability/2",
        "ok": all(checks.values()),
        "reason": next((name for name, ok in checks.items() if not ok),
                       "truth_title_agent_visible"),
        "checks": checks,
        "canonical_path": path if valid_path else None,
        "truth_title": truth if truth_present else None,
        "summary_claim_binding": title_binding,
        "summary_title_claims": title_claims,
        "unclassified_summary_title_claims": unclassified_title_claims,
    }


def _legacy_title_recoverability(
    before: Mapping[str, Any] | None,
    target: Mapping[str, Any],
    summary: str,
) -> dict[str, Any]:
    """Recompute the frozen title-recoverability/1 contract."""
    state = _state(before)
    path = target.get("path")
    valid_path = isinstance(path, str) and bool(
        path == "suptitle" or re.fullmatch(r"axes\[\d+\]\.title", path)
    )
    truth = state.flat.get(path, _MISSING) if valid_path else _MISSING
    target_truth = target.get("truth", target.get("truth_title", _MISSING))
    injected = target.get("injected", target.get("injected_title", _MISSING))
    truth_present = _has_renderable_text(truth)
    title_binding = _summary_claim_binding(
        summary,
        [truth] if truth_present else [],
        injected_phrases=[injected],
        relation_patterns=(
            r"(?:the |this )?(?:(?:chart|plot|graph|visualization) )?title "
            r"(?:is|reads|says|states) truthclaimtoken(?: |$)",
            r"(?:the |this )?(?:chart|plot|graph|visualization) "
            r"(?:is )?titled truthclaimtoken(?: |$)",
            r"truthclaimtoken (?:is|appears as) (?:the |this )?"
            r"(?:(?:chart|plot|graph|visualization) )?title(?: |$)",
        ),
    )
    title_claims = _legacy_summary_title_claims(summary)
    parsed_title_segments = {claim["segment"] for claim in title_claims}
    unclassified_title_claims = [
        segment for segment in _summary_segments(summary)
        if (
            re.search(r"\btitle(?:d)?\b", _claim_text(segment))
            or (
                title_claims
                and re.match(
                    r"^(?:it|that|the former|the latter|this) "
                    r"(?:(?:also )?(?:is|reads|says|states)(?: also)?\b)",
                    _claim_text(segment),
                )
            )
        )
        and segment not in parsed_title_segments
    ]
    distinct_claims = sorted({claim["claim"] for claim in title_claims})
    one_truth_claim = bool(
        truth_present and distinct_claims == [_claim_text(truth)]
    )
    checks = {
        "before_capture_ok": state.valid,
        "canonical_title_path": valid_path,
        "gt_title_present": truth_present,
        "target_truth_matches_gt": _has_renderable_text(target_truth)
        and _equal(truth, target_truth),
        "injected_title_present": _has_renderable_text(injected),
        "truth_title_verbatim_in_summary": truth_present
        and _contains_exact_visible_text(summary, truth),
        "truth_title_affirmatively_bound": title_binding[
            "direct_affirmative_relation"
        ],
        "title_claim_segments_classified": not unclassified_title_claims,
        "one_distinct_truth_title_claim": one_truth_claim,
        "injected_title_absent_from_summary": title_binding["injected_absent"],
    }
    return {
        "schema_version": "chartrepairbench.v2.title-recoverability/1",
        "ok": all(checks.values()),
        "reason": next(
            (name for name, ok in checks.items() if not ok),
            "truth_title_agent_visible",
        ),
        "checks": checks,
        "canonical_path": path if valid_path else None,
        "truth_title": truth if truth_present else None,
        "summary_claim_binding": title_binding,
        "summary_title_claims": title_claims,
        "unclassified_summary_title_claims": unclassified_title_claims,
    }


def _chart_type_recoverability(
    before: Mapping[str, Any] | None,
    target: Mapping[str, Any],
    summary: str,
) -> dict[str, Any]:
    state = _state(before)
    series = _series(state)
    labels = _labels(target)
    roots = [_series_key(series, label, target.get("axis_index")) for label in labels]
    truth_kind = target.get("truth_kind", target.get("before_kind"))
    injected_kind = target.get("injected_kind", target.get("after_kind"))
    truth_kind_key = str(truth_kind).casefold()
    injected_kind_key = str(injected_kind).casefold()
    visible_phrases = _CHART_KIND_PHRASES.get(truth_kind_key, (truth_kind,))
    injected_phrases = _CHART_KIND_PHRASES.get(
        injected_kind_key, (injected_kind,),
    )
    relation_patterns = (
        r"(?:the |this )?(?:chart|figure|image|plot|graph|visualization) "
        r"(?:appears as|is|presents as|uses) (?:a |an )?"
        r"truthclaimtoken(?: titled .+)?",
        r"(?:this|it) (?:appears as|is) (?:a |an )?"
        r"truthclaimtoken",
        r"subjectclaimtoken(?: series)? "
        r"(?:appears|appear|is|are)(?: displayed| plotted| presented| shown)? "
        r"(?:in|as|using)? ?(?:a |an )?truthclaimtoken"
        rf"{_CLAIM_DIMENSION_TAIL}",
        r"(?:a |an |the )?truthclaimtoken(?: chart| plot| graph)? "
        r"(?:depicts|displays|plots|presents|reports|shows|tracks) "
        r"subjectclaimtoken",
    )
    truth_binding = _summary_claim_binding(
        summary,
        visible_phrases,
        subject_phrases=labels,
        relation_patterns=relation_patterns,
    )
    injected_binding = _summary_claim_binding(
        summary,
        injected_phrases,
        subject_phrases=labels,
        relation_patterns=relation_patterns,
    )
    claim_phrases = dict(_CHART_KIND_PHRASES)
    claim_phrases.setdefault(truth_kind_key, visible_phrases)
    claim_bindings = {
        kind: _summary_claim_binding(
            summary,
            phrases,
            subject_phrases=labels,
            relation_patterns=relation_patterns,
        )
        for kind, phrases in claim_phrases.items()
    }
    scoped_kind_claims = sorted(
        kind for kind, binding in claim_bindings.items()
        if binding["direct_affirmative_relation"]
    )
    blocked_target_kind_claims = []
    unclassified_target_kind_claims = []
    for segment in _summary_segments(summary):
        text = _claim_text(segment)
        target_named = any(
            _contains_exact_visible_text(segment, label) for label in labels
        )
        whole_chart_scope = bool(re.search(
            r"\b(?:(?:the|this) )?(?:chart|figure|image|graph|plot|visualization) "
            r"(?:appears|is|presents|uses)\b|"
            r"\b(?:it|this) (?:appears as|is)\b|"
            r"^(?:that series|the former|the latter|this series) "
            r"(?:appears|is|uses)\b",
            text,
        ))
        if not (target_named or whole_chart_scope):
            continue
        for kind, phrases in claim_phrases.items():
            if not any(
                _contains_exact_visible_text(segment, phrase)
                for phrase in phrases
            ):
                continue
            segment_binding = _summary_claim_binding(
                segment,
                phrases,
                subject_phrases=labels,
                relation_patterns=relation_patterns,
            )
            record = {"segment": segment, "kind": kind}
            if segment_binding["blocked_truth_segments"]:
                blocked_target_kind_claims.append(record)
            elif not segment_binding["direct_affirmative_relation"]:
                unclassified_target_kind_claims.append(record)
    summary_kind_conflict_free_for_code = bool(
        not blocked_target_kind_claims
        and not unclassified_target_kind_claims
        and (
            not scoped_kind_claims
            or scoped_kind_claims == [truth_kind_key]
        )
    )
    checks = {
        "before_capture_ok": state.valid,
        "target_series_unique": bool(labels) and all(root is not None for root in roots),
        "truth_kind_present": _has_renderable_text(truth_kind),
        "injected_kind_present": _has_renderable_text(injected_kind),
        "truth_kind_matches_gt": bool(roots) and all(
            root is not None and _equal(series[root].get("kind"), truth_kind)
            for root in roots
        ),
        "truth_kind_verbatim_in_summary": _has_renderable_text(truth_kind)
        and any(_contains_exact_visible_text(summary, phrase)
                for phrase in visible_phrases),
        "truth_kind_affirmatively_bound": truth_binding[
            "direct_affirmative_relation"
        ],
        "truth_kind_claims_unblocked": not truth_binding[
            "blocked_truth_segments"
        ],
        "target_kind_segments_classified": not (
            unclassified_target_kind_claims
        ),
        "one_distinct_truth_kind_for_target": scoped_kind_claims == [
            truth_kind_key
        ],
        "injected_kind_not_affirmed_for_target": not injected_binding[
            "direct_affirmative_relation"
        ],
    }
    return {
        "schema_version": "chartrepairbench.v2.chart-type-recoverability/1",
        "ok": all(checks.values()),
        "reason": next((name for name, ok in checks.items() if not ok),
                       "truth_chart_type_agent_visible"),
        "checks": checks,
        "truth_kind": truth_kind,
        "accepted_summary_phrases": list(visible_phrases),
        "truth_summary_claim_binding": truth_binding,
        "injected_summary_claim_binding": injected_binding,
        "scoped_kind_claims": scoped_kind_claims,
        "scoped_kind_claim_bindings": claim_bindings,
        "blocked_target_kind_claims": blocked_target_kind_claims,
        "unclassified_target_kind_claims": unclassified_target_kind_claims,
        "summary_kind_conflict_free_for_code": (
            summary_kind_conflict_free_for_code
        ),
        "target_series": [root for root in roots if root is not None],
    }


def _dropped_series_kind_recoverability(
    before: Mapping[str, Any] | None,
    target: Mapping[str, Any],
    *,
    summary: str,
    flawed_code: str,
) -> dict[str, Any]:
    state = _state(before)
    series = _series(state)
    labels = _labels(target)
    root = _series_key(series, labels[0], target.get("axis_index")) \
        if len(labels) == 1 else None
    truth_kind = series[root].get("kind") if root else None
    summary_binding = _chart_type_recoverability(
        before,
        {
            "series_label": labels[0] if len(labels) == 1 else None,
            "axis_index": target.get("axis_index"),
            "truth_kind": truth_kind,
            "injected_kind": "removed-series",
        },
        summary,
    )
    summary_target_kind = bool(
        len(labels) == 1
        and summary_binding.get("ok") is True
        and any(
            _contains_exact_visible_text(segment, labels[0])
            for segment in summary_binding.get(
                "truth_summary_claim_binding", {}
            ).get("affirmative_segments", [])
        )
    )
    calls = _active_labeled_kind_calls(flawed_code)
    survivor_labels = {
        rec["label"] for candidate_root, rec in series.items()
        if candidate_root != root
        and (root is None or rec["axis"] == series[root]["axis"])
    }
    survivor_calls = [
        call for call in calls
        if any(_equal(call["label"], label) for label in survivor_labels)
    ]
    survivor_kinds = sorted({call["kind"] for call in survivor_calls})
    gt_axis_kinds = sorted({
        str(rec.get("kind")).casefold()
        for rec in series.values()
        if root is not None and rec["axis"] == series[root]["axis"]
        and _has_renderable_text(rec.get("kind"))
    })
    gt_axis_uniform = bool(
        _has_renderable_text(truth_kind)
        and gt_axis_kinds == [str(truth_kind).casefold()]
    )
    code_uniform_kind = bool(
        gt_axis_uniform
        and survivor_calls
        and survivor_kinds == [str(truth_kind).casefold()]
    )
    summary_kind_conflict_free = bool(
        summary_binding.get("summary_kind_conflict_free_for_code") is True
    )
    checks = {
        "before_capture_ok": state.valid,
        "target_series_unique": root is not None,
        "truth_kind_present": _has_renderable_text(truth_kind),
        "summary_target_kind_claims_conflict_free": summary_kind_conflict_free,
        "kind_visible_in_direct_summary_or_uniform_survivor_code": bool(
            summary_target_kind or code_uniform_kind
        ),
    }
    return {
        "schema_version": "chartrepairbench.v2.dropped-series-kind-binding/1",
        "ok": all(checks.values()),
        "reason": next(
            (name for name, passed in checks.items() if not passed),
            "dropped_series_kind_agent_visible",
        ),
        "checks": checks,
        "target_series": root,
        "truth_kind": truth_kind,
        "summary_kind_binding": summary_binding,
        "summary_target_kind_bound": summary_target_kind,
        "active_labeled_kind_calls": calls,
        "survivor_kind_calls": survivor_calls,
        "survivor_kinds": survivor_kinds,
        "gt_axis_kinds": gt_axis_kinds,
    }


def _candidate_value_binding(
    flawed_code: str,
    source_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    binding = _mapping(source_binding)
    path = binding.get("source_value_ast_path")
    injected = binding.get("injected_value")
    checks = {
        "source_binding_ok": binding.get("ok") is True,
        "candidate_code_present": isinstance(flawed_code, str)
        and bool(flawed_code.strip()),
        "source_ast_path_present": isinstance(path, list) and bool(path),
        "candidate_ast_path_matches_injected": False,
    }
    if all(list(checks.values())[:3]):
        try:
            node: Any = ast.parse(flawed_code)
            for segment in path:
                node = node[segment] if isinstance(segment, int) else getattr(node, segment)
            value = ast.literal_eval(node)
        except (AttributeError, IndexError, KeyError, SyntaxError, TypeError, ValueError):
            value = _MISSING
        checks["candidate_ast_path_matches_injected"] = _equal(value, injected)
    return {
        "schema_version": "chartrepairbench.v2.candidate-value-binding/1",
        "ok": all(checks.values()),
        "reason": next((name for name, ok in checks.items() if not ok),
                       "candidate_wrong_value_agent_visible"),
        "checks": checks,
        "source_value_ast_path": path,
        "truth_value": binding.get("truth_value"),
        "injected_value": injected,
    }


def _axis_truncation_recoverability(
    before: Mapping[str, Any] | None,
    target: Mapping[str, Any],
    *,
    summary: str,
    table: str,
    flawed_code: str,
) -> dict[str, Any]:
    clipped = target.get("clipped")
    points = [dict(item) for item in clipped if isinstance(item, Mapping)] \
        if isinstance(clipped, list) else []
    labels = list(dict.fromkeys(
        str(point.get("series_label")).strip()
        for point in points if _has_renderable_text(point.get("series_label"))
    ))
    table_binding = table_capture_series_binding(
        before, table, labels, axis_index=target.get("axis_index"),
    )
    state_series = _series(_state(before))
    values_bound = bool(points) and table_binding.get("ok") is True
    if values_bound:
        for point in points:
            root = _series_key(
                state_series, point.get("series_label"), target.get("axis_index"),
            )
            sequence = _series_value_sequence(state_series[root]) if root else None
            point_index = _target_index(point.get("point_index"))
            if sequence is None or point_index is None or point_index >= len(sequence) \
                    or not _equal(sequence[point_index], point.get("value")):
                values_bound = False
                break
    labels_bound = bool(points) and all(
        _has_renderable_text(point.get("series_label"))
        and any(_contains_exact_visible_text(source, point["series_label"])
                for source in (summary, table, flawed_code))
        for point in points
    )
    axis = target.get("axis")
    axis_index = _target_index(target.get("axis_index"))
    limits = target.get("flawed_limits")
    limit_records = []
    try:
        tree = ast.parse(flawed_code) if isinstance(flawed_code, str) else None
    except SyntaxError:
        tree = None
    if tree is not None and axis in {"x", "y"}:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            name = node.func.attr
            value = _MISSING
            if name in {f"{axis}lim", f"set_{axis}lim"} and len(node.args) >= 2:
                try:
                    value = [ast.literal_eval(node.args[0]), ast.literal_eval(node.args[1])]
                except (ValueError, TypeError):
                    pass
            elif name == f"update_{axis}axes":
                keyword = next(
                    (item.value for item in node.keywords if item.arg == "range"), None,
                )
                if keyword is not None:
                    try:
                        value = ast.literal_eval(keyword)
                    except (ValueError, TypeError):
                        pass
            if isinstance(value, (list, tuple)) and len(value) == 2:
                limit_records.append((node.lineno, node.col_offset, list(value)))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value_node = node.value
            try:
                value = ast.literal_eval(value_node)
            except (ValueError, TypeError):
                continue
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                continue
            for assignment_target in targets:
                parts = []
                current = assignment_target
                while isinstance(current, ast.Attribute):
                    parts.append(current.attr)
                    current = current.value
                parts.reverse()
                if parts[-3:] == ["layout", f"{axis}axis", "range"]:
                    limit_records.append(
                        (node.lineno, node.col_offset, list(value))
                    )
    limit_candidates = [
        value for _, _, value in sorted(limit_records, key=lambda item: item[:2])
    ]
    limit_literal_visible = bool(
        axis in {"x", "y"} and axis_index is not None
        and isinstance(limits, list) and len(limits) == 2
        and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in limits)
        and bool(limit_candidates)
        and _equal(limit_candidates[-1], limits)
    )
    checks = {
        "clipped_points_present": bool(points),
        "clipped_values_bound_to_table": values_bound,
        "clipped_series_labels_agent_visible": labels_bound,
        "flawed_limit_visible_in_code": limit_literal_visible,
    }
    return {
        "schema_version": "chartrepairbench.v2.axis-recoverability/1",
        "ok": all(checks.values()),
        "reason": next((name for name, ok in checks.items() if not ok),
                       "clipped_truth_agent_visible"),
        "checks": checks,
        "clipped_points": points,
        "series_table_binding": table_binding,
        "visible_limit_candidates": limit_candidates,
    }


def repair_input_recoverability(
    category: str,
    before: Mapping[str, Any] | None,
    target: Mapping[str, Any] | None,
    *,
    summary: str,
    table: str,
    flawed_code: str,
    evidence: Mapping[str, Any] | None = None,
    value_source_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prove the exact repair truth exists in the agent's physical inputs."""
    target = _mapping(target)
    state = _state(before)
    if category == "wrong-title":
        decision = _title_recoverability(before, target, summary)
    elif category == "wrong-axis-label":
        decision = wrong_axis_label_text_recoverability(
            before, target, summary=summary, table=table, flawed_code=flawed_code,
        )
    elif category == "missing-legend":
        decision = missing_legend_text_recoverability(
            before, target, summary=summary, table=table, flawed_code=flawed_code,
        )
    elif category == "color-contradicts-summary":
        decision = color_summary_evidence_binding(
            summary, target, evidence, before=before,
        )
    elif category == "wrong-chart-type":
        decision = _chart_type_recoverability(before, target, summary)
    elif category == "axis-truncation":
        decision = _axis_truncation_recoverability(
            before, target, summary=summary, table=table, flawed_code=flawed_code,
        )
    elif category in {"dropped-series", "swapped-series",
                      "swapped-legend-labels"}:
        labels = _labels(target)
        decision = table_capture_series_binding(
            before, table, labels, axis_index=target.get("axis_index"),
        )
        if category == "dropped-series":
            kind_binding = _dropped_series_kind_recoverability(
                before,
                target,
                summary=summary,
                flawed_code=flawed_code,
            )
            decision = {
                **decision,
                "dropped_series_kind_binding": kind_binding,
            }
            if decision.get("ok") is True and kind_binding.get("ok") is not True:
                decision = {
                    **decision,
                    "ok": False,
                    "reason": kind_binding.get(
                        "reason", "dropped_series_kind_not_agent_visible",
                    ),
                }
        elif category == "swapped-legend-labels" and decision.get("ok") is True:
            bindings = decision.get("bindings", [])
            legend_path = target.get("path")
            truth_labels = _legend_groups(state).get(legend_path, ())
            positions = [binding.get("value_column") for binding in bindings]
            ordered = bool(
                len(labels) == 2
                and len(bindings) == 2
                and all(binding.get("mode") == "wide" for binding in bindings)
                and labels[0] in truth_labels and labels[1] in truth_labels
                and truth_labels.index(labels[0]) < truth_labels.index(labels[1])
                and positions[0] < positions[1]
            )
            decision = {
                **decision,
                "ok": ordered,
                "reason": "legend_truth_order_bound" if ordered
                else "legend_truth_order_not_agent_visible",
                "legend_path": legend_path,
                "truth_legend_labels": list(truth_labels),
            }
    elif category == "value-perturbation":
        decision = _candidate_value_binding(flawed_code, value_source_binding)
    else:
        decision = {
            "schema_version": "chartrepairbench.v2.unsupported-recoverability/1",
            "ok": False,
            "reason": "unknown_category",
        }
    return {
        "schema_version": REPAIR_INPUT_RECOVERABILITY_SCHEMA,
        "category": category,
        "ok": decision.get("ok") is True,
        "reason": decision.get("reason", "repair_truth_not_agent_visible"),
        "summary_sha256": hashlib.sha256(summary.encode()).hexdigest()
        if isinstance(summary, str) else None,
        "table_sha256": hashlib.sha256(table.encode()).hexdigest()
        if isinstance(table, str) else None,
        "flawed_code_sha256": hashlib.sha256(flawed_code.encode()).hexdigest()
        if isinstance(flawed_code, str) else None,
        "gt_capture_sha256": _capture_sha256(state),
        "decision": decision,
    }


def wrong_title_postcondition(before: dict, after: dict, target=None, evidence=None) -> dict:
    before, after = _state(before), _state(after)
    target, _ = _metadata(target, evidence)
    changed = _changed(before, after)
    paths = target.get("paths") or ([target["path"]] if target.get("path") else [])
    if not paths:
        paths = [path for path in changed
                 if path == "suptitle" or (path.endswith(".title") and ".legend." not in path)]
    paths = [str(path) for path in paths]
    failures = []
    if len(paths) != 1:
        failures.append("title_target_not_unique")
    path = paths[0] if len(paths) == 1 else None
    truth = target.get("truth", target.get("truth_title"))
    injected = target.get("injected", target.get("injected_title"))
    before_value = before.flat.get(path) if path else None
    after_value = after.flat.get(path) if path else None
    exists = bool(
        path and _has_renderable_text(before_value)
        and _has_renderable_text(after_value)
    )
    changed_title = exists and not _equal(before_value, after_value)
    target_match = exists and _expected(before_value, truth) and _expected(after_value, injected)
    if not exists:
        failures.append("title_missing_before_or_after")
    elif not changed_title:
        failures.append("title_not_changed")
    if exists and not target_match:
        failures.append("title_does_not_match_target")
    collateral = _collateral(changed, exact=tuple(paths), target=target)
    return _base(
        "wrong-title", before, after,
        checks={"target_unique": len(paths) == 1, "nonempty_before_after": exists,
                "title_changed": changed_title, "target_match": target_match},
        success_reason="target_title_replaced", failures=failures,
        collateral_paths=collateral, target_paths=paths,
        truth_title=before_value, flawed_title=after_value,
    )


def wrong_axis_label_postcondition(before: dict, after: dict, target=None, evidence=None) -> dict:
    before, after = _state(before), _state(after)
    target, _ = _metadata(target, evidence)
    changed = _changed(before, after)
    has_axis_identity = "axis" in target or "axis_index" in target
    canonical_path = _axis_label_path(target)
    if has_axis_identity:
        # Backend-native paths such as ``yaxis.title.text`` do not identify a
        # Figure-Object property. Structured axis identity is authoritative.
        paths = [canonical_path] if canonical_path else []
    else:
        paths = target.get("paths") or ([target["path"]] if target.get("path") else [])
    if not paths and not has_axis_identity:
        paths = [path for path in changed if path.endswith((".xlabel", ".ylabel"))]
    paths = [str(path) for path in paths]
    failures = []
    if not paths or len(paths) > 2:
        failures.append("axis_label_target_not_identified")
    values = [(before.flat.get(path), after.flat.get(path)) for path in paths]
    nonempty = bool(values) and all(
        _has_renderable_text(old) and _has_renderable_text(new)
        for old, new in values
    )
    changed_labels = nonempty and all(not _equal(old, new) for old, new in values)
    truth = target.get("truth", target.get("truth_label"))
    injected = target.get("injected", target.get("injected_label"))
    target_match = nonempty and (
        len(values) != 1 or (_expected(values[0][0], truth) and _expected(values[0][1], injected))
    )
    if not nonempty:
        failures.append("axis_label_missing_before_or_after")
    elif not changed_labels:
        failures.append("axis_label_not_changed")
    if nonempty and not target_match:
        failures.append("axis_label_does_not_match_target")
    collateral = _collateral(changed, exact=tuple(paths), target=target)
    return _base(
        "wrong-axis-label", before, after,
        checks={"target_identified": bool(paths) and len(paths) <= 2,
                "nonempty_before_after": nonempty, "labels_changed": changed_labels,
                "target_match": target_match},
        success_reason="target_axis_label_replaced", failures=failures,
        collateral_paths=collateral, target_paths=paths,
        truth_labels=[old for old, _ in values], flawed_labels=[new for _, new in values],
    )


def missing_legend_postcondition(before: dict, after: dict, target=None, evidence=None) -> dict:
    """Prove that a concrete GT legend group disappears everywhere.

    V2 repair scoring requires the missing label group, which reconstructs the
    legend. A captured legend title is incidental: it may be library-generated
    or absent from every repair-agent input, and is not required to restore the
    category's semantic defect.
    """
    before, after = _state(before), _state(after)
    target, _ = _metadata(target, evidence)
    reference = _legend_groups(before)
    remaining = _legend_groups(after)
    target_paths = target.get("paths") or ([target["path"]] if target.get("path") else [])
    if target_paths:
        reference = {path: labels for path, labels in reference.items() if path in target_paths}
    requested_labels = target.get("labels")
    if requested_labels:
        wanted = {str(label).strip() for label in requested_labels}
        reference = {path: labels for path, labels in reference.items() if wanted <= set(labels)}
    visible_labels = {label for labels in remaining.values() for label in labels}
    removed = {
        path: list(labels) for path, labels in reference.items()
        if path not in remaining and set(labels).isdisjoint(visible_labels)
    }
    blocked = {path: list(labels) for path, labels in reference.items() if path not in removed}
    failures = []
    if not reference:
        failures.append("reference_has_no_nonempty_legend")
    elif not removed:
        failures.append("legend_labels_still_visible")
    changed = _changed(before, after)
    required_recovery_paths = sorted(removed)
    allowed_prefixes = tuple(path.rsplit(".labels", 1)[0] for path in removed)
    incidental_recovery_paths = sorted(
        path for path in changed
        if path.endswith(".legend.title")
        and any(path.startswith(prefix + ".") for prefix in allowed_prefixes)
    )
    recovery_contract = {
        "schema_version": "chartrepairbench.v2.category-recovery/1",
        "required_paths": required_recovery_paths,
        "incidental_paths": incidental_recovery_paths,
    }
    collateral = _collateral(changed, prefixes=allowed_prefixes, target=target)
    result = _base(
        "missing-legend", before, after,
        checks={"reference_nonempty": bool(reference), "group_removed": bool(removed),
                "labels_absent_everywhere": bool(removed)},
        success_reason="legend_labels_removed", failures=failures,
        collateral_paths=collateral, reference_paths=sorted(reference),
        remaining_paths=sorted(remaining), removed_paths=sorted(removed),
        blocked_paths=sorted(blocked),
        required_recovery_paths=required_recovery_paths,
        incidental_recovery_paths=incidental_recovery_paths,
        recovery_contract=recovery_contract,
    )
    result["name"] = "missing_legend_labels_removed"
    return result


def _legend_order_swap(
    before: _State, after: _State, labels: list[str], target_path: str | None,
) -> tuple[bool, list[str]]:
    if len(labels) != 2:
        return False, []
    a, b = labels
    matches = []
    for path, old in _legend_groups(before).items():
        if target_path and path != target_path:
            continue
        new = _legend_groups(after).get(path)
        if new is None or len(old) != len(new):
            continue
        indices = [i for i, value in enumerate(old) if value in (a, b)]
        if len(indices) != 2:
            continue
        expected = list(old)
        expected[indices[0]], expected[indices[1]] = expected[indices[1]], expected[indices[0]]
        if _equal(expected, list(new)):
            matches.append(path)
    return len(matches) == 1, matches


def swapped_legend_labels_postcondition(before: dict, after: dict, target=None, evidence=None) -> dict:
    before, after = _state(before), _state(after)
    target, _ = _metadata(target, evidence)
    bs, fs = _series(before), _series(after)
    labels = _labels(target)
    target_path = target.get("path") or target.get("legend_path")

    def candidate_pairs():
        for left, right in itertools.combinations(sorted(set(bs) & set(fs)), 2):
            if _equal(_identity(fs[left]), _identity(bs[right])) and \
                    _equal(_identity(fs[right]), _identity(bs[left])) and \
                    not _equal(_identity(bs[left]), _identity(bs[right])):
                yield left, right

    pairs = []
    if len(labels) == 2:
        left = _series_key(bs, labels[0], target.get("axis_index"))
        right = _series_key(bs, labels[1], target.get("axis_index"))
        if left and right:
            pairs = [(left, right)]
    else:
        inferred = list(candidate_pairs())
        if len(inferred) == 1:
            inferred_labels = [bs[inferred[0][0]]["label"], bs[inferred[0][1]]["label"]]
            if not labels or set(inferred_labels) <= set(labels):
                pairs = inferred
                labels = inferred_labels

    series_swap = False
    if len(pairs) == 1:
        left, right = pairs[0]
        series_swap = set(bs) == set(fs) and \
            _equal(_identity(fs[left]), _identity(bs[right])) and \
            _equal(_identity(fs[right]), _identity(bs[left])) and \
            all(_equal(_identity(bs[root]), _identity(fs[root]))
                for root in bs if root not in (left, right)) and \
            not _equal(_identity(bs[left]), _identity(bs[right]))
    order_swap, legend_paths = _legend_order_swap(before, after, labels, target_path)
    series_unchanged_for_order = set(bs) == set(fs) and all(
        _equal(_identity(bs[root]), _identity(fs[root])) for root in bs
    )
    order_mode = order_swap and series_unchanged_for_order
    legend_labels = {label for group in _legend_groups(before).values() for label in group}
    labels_truth_visible = (
        len(labels) == 2
        and all(_has_renderable_text(label) for label in labels)
        and len({_normalized_visible_text(label) for label in labels}) == 2
        and set(labels) <= legend_labels
    )
    failures = []
    if len(labels) != 2:
        failures.append("legend_swap_target_not_unique")
    elif not labels_truth_visible:
        failures.append("target_labels_not_in_gt_legend")
    elif not (series_swap or order_mode):
        failures.append("legend_mapping_not_exactly_swapped")

    changed = _changed(before, after)
    if series_swap:
        prefixes = tuple(pairs[0])
        collateral = _collateral(
            changed, exact=tuple(legend_paths) if order_swap else (),
            prefixes=prefixes,
            target=target,
        )
        mode = "series_label_assignment"
    else:
        collateral = _collateral(changed, exact=tuple(legend_paths), target=target)
        mode = "legend_order" if order_mode else None
    return _base(
        "swapped-legend-labels", before, after,
        checks={"two_target_labels": len(labels) == 2,
                "labels_visible_in_gt_legend": labels_truth_visible,
                "series_identity_swap": series_swap,
                "legend_order_swap": order_mode},
        success_reason="legend_label_mapping_swapped", failures=failures,
        collateral_paths=collateral, labels=labels, mapping_mode=mode,
        target_series=list(pairs[0]) if len(pairs) == 1 else [],
        legend_paths=legend_paths,
    )


def _normal_color(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        from matplotlib.colors import to_hex

        return to_hex(value, keep_alpha=False).casefold()
    except (TypeError, ValueError):
        return value.strip().casefold()


_NAMED_COLOR_HUES = {
    "red": 0.0,
    "orange": 30 / 360,
    "yellow": 60 / 360,
    "green": 120 / 360,
    "teal": 180 / 360,
    "cyan": 180 / 360,
    "blue": 240 / 360,
    "purple": 275 / 360,
    "magenta": 300 / 360,
    "pink": 345 / 360,
    "brown": 25 / 360,
}


def _color_matches_claim(value: Any, claim: Any) -> bool:
    actual = _normal_color(value)
    expected = _normal_color(claim)
    if actual is None or expected is None:
        return False
    if actual == expected:
        return True
    name = str(claim).strip().casefold()
    if name not in {*_NAMED_COLOR_HUES, "black", "white", "gray", "grey"}:
        return False
    try:
        from matplotlib.colors import to_rgb

        red, green, blue = to_rgb(value)
    except (TypeError, ValueError):
        return False
    hue, saturation, brightness = colorsys.rgb_to_hsv(red, green, blue)
    if name == "black":
        return brightness <= 0.20 and saturation <= 0.12
    if name == "white":
        return brightness >= 0.85 and saturation <= 0.12
    if name in {"gray", "grey"}:
        return 0.20 < brightness < 0.85 and saturation <= 0.12
    expected_hue = _NAMED_COLOR_HUES[name]
    hue_distance = min(abs(hue - expected_hue), 1 - abs(hue - expected_hue))
    return saturation >= 0.08 and hue_distance <= 30 / 360


_COLOR_GROUP_STOPWORDS = {
    "bar", "bars", "color", "colors", "colour", "colours", "data", "edge", "edges",
    "histogram", "line", "lines", "marker", "markers", "of", "series", "the", "trace",
    "traces",
}


def _polarity_signature(tokens: set[str]) -> tuple[bool, bool, bool] | None:
    positive = "positive" in tokens
    negative = "negative" in tokens
    if not positive and not negative:
        return None
    return positive, negative, bool(
        tokens & {"non", "not", "no", "without", "excluding", "except"}
    )


def _color_group_matches_entity(label: Any, paths: list[str]) -> bool:
    target_text = str(label).casefold().replace("_", " ")
    target_tokens = set(re.findall(r"[\w]+", target_text)) - _COLOR_GROUP_STOPWORDS
    if not target_tokens:
        return False
    target_polarity = _polarity_signature(target_tokens)
    common_tokens = set(target_tokens)
    for path in paths:
        match = _SERIES_RE.match(path)
        if match is None:
            return False
        series_tokens = set(re.findall(
            r"[\w]+", match.group(2).casefold().replace("_", " ")
        ))
        if target_polarity is not None and \
                _polarity_signature(series_tokens) != target_polarity:
            return False
        common_tokens &= series_tokens
    return bool(common_tokens)


def _is_series_color_path(path: Any) -> bool:
    if not isinstance(path, str):
        return False
    match = _SERIES_RE.match(path)
    return bool(match and match.group(3) in {"color", "colors", "edgecolor"})


def _color_mapping(target: dict, evidence: dict) -> dict[str, Any]:
    raw = (evidence.get("color_mapping") or evidence.get("summary_color_mapping")
           or target.get("summary_color_mapping"))
    if isinstance(raw, Mapping):
        return {str(key): value for key, value in raw.items()}
    mapping = {}
    if isinstance(raw, list):
        for item in raw:
            item = _mapping(item)
            label = item.get("entity") or item.get("series_label") or item.get("label")
            color = item.get("color") or item.get("truth_color")
            if label is not None and color is not None:
                mapping[str(label)] = color
    return mapping


def color_contradicts_summary_postcondition(before: dict, after: dict, target=None, evidence=None) -> dict:
    before, after = _state(before), _state(after)
    target, evidence = _metadata(target, evidence)
    bs = _series(before)
    mapping = _color_mapping(target, evidence)
    label = target.get("series_label") or target.get("entity") or target.get("label")
    explicit_path = target.get("path")
    path = explicit_path if _is_series_color_path(explicit_path) else None
    invalid_explicit_path = explicit_path is not None and path is None
    if not label and len(mapping) == 1:
        label = next(iter(mapping))
    root = _series_key(bs, label, target.get("axis_index")) if label else None
    truth_color = mapping.get(str(label)) if label is not None else None
    truth_color = target.get("summary_color", target.get("truth_color", truth_color))
    if explicit_path is None and root:
        candidates = [f"{root}.{field}" for field in ("color", "colors", "edgecolor")
                      if f"{root}.{field}" in before.flat]
        matches = [candidate for candidate in candidates
                   if _color_matches_claim(before.flat[candidate], truth_color)]
        path = matches[0] if len(matches) == 1 else None
    missing = []
    if not (label or path):
        missing.append("target.series_label_or_path")
    if truth_color is None:
        missing.append("evidence.summary_color_mapping")
    failures = []
    if invalid_explicit_path:
        failures.append("color_target_path_not_color")
    if explicit_path is not None and label and root is None:
        failures.append("color_target_entity_not_found")
    if path and root and path.rsplit(".", 1)[0] != root:
        failures.append("color_target_path_entity_mismatch")
    injected = target.get("injected_color") or target.get("injected")
    if injected is None:
        missing.append("target.injected_color")
    changed = _changed(before, after)
    paths = [path] if path else []
    if not paths and not missing:
        changed_color_paths = []
        for candidate in changed:
            match = _SERIES_RE.match(candidate)
            if match and match.group(3) in {"color", "colors", "edgecolor"}:
                changed_color_paths.append(candidate)
        matching_paths = [
            candidate for candidate in changed_color_paths
            if _color_matches_claim(before.flat.get(candidate), truth_color)
            and not _color_matches_claim(after.flat.get(candidate), truth_color)
            and (injected is None
                 or _color_matches_claim(after.flat.get(candidate), injected))
        ]
        semantic_changes = [candidate for candidate in changed if _semantic_path(candidate)]
        group_bound = len(matching_paths) == 1 or _color_group_matches_entity(
            label, matching_paths,
        )
        if matching_paths and group_bound and set(matching_paths) == set(semantic_changes):
            paths = matching_paths
    before_colors = [before.flat.get(candidate) for candidate in paths]
    after_colors = [after.flat.get(candidate) for candidate in paths]
    truth_matches = bool(paths) and all(
        _color_matches_claim(color, truth_color) for color in before_colors
    )
    contradicted = bool(paths) and all(
        color is not None and not _color_matches_claim(color, truth_color)
        for color in after_colors
    )
    injected_matches = injected is None or bool(paths) and all(
        _color_matches_claim(color, injected) for color in after_colors
    )
    if not missing and not paths:
        failures.append("color_target_not_unique")
    elif paths and not truth_matches:
        failures.append("gt_color_does_not_match_summary_mapping")
    elif paths and not contradicted:
        failures.append("summary_color_mapping_not_contradicted")
    if paths and not injected_matches:
        failures.append("flawed_color_does_not_match_target")
    # matplotlib derives marker face/edge colors from a line's ``color`` unless
    # they are set explicitly, so a legitimate single ``color=`` edit surfaces
    # as three changed paths on the same series.  Those derived paths are part
    # of the target when they carry the same injected color, never collateral.
    derived = []
    for candidate in paths:
        root_prefix, _, field = candidate.rpartition(".")
        if field != "color":
            continue
        for suffix in ("markeredgecolor", "markerfacecolor"):
            derived_path = f"{root_prefix}.{suffix}"
            if derived_path in changed and derived_path not in paths and (
                injected is None
                or _color_matches_claim(after.flat.get(derived_path), injected)
            ):
                derived.append(derived_path)
    collateral = _collateral(
        changed, exact=tuple(paths) + tuple(derived), target=target,
    )
    return _base(
        "color-contradicts-summary", before, after,
        checks={"concrete_summary_mapping": truth_color is not None,
                "gt_matches_summary": truth_matches, "flawed_contradicts_summary": contradicted,
                "target_match": injected_matches},
        success_reason="summary_color_mapping_contradicted", failures=failures,
        missing_metadata=missing, collateral_paths=collateral,
        target_path=paths[0] if len(paths) == 1 else None, target_paths=paths,
        derived_color_paths=derived,
        entity=label, truth_color=truth_color,
        flawed_color=after_colors[0] if len(after_colors) == 1 else after_colors or None,
    )


def wrong_chart_type_postcondition(before: dict, after: dict, target=None, evidence=None) -> dict:
    before, after = _state(before), _state(after)
    target, _ = _metadata(target, evidence)
    bs, fs = _series(before), _series(after)
    common = set(bs) & set(fs)
    same_series = set(bs) == set(fs)
    data_preserved = same_series and all(_equal(bs[root].get("values"), fs[root].get("values"))
                                         for root in bs)
    changed_kind = [root for root in common
                    if not _equal(bs[root].get("kind"), fs[root].get("kind"))]
    labels = _labels(target)
    if labels:
        requested = {_series_key(bs, label, target.get("axis_index")) for label in labels}
        target_roots = sorted(root for root in requested if root)
    else:
        target_roots = changed_kind
    truth_kind = target.get("truth_kind") or target.get("before_kind")
    injected_kind = target.get("injected_kind") or target.get("after_kind")
    target_match = bool(target_roots) and all(
        root in changed_kind and _expected(bs[root].get("kind"), truth_kind)
        and _expected(fs[root].get("kind"), injected_kind)
        for root in target_roots
    )
    failures = []
    if not same_series:
        failures.append("series_added_or_dropped")
    if not data_preserved:
        failures.append("chart_type_changed_with_data_loss")
    if not changed_kind:
        failures.append("chart_type_not_changed")
    elif not target_match:
        failures.append("chart_type_does_not_match_target")
    changed = _changed(before, after)
    # A chart-type swap changes the renderer's artist class. Its normalized
    # representation therefore gains/loses class-specific style properties
    # even when the source edit changes only the constructor. Data are proved
    # separately above; non-target series and all chart-level paths stay strict.
    representation_fields = {*_SERIES_FIELDS, "rendered_colors"} - {"values"}
    collateral = _collateral(
        changed,
        predicate=lambda path: any(
            path == f"{root}.{field}"
            for root in target_roots for field in representation_fields
        ),
        ignored_suffixes=(".xlim", ".ylim", ".xticklabels", ".yticklabels"),
        target=target,
    )
    return _base(
        "wrong-chart-type", before, after,
        checks={"same_series": same_series, "all_data_preserved": data_preserved,
                "kind_changed": bool(changed_kind), "target_match": target_match},
        success_reason="chart_encoding_changed_data_preserved", failures=failures,
        collateral_paths=collateral, target_series=target_roots,
        kind_changes={root: [bs[root].get("kind"), fs[root].get("kind")]
                      for root in changed_kind},
    )


def _series_data_match(left: dict, right: dict) -> bool:
    return left["axis"] == right["axis"] and _equal(
        {"kind": left.get("kind", _MISSING), "values": left.get("values", _MISSING)},
        {"kind": right.get("kind", _MISSING), "values": right.get("values", _MISSING)},
    )


def _perfect_series_mapping(
    candidates: dict[str, list[str]], forbidden: tuple[str, str] | None = None,
) -> dict[str, str] | None:
    """Find one complete after->before matching, optionally excluding one edge."""
    matched_before: dict[str, str] = {}

    def assign(after_root: str, seen: set[str]) -> bool:
        for before_root in candidates[after_root]:
            if (after_root, before_root) == forbidden or before_root in seen:
                continue
            seen.add(before_root)
            previous = matched_before.get(before_root)
            if previous is None or assign(previous, seen):
                matched_before[before_root] = after_root
                return True
        return False

    for after_root in sorted(candidates, key=lambda root: (len(candidates[root]), root)):
        if not assign(after_root, set()):
            return None
    return {after_root: before_root for before_root, after_root in matched_before.items()}


def _unique_survivor_mapping(
    before: dict[str, dict], after: dict[str, dict],
) -> dict[str, str] | None:
    """Map re-keyed survivors by rendered data, rejecting ambiguous identity."""
    if len(before) != len(after) + 1:
        return None
    candidates = {
        after_root: sorted(
            before_root for before_root, before_series in before.items()
            if _series_data_match(before_series, after_series)
        )
        for after_root, after_series in after.items()
    }
    if any(not roots for roots in candidates.values()):
        return None
    mapping = _perfect_series_mapping(candidates)
    if mapping is None:
        return None
    # A second perfect matching would make the missing series' identity ambiguous.
    if any(
        _perfect_series_mapping(candidates, (after_root, before_root)) is not None
        for after_root, before_root in mapping.items()
    ):
        return None
    return mapping


def _legend_label_removals(before: _State, after: _State) -> list[tuple[str, str]]:
    """Return legend paths where the after sequence is exactly one label shorter."""
    removals = []
    after_groups = _legend_groups(after)
    for path, old_labels in _legend_groups(before).items():
        new_labels = after_groups.get(path)
        if new_labels is None or len(old_labels) != len(new_labels) + 1:
            continue
        matches = [
            old_labels[index]
            for index in range(len(old_labels))
            if _equal(old_labels[:index] + old_labels[index + 1:], new_labels)
        ]
        if len(matches) == 1:
            removals.append((path, matches[0]))
    return removals


def dropped_series_postcondition(before: dict, after: dict, target=None, evidence=None) -> dict:
    before, after = _state(before), _state(after)
    target, _ = _metadata(target, evidence)
    bs, fs = _series(before), _series(after)
    mapping = _unique_survivor_mapping(bs, fs)
    matched_before = set(mapping.values()) if mapping else set()
    missing_roots = sorted(set(bs) - matched_before) if mapping else []
    added_roots = sorted(set(fs) - set(mapping or {}))
    requested = target.get("series_label") or target.get("label")
    requested_root = _series_key(bs, requested, target.get("axis_index")) \
        if requested else None
    legend_removals = _legend_label_removals(before, after)
    matching_legend_removals = [
        (path, label) for path, label in legend_removals
        if requested is not None and _equal(label, requested)
    ]
    exactly_one = mapping is not None and len(missing_roots) == 1 and not added_roots
    if requested_root is not None:
        target_match = exactly_one and requested_root == missing_roots[0]
        if legend_removals:
            target_match = target_match and len(matching_legend_removals) == 1
    else:
        target_match = exactly_one and len(matching_legend_removals) == 1
    if not requested and exactly_one:
        target_match = True
    survivors_stable = mapping is not None and all(
        _equal(_identity(bs[before_root], values=False),
               _identity(fs[after_root], values=False))
        for after_root, before_root in mapping.items()
    )
    order_stable = mapping is not None and [
        list(bs).index(mapping[after_root]) for after_root in fs
    ] == sorted(list(bs).index(mapping[after_root]) for after_root in fs)
    survivors_stable = survivors_stable and order_stable
    failures = []
    if not exactly_one:
        failures.append("not_exactly_one_visible_series_dropped")
    if exactly_one and not target_match:
        failures.append("wrong_series_dropped")
    if not survivors_stable:
        failures.append("surviving_series_changed")
    changed = _changed(before, after)
    allowed_removals = matching_legend_removals if requested else legend_removals
    allowed_legend_paths = tuple(path for path, _ in allowed_removals)
    collateral = _collateral(
        changed, exact=allowed_legend_paths,
        predicate=lambda path: mapping is not None and _SERIES_RE.match(path) is not None,
        ignored_suffixes=(".xlim", ".ylim", ".xticklabels", ".yticklabels"),
        target=target,
    )
    return _base(
        "dropped-series", before, after,
        checks={"exactly_one_missing": exactly_one, "target_match": target_match,
                "surviving_series_identical": survivors_stable,
                "survivor_mapping_unique": mapping is not None,
                "survivor_order_stable": order_stable},
        success_reason="exactly_one_target_series_missing", failures=failures,
        collateral_paths=collateral, missing_series=missing_roots,
        added_series=added_roots, target_series=missing_roots[0] if exactly_one else None,
        survivor_mapping=mapping or {},
        removed_legend_labels=[{"path": path, "label": label}
                               for path, label in legend_removals],
    )


def _scalar_differences(left: Any, right: Any, path=()) -> list[tuple[tuple[int, ...], Any, Any]] | None:
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return None
        out = []
        for index, (old, new) in enumerate(zip(left, right)):
            diff = _scalar_differences(old, new, path + (index,))
            if diff is None:
                return None
            out.extend(diff)
        return out
    if _equal(left, right):
        return []
    if isinstance(left, (int, float)) and not isinstance(left, bool) and \
            isinstance(right, (int, float)) and not isinstance(right, bool):
        return [(path, left, right)]
    return None


def _value_visual_fraction(
    before: _State, after: _State, series: dict, index: tuple[int, ...],
    old: Any, new: Any, all_series: Mapping[str, dict],
    before_series: Mapping[str, Any] | None = None,
) -> tuple[float | None, str]:
    """Measure displacement against the conservative union of rendered spans."""
    kind = series.get("kind")
    if kind == "heatmap":
        values = series.get("values")
        if not isinstance(values, (list, tuple)) or len(index) != 2 or index[1] != 1:
            return None, "unsupported_value_encoding"
        for point in values:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                return None, "heatmap_values_malformed"
            value = point[1]
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(float(value)):
                return None, "heatmap_values_not_finite"
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(float(value)) for value in (old, new)):
            return None, "heatmap_values_not_finite"
        before_clim = _mapping(before_series).get("clim")
        after_clim = series.get("clim")
        if any(not isinstance(clim, (list, tuple)) or len(clim) != 2
               for clim in (before_clim, after_clim)):
            return None, "heatmap_color_limits_missing"
        endpoints = [*before_clim, *after_clim]
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(float(value)) for value in endpoints):
            return None, "heatmap_color_limits_not_finite"
        if not _equal(before_clim, after_clim):
            return None, "heatmap_color_limits_changed"
        span = float(after_clim[1]) - float(after_clim[0])
        scale = max(*(abs(float(value)) for value in endpoints), 1.0)
        if span <= scale * 1e-12:
            return None, "heatmap_color_span_not_positive"
        bounds = sorted(float(value) for value in after_clim)
        if not all(bounds[0] <= float(value) <= bounds[1] for value in (old, new)):
            return None, "heatmap_target_outside_color_limits"
        return abs(float(new) - float(old)) / span, f"{series['root']}.clim"

    axis_name = "x" if kind == "barh" else "y" if kind in {
        "area", "bar", "line", "scatter",
    } else None
    if axis_name is None or len(index) != 2 or index[1] != 1:
        return None, "unsupported_value_encoding"
    axis_root = f"axes[{series['axis']}]"
    scale_path = f"{axis_root}.{axis_name}scale"
    if before.flat.get(scale_path) != "linear" or after.flat.get(scale_path) != "linear":
        return None, "value_axis_not_known_linear"
    limit_path = f"{axis_root}.{axis_name}lim"
    before_limits = before.flat.get(limit_path)
    after_limits = after.flat.get(limit_path)
    if any(not isinstance(limits, (list, tuple)) or len(limits) != 2
           for limits in (before_limits, after_limits)):
        return None, "value_axis_limits_missing"
    values = (old, new, *before_limits, *after_limits)
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(float(value)) for value in values):
        return None, "value_axis_span_not_finite"
    if not _equal(before_limits, after_limits):
        return None, "value_axis_limits_changed"
    old_value, new_value = float(old), float(new)
    before_direction = float(before_limits[1]) - float(before_limits[0])
    after_direction = float(after_limits[1]) - float(after_limits[0])
    before_span = abs(before_direction)
    after_span = abs(after_direction)
    scale = max(*(abs(float(value)) for value in (*before_limits, *after_limits)), 1.0)
    if before_span <= scale * 1e-12 or after_span <= scale * 1e-12:
        return None, "value_axis_span_not_positive"
    if before_direction * after_direction < 0:
        return None, "value_axis_direction_changed"
    before_bounds = sorted(float(value) for value in before_limits)
    after_bounds = sorted(float(value) for value in after_limits)
    endpoints = [*before_bounds, *after_bounds]
    span = max(endpoints) - min(endpoints)
    if span <= max(*(abs(value) for value in endpoints), 1.0) * 1e-12:
        return None, "value_axis_span_not_positive"
    tolerance = span * 1e-12
    if not before_bounds[0] - tolerance <= old_value <= before_bounds[1] + tolerance \
            or not after_bounds[0] - tolerance <= new_value <= after_bounds[1] + tolerance:
        return None, "target_value_outside_visible_axis"
    for candidate in all_series.values():
        candidate_axis = "x" if candidate.get("kind") == "barh" else "y" if \
            candidate.get("kind") in {"area", "bar", "line", "scatter"} else None
        if candidate.get("axis") != series.get("axis") or candidate_axis != axis_name:
            continue
        candidate_values = candidate.get("values")
        if not isinstance(candidate_values, (list, tuple)):
            return None, "unchanged_value_visibility_unproven"
        for point_position, point in enumerate(candidate_values):
            if candidate.get("root") == series.get("root") and \
                    (point_position, 1) == index:
                continue
            if not isinstance(point, (list, tuple)) or len(point) != 2 or \
                    isinstance(point[1], bool) or not isinstance(point[1], (int, float)):
                return None, "unchanged_value_visibility_unproven"
            value = float(point[1])
            visible_before = (
                before_bounds[0] - tolerance <= value <= before_bounds[1] + tolerance
            )
            visible_after = (
                after_bounds[0] - tolerance <= value <= after_bounds[1] + tolerance
            )
            if visible_before != visible_after:
                return None, (
                    "unchanged_value_clipped"
                    if visible_before else "unchanged_value_revealed"
                )
    return abs(new_value - old_value) / span, f"{limit_path}:union"


def _source_ast_binding_fields_valid(
    binding: Mapping[str, Any], point_index: int, truth: Any, injected: Any,
) -> bool:
    series_path = binding.get("source_series_ast_path")
    value_path = binding.get("source_value_ast_path")
    valid_segments = lambda path: isinstance(path, list) and bool(path) and all(
        isinstance(segment, str) or (
            isinstance(segment, int) and not isinstance(segment, bool)
            and segment >= 0
        )
        for segment in path
    )
    return bool(
        valid_segments(series_path)
        and valid_segments(value_path)
        and value_path == [*series_path, "elts", point_index]
        and _equal(binding.get("source_value_literal"), truth)
        and _equal(binding.get("injected_value"), injected)
    )


def _heatmap_target_binding(
    before: _State, after: _State, series: dict, index: tuple[int, ...], requested: Any,
    point_index: Any, truth: Any, injected: Any, source_binding: Any,
) -> str | None:
    """Match the sole candidate diff to the orchestrator's immutable GT binding."""
    values = series.get("values")
    if not requested or not isinstance(values, (list, tuple)) or \
            len(index) != 2 or index[1] != 1:
        return None
    flat_index = _target_index(index[0])
    target_point_index = _target_index(point_index)
    binding = _mapping(source_binding)
    source_length = _target_index(binding.get("source_series_length"))
    bound_point_index = _target_index(binding.get("point_index"))
    bound_axis = _target_index(binding.get("axis_index"))
    bound_row = _target_index(binding.get("row"))
    bound_column = _target_index(binding.get("column"))
    bound_flat_index = _target_index(binding.get("flat_index"))
    locator_label = binding.get("locator_label")
    dimension = binding.get("requested_dimension")
    render = _mapping(binding.get("render"))
    hashes_valid = all(
        isinstance(binding.get(name), str)
        and re.fullmatch(r"[0-9a-f]{64}", binding[name]) is not None
        for name in ("source_code_sha256", "table_sha256", "gt_capture_sha256")
    )
    if flat_index is None or flat_index >= len(values) or \
            target_point_index is None or binding.get("schema") != \
            VALUE_SOURCE_BINDING_SCHEMA or binding.get("ok") is not True or \
            bound_point_index != target_point_index or \
            not _equal(binding.get("series_label"), requested) or \
            not _equal(binding.get("truth_value"), truth) or \
            not _source_ast_binding_fields_valid(
                binding, target_point_index, truth, injected,
            ) or \
            source_length is None or target_point_index >= source_length or \
            not isinstance(locator_label, str) or not locator_label.strip() or \
            dimension not in ("x", "y") or bound_axis != series.get("axis") or \
            binding.get("render_kind") != "heatmap" or \
            bound_row is None or bound_column is None or \
            bound_flat_index != flat_index or binding.get("series_root") != \
            series.get("root") or binding.get("values_path") != \
            f"{series.get('root')}.values" or not hashes_valid or \
            render != {
                "render_kind": "heatmap",
                "series_root": binding.get("series_root"),
                "row": bound_row,
                "column": bound_column,
                "flat_index": bound_flat_index,
                "values_path": binding.get("values_path"),
            }:
        return None
    axis_root = f"axes[{series['axis']}]"
    x_path = f"{axis_root}.xticklabels"
    y_path = f"{axis_root}.yticklabels"
    xlabels = before.flat.get(x_path)
    ylabels = before.flat.get(y_path)
    if not isinstance(xlabels, (list, tuple)) or not xlabels or \
            not isinstance(ylabels, (list, tuple)) or not ylabels or \
            not _equal(xlabels, after.flat.get(x_path)) or \
            not _equal(ylabels, after.flat.get(y_path)) or \
            len({_normalized_table_text(value) for value in xlabels}) != len(xlabels) or \
            len({_normalized_table_text(value) for value in ylabels}) != len(ylabels) or \
            None in {_normalized_table_text(value) for value in (*xlabels, *ylabels)}:
        return None
    before_series = _series(before).get(series["root"])
    if before_series is None or before_series.get("kind") != "heatmap" or \
            not _canonical_heatmap(before_series, len(xlabels), len(ylabels)) or \
            not _canonical_heatmap(series, len(xlabels), len(ylabels)) or \
            bound_row >= len(ylabels) or bound_column >= len(xlabels) or \
            bound_flat_index != bound_row * len(xlabels) + bound_column or \
            not _equal(before_series["values"][bound_flat_index][1], truth):
        return None
    if dimension == "x":
        if _equal(xlabels[bound_column], requested) and \
                _equal(ylabels[bound_row], locator_label):
            return x_path
    elif _equal(ylabels[bound_row], requested) and \
            _equal(xlabels[bound_column], locator_label):
        return y_path
    return None


def _heatmap_annotation_change(
    before: _State, after: _State, series: dict, index: tuple[int, ...],
    old: Any, new: Any,
) -> tuple[bool, str | None]:
    """Allow only a cell-aligned annotation text update derived from the value."""
    path = f"axes[{series['axis']}].annotations"
    old_annotations = before.flat.get(path, _MISSING)
    new_annotations = after.flat.get(path, _MISSING)
    if old_annotations is _MISSING and new_annotations is _MISSING:
        return True, None
    values = series.get("values")
    if not isinstance(old_annotations, (list, tuple)) or \
            not isinstance(new_annotations, (list, tuple)) or \
            not isinstance(values, (list, tuple)) or \
            len(old_annotations) != len(new_annotations) or \
            len(old_annotations) != len(values) or len(index) != 2 or index[1] != 1:
        return False, None
    cell_index = index[0]
    if not isinstance(cell_index, int) or not 0 <= cell_index < len(old_annotations):
        return False, None
    changed = [
        position for position, (left, right) in enumerate(
            zip(old_annotations, new_annotations)
        ) if not _equal(left, right)
    ]
    if changed != [cell_index]:
        return False, None
    old_annotation = old_annotations[cell_index]
    new_annotation = new_annotations[cell_index]
    if not isinstance(old_annotation, (list, tuple)) or \
            not isinstance(new_annotation, (list, tuple)) or \
            len(old_annotation) < 1 or len(old_annotation) != len(new_annotation) or \
            not _equal(old_annotation[1:], new_annotation[1:]):
        return False, None
    try:
        old_text = float(old_annotation[0])
        new_text = float(new_annotation[0])
    except (TypeError, ValueError):
        return False, None
    if not math.isfinite(old_text) or not math.isfinite(new_text) or \
            not _equal(old_text, old) or not _equal(new_text, new):
        return False, None
    return True, path


def _cartesian_target_binding(
    series: dict, requested: Any, point_index: Any, truth: Any, injected: Any,
    source_binding: Any,
) -> bool:
    binding = _mapping(source_binding)
    target_index = _target_index(point_index)
    source_length = _target_index(binding.get("source_series_length"))
    bound_index = _target_index(binding.get("point_index"))
    bound_axis = _target_index(binding.get("axis_index"))
    row = _target_index(binding.get("row"))
    column = _target_index(binding.get("column"))
    flat_index = _target_index(binding.get("flat_index"))
    render = _mapping(binding.get("render"))
    values = series.get("values")
    hashes_valid = all(
        isinstance(binding.get(name), str)
        and re.fullmatch(r"[0-9a-f]{64}", binding[name]) is not None
        for name in ("source_code_sha256", "table_sha256", "gt_capture_sha256")
    )
    return bool(
        binding.get("schema") == VALUE_SOURCE_BINDING_SCHEMA
        and binding.get("ok") is True
        and binding.get("render_kind") == "cartesian"
        and target_index is not None
        and bound_index == target_index
        and source_length is not None
        and target_index < source_length
        and _equal(binding.get("series_label"), requested)
        and _equal(binding.get("truth_value"), truth)
        and _source_ast_binding_fields_valid(
            binding, target_index, truth, injected,
        )
        and isinstance(binding.get("locator_label"), str)
        and bool(binding["locator_label"].strip())
        and bound_axis == series.get("axis")
        and binding.get("series_root") == series.get("root")
        and binding.get("values_path") == f"{series.get('root')}.values"
        and row == target_index
        and column == 1
        and flat_index == target_index
        and render == {
            "render_kind": "cartesian",
            "series_root": binding.get("series_root"),
            "row": row,
            "column": column,
            "flat_index": flat_index,
            "values_path": binding.get("values_path"),
        }
        and hashes_valid
        and isinstance(values, (list, tuple))
        and target_index < len(values)
        and isinstance(values[target_index], (list, tuple))
        and len(values[target_index]) == 2
        and _equal(values[target_index][1], truth)
    )


def _heatmap_rendered_color_change(
    before: _State, after: _State, series: dict, index: tuple[int, ...],
) -> tuple[bool, bool, str | None, float | None]:
    """Prove only the target cell's effective rendered color may change."""
    path = f"{series['root']}.rendered_colors"
    old_colors = before.flat.get(path, _MISSING)
    new_colors = after.flat.get(path, _MISSING)
    if old_colors is _MISSING and new_colors is _MISSING:
        return True, False, None, None
    values = series.get("values")
    if not isinstance(old_colors, (list, tuple)) or \
            not isinstance(new_colors, (list, tuple)) or \
            not isinstance(values, (list, tuple)) or \
            len(old_colors) != len(new_colors) or len(old_colors) != len(values) or \
            len(index) != 2 or index[1] != 1:
        return False, False, path, None
    target_index = _target_index(index[0])
    if target_index is None or target_index >= len(old_colors) or any(
        not _equal(old, new)
        for position, (old, new) in enumerate(zip(old_colors, new_colors))
        if position != target_index
    ):
        return False, False, path, None

    def effective_rgb(color: Any) -> tuple[float, float, float] | None:
        if not isinstance(color, (list, tuple)) or len(color) != 4 or any(
            isinstance(channel, bool) or not isinstance(channel, (int, float))
            or not math.isfinite(float(channel)) or not 0 <= float(channel) <= 1
            for channel in color
        ):
            return None
        alpha = float(color[3])
        return tuple(float(channel) * alpha for channel in color[:3])

    old_rgb = effective_rgb(old_colors[target_index])
    new_rgb = effective_rgb(new_colors[target_index])
    if old_rgb is None or new_rgb is None:
        return False, False, path, None
    distance = math.sqrt(sum((old - new) ** 2 for old, new in zip(old_rgb, new_rgb))) \
        / math.sqrt(3)
    return True, distance + 1e-12 >= MIN_HEATMAP_COLOR_DISTANCE, path, distance


def value_perturbation_postcondition(before: dict, after: dict, target=None, evidence=None) -> dict:
    before, after = _state(before), _state(after)
    target, evidence = _metadata(target, evidence)
    bs, fs = _series(before), _series(after)
    same_series = set(bs) == set(fs)
    diffs = []
    shape_ok = same_series
    if same_series:
        for root in bs:
            delta = _scalar_differences(bs[root].get("values"), fs[root].get("values"))
            if delta is None:
                shape_ok = False
                continue
            diffs.extend((root, index, old, new) for index, old, new in delta)
    exactly_one = shape_ok and len(diffs) == 1
    requested = target.get("series_label") or target.get("label")
    truth = target.get("truth_value", target.get("truth"))
    injected = target.get("injected_value", target.get("injected"))
    explicit_pair = (
        ("truth_value" in target or "truth" in target)
        and ("injected_value" in target or "injected" in target)
        and isinstance(truth, (int, float)) and not isinstance(truth, bool)
        and isinstance(injected, (int, float)) and not isinstance(injected, bool)
        and math.isfinite(float(truth)) and math.isfinite(float(injected))
    )
    target_root = _series_key(bs, requested, target.get("axis_index")) if requested else None
    heatmap_mode = exactly_one and fs[diffs[0][0]].get("kind") == "heatmap"
    target_resolution = "series_root"
    heatmap_axis_binding = None
    source_binding_ok = False
    if heatmap_mode:
        root, index, old, new = diffs[0]
        target_root = root
        target_resolution = "heatmap_axis_tick"
        heatmap_axis_binding = _heatmap_target_binding(
            before, after, fs[root], index, requested, target.get("point_index"),
            truth, injected, evidence.get("_value_source_binding"),
        )
        source_binding_ok = bool(heatmap_axis_binding)
        axis_matches = True
        if target.get("axis_index") is not None:
            axis_matches = _target_index(target["axis_index"]) == fs[root]["axis"]
        target_match = bool(
            heatmap_axis_binding and axis_matches and explicit_pair
            and _equal(old, truth) and _equal(new, injected)
        )
    else:
        point_index = _target_index(target.get("point_index"))
        source_binding_ok = bool(
            exactly_one and requested and explicit_pair
            and _cartesian_target_binding(
                bs[diffs[0][0]], requested, target.get("point_index"), truth,
                injected, evidence.get("_value_source_binding"),
            )
        )
        target_resolution = "source_table_series"
        target_match = bool(
            exactly_one and requested and explicit_pair
            and diffs[0][0] == target_root
            and point_index is not None
            and source_binding_ok
        )
        if target_match:
            target_match = bool(diffs[0][1] and diffs[0][1][0] == point_index)
        if target_match and "component_index" in target:
            component_index = _target_index(target.get("component_index"))
            target_match = bool(
                component_index is not None
                and len(diffs[0][1]) > 1
                and diffs[0][1][1] == component_index
            )
        if target_match:
            target_match = _equal(diffs[0][2], truth) and \
                _equal(diffs[0][3], injected)
    nonvalues_stable = same_series and all(
        _equal(_identity(bs[root], values=False), _identity(fs[root], values=False)) for root in bs
    )
    visual_fraction = None
    visibility_basis = "target_not_proven"
    if exactly_one and target_match:
        root, index, old, new = diffs[0]
        visual_fraction, visibility_basis = _value_visual_fraction(
            before, after, fs[root], index, old, new, fs, bs.get(root),
        )
    annotation_ok = True
    annotation_path = None
    rendered_color_integrity = True
    rendered_color_visible = False
    rendered_color_path = None
    rendered_color_distance = None
    if heatmap_mode:
        root, index, old, new = diffs[0]
        annotation_ok, annotation_path = _heatmap_annotation_change(
            before, after, fs[root], index, old, new,
        )
        (rendered_color_integrity, rendered_color_visible, rendered_color_path,
         rendered_color_distance) = _heatmap_rendered_color_change(
            before, after, fs[root], index,
        )
    heatmap_visibly_rendered = (
        not heatmap_mode or annotation_path is not None or rendered_color_visible
    )
    visible_enough = visual_fraction is not None and \
        visual_fraction + 1e-12 >= MIN_VALUE_VISUAL_FRACTION
    failures = []
    if not exactly_one:
        failures.append("not_exactly_one_numeric_value_changed")
    elif not target_match:
        failures.append("changed_value_does_not_match_target")
    elif visual_fraction is None:
        failures.append("value_visibility_unproven")
    elif not visible_enough:
        failures.append("value_change_below_visual_threshold")
    if not annotation_ok:
        failures.append("derived_heatmap_annotation_mismatch")
    if not rendered_color_integrity:
        failures.append("heatmap_rendered_colors_changed_elsewhere")
    elif not heatmap_visibly_rendered:
        failures.append("heatmap_change_not_visibly_rendered")
    if not nonvalues_stable:
        failures.append("series_encoding_changed")
    allowed_paths = tuple(
        path for path in (f"{target_root}.values" if target_root else None,
                          annotation_path, rendered_color_path) if path
    )
    collateral = _collateral(
        _changed(before, after),
        exact=allowed_paths,
        target=target,
    )
    changed_value = None
    if exactly_one:
        root, index, old, new = diffs[0]
        changed_value = {"series": root, "index": list(index), "truth": old, "flawed": new}
    return _base(
        "value-perturbation", before, after,
        checks={"same_series": same_series, "same_value_shapes": shape_ok,
                "exactly_one_numeric_change": exactly_one, "target_match": target_match,
                "source_binding_ok": source_binding_ok,
                "other_series_semantics_stable": nonvalues_stable,
                "visible_axis_fraction_at_least_0_10": visible_enough,
                "derived_heatmap_annotation_exact": annotation_ok,
                "heatmap_rendered_color_integrity": rendered_color_integrity,
                "heatmap_visibly_rendered": heatmap_visibly_rendered},
        success_reason="exactly_one_target_value_changed", failures=failures,
        collateral_paths=collateral, changed_value=changed_value,
        visual_fraction=visual_fraction, visibility_basis=visibility_basis,
        minimum_visual_fraction=MIN_VALUE_VISUAL_FRACTION,
        heatmap_rendered_color_distance=rendered_color_distance,
        minimum_heatmap_color_distance=MIN_HEATMAP_COLOR_DISTANCE,
        target_resolution=target_resolution,
        heatmap_axis_binding=heatmap_axis_binding,
    )


def swapped_series_postcondition(before: dict, after: dict, target=None, evidence=None) -> dict:
    before, after = _state(before), _state(after)
    target, _ = _metadata(target, evidence)
    bs, fs = _series(before), _series(after)

    def candidates():
        if set(bs) != set(fs):
            return []
        out = []
        for left, right in itertools.combinations(sorted(bs), 2):
            if _equal(fs[left].get("values"), bs[right].get("values")) and \
                    _equal(fs[right].get("values"), bs[left].get("values")) and \
                    not _equal(bs[left].get("values"), bs[right].get("values")) and \
                    all(_equal(bs[root].get("values"), fs[root].get("values"))
                        for root in bs if root not in (left, right)):
                out.append((left, right))
        return out

    labels = _labels(target)
    if len(labels) == 2:
        left = _series_key(bs, labels[0], target.get("axis_index"))
        right = _series_key(bs, labels[1], target.get("axis_index"))
        pairs = [(left, right)] if left and right else []
    elif not labels:
        pairs = candidates()
        if len(pairs) == 1:
            labels = [bs[pairs[0][0]]["label"], bs[pairs[0][1]]["label"]]
    exact_swap = False
    encodings_stable = False
    if len(pairs) == 1:
        left, right = pairs[0]
        exact_swap = (left, right) in candidates() or (right, left) in candidates()
        encodings_stable = all(
            _equal(_identity(bs[root], values=False), _identity(fs[root], values=False))
            for root in bs
        )
    failures = []
    if len(pairs) != 1:
        failures.append("series_swap_target_not_unique")
    elif not exact_swap:
        failures.append("two_data_assignments_not_exactly_swapped")
    if not encodings_stable:
        failures.append("labels_or_encodings_changed")
    exact = tuple(f"{root}.values" for root in pairs[0]) if len(pairs) == 1 else ()
    collateral = _collateral(
        _changed(before, after), exact=exact,
        ignored_suffixes=(".xlim", ".ylim", ".xticklabels", ".yticklabels"),
        target=target,
    )
    return _base(
        "swapped-series", before, after,
        checks={"two_target_series": len(pairs) == 1, "exact_value_swap": exact_swap,
                "labels_and_encodings_stable": encodings_stable},
        success_reason="two_target_data_assignments_swapped", failures=failures,
        collateral_paths=collateral, labels=labels,
        target_series=list(pairs[0]) if len(pairs) == 1 else [],
    )


def _coordinate_points(root: str, series: dict, axis: str) -> list[dict[str, Any]]:
    values = series.get("values") or []
    kind = series.get("kind")
    component = 1 if (kind == "barh" and axis == "x") else \
        0 if kind == "barh" else 0 if axis == "x" else 1
    points = []
    for point_index, point in enumerate(values):
        if not isinstance(point, (list, tuple)) or len(point) <= component:
            continue
        value = point[component]
        if isinstance(value, (int, float)) and not isinstance(value, bool) and \
                math.isfinite(float(value)):
            points.append({
                "series_root": root,
                "point_index": point_index,
                "value": float(value),
            })
    return points


def _inside(value: float, limits: Any) -> bool:
    if not isinstance(limits, (list, tuple)) or len(limits) != 2:
        return False
    try:
        low, high = sorted((float(limits[0]), float(limits[1])))
    except (TypeError, ValueError):
        return False
    tolerance = max(abs(low), abs(high), abs(value), 1.0) * 1e-9
    return low - tolerance <= value <= high + tolerance


def _limit_derived_tick_changes(
    before: _State, after: _State, target_path: str | None,
) -> tuple[bool, tuple[str, ...]]:
    """Allow only unchanged automatic tick policies on the target axis."""
    match = re.fullmatch(r"(axes\[\d+\])\.([xy])lim", str(target_path))
    changed = set(_changed(before, after))
    if not match:
        has_tick_change = any(
            path.endswith((".xticks", ".yticks", ".xticklabels", ".yticklabels"))
            for path in changed
        )
        return not has_tick_change, ()

    axis_root, axis = match.groups()
    tick_path = f"{axis_root}.{axis}ticks"
    label_path = f"{axis_root}.{axis}ticklabels"
    derived_paths = tuple(
        path for path in (tick_path, label_path) if path in changed
    )
    if not derived_paths:
        return True, ()

    policy_path = f"{axis_root}.{axis}tick_policy"
    before_policy = before.flat.get(policy_path)
    after_policy = after.flat.get(policy_path)
    if not isinstance(before_policy, Mapping) or not _equal(
        before_policy, after_policy,
    ):
        return False, ()
    locator = before_policy.get("locator")
    formatter = before_policy.get("formatter")
    if not isinstance(locator, Mapping) or not isinstance(formatter, Mapping):
        return False, ()
    if tick_path in derived_paths and locator.get("automatic") is not True:
        return False, ()
    if label_path in derived_paths and formatter.get("automatic") is not True:
        return False, ()
    return True, derived_paths


def axis_truncation_postcondition(before: dict, after: dict, target=None, evidence=None) -> dict:
    before, after = _state(before), _state(after)
    target, _ = _metadata(target, evidence)
    bs, fs = _series(before), _series(after)
    candidates = []
    for path in _changed(before, after):
        match = re.fullmatch(r"axes\[(\d+)\]\.(xlim|ylim)", path)
        if not match:
            continue
        axis_index, name = int(match.group(1)), match.group(2)
        axis = name[0]
        points = [
            point for root, rec in bs.items() if rec["axis"] == axis_index
            for point in _coordinate_points(root, rec, axis)
        ]
        coordinates = [point["value"] for point in points]
        gt_limits = before.flat.get(path, target.get("gt_limits"))
        flawed_limits = after.flat.get(path)
        gt_visible = bool(coordinates) and all(_inside(value, gt_limits) for value in coordinates)
        clipped_points = [
            point for point in points if not _inside(point["value"], flawed_limits)
        ]
        clipped = [point["value"] for point in clipped_points]
        if gt_visible and clipped:
            candidates.append((
                path, axis_index, axis, coordinates, clipped, gt_limits,
                flawed_limits, clipped_points,
            ))

    requested_path = target.get("path")
    if not requested_path and target.get("axis") in ("x", "y"):
        try:
            requested_path = f"axes[{int(target.get('axis_index', 0))}].{target['axis']}lim"
        except (TypeError, ValueError):
            requested_path = None
    if requested_path:
        candidates = [candidate for candidate in candidates if candidate[0] == requested_path]
    unique = len(candidates) == 1
    same_series = set(bs) == set(fs)
    data_unchanged = same_series and all(_equal(bs[root].get("values"), fs[root].get("values"))
                                         for root in bs)
    encodings_unchanged = same_series and all(
        _equal(_identity(bs[root], values=False), _identity(fs[root], values=False)) for root in bs
    )
    failures = []
    if not unique:
        failures.append("clipped_axis_target_not_unique" if candidates
                        else "no_truth_value_clipped")
    if not data_unchanged:
        failures.append("data_changed_or_deleted")
    if not encodings_unchanged:
        failures.append("series_encoding_changed")
    path = candidates[0][0] if unique else requested_path
    ticks_derived, derived_tick_paths = _limit_derived_tick_changes(
        before, after, path,
    )
    if not ticks_derived:
        failures.append("tick_changes_not_limit_derived")
    collateral = _collateral(
        _changed(before, after),
        exact=tuple(item for item in (path, *derived_tick_paths) if item),
        target=target,
    )
    clipped_values = candidates[0][4] if unique else []
    return _base(
        "axis-truncation", before, after,
        checks={"one_target_axis": unique,
                "gt_shows_all_truth_values": unique and bool(candidates[0][3]),
                "flawed_clips_truth_value": bool(clipped_values),
                "data_unchanged": data_unchanged,
                "series_encoding_unchanged": encodings_unchanged,
                "tick_changes_limit_derived": ticks_derived},
        success_reason="truth_value_outside_visible_axis", failures=failures,
        collateral_paths=collateral, target_path=path,
        derived_tick_paths=list(derived_tick_paths),
        gt_limits=candidates[0][5] if unique else None,
        flawed_limits=candidates[0][6] if unique else None,
        clipped_values=clipped_values,
        clipped_points=candidates[0][7] if unique else [],
    )


SEMANTIC_GATES = {
    "axis-truncation": axis_truncation_postcondition,
    "color-contradicts-summary": color_contradicts_summary_postcondition,
    "dropped-series": dropped_series_postcondition,
    "missing-legend": missing_legend_postcondition,
    "swapped-legend-labels": swapped_legend_labels_postcondition,
    "swapped-series": swapped_series_postcondition,
    "value-perturbation": value_perturbation_postcondition,
    "wrong-axis-label": wrong_axis_label_postcondition,
    "wrong-chart-type": wrong_chart_type_postcondition,
    "wrong-title": wrong_title_postcondition,
}


def category_postcondition(
    category: str, before: dict, after: dict, target=None, evidence=None,
):
    """Run one semantic gate; return ``None`` only for an unknown category.

    The first three parameters retain the v1.2 call shape. V2 callers should
    additionally pass the persisted structured ``target`` and ``evidence``.
    """
    gate = SEMANTIC_GATES.get(category)
    return gate(before, after, target, evidence) if gate else None
