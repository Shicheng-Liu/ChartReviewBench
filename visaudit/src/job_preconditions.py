"""Static preconditions: is one (sample, category) pair worth a single API call?

`transforms.perturb_all` answers a stricter question than we need here, namely
"can I rewrite this exact AST node?", because the deterministic path has to
splice source text. The LLM path rewrites the whole script and declines on its
own, so gating it on the deterministic candidate silently discards instances
that are perfectly buildable: `t_ylim_truncation` only fires when the ChartNet
label contains "Bar", and `t_dropped_series` only on >=2 plot calls that each
carry a literal `label=`, which misses every seaborn `hue=`, every melted frame,
every `df.plot()` and every plotly `add_trace`.

The trade is deliberately asymmetric, and that asymmetry is the whole design of
this module: a false positive costs one API call and a clean decline from the
model, a false negative costs an instance we can never build. So every check
below is PERMISSIVE. It asks whether the category is meaningful for this chart,
never whether one particular edit is mechanically possible.

Permissive is not vacuous. Each precondition stays true-by-construction about
its own category, so a pair that passes really can carry that flaw:
  * axis-truncation uses a supported continuous value axis with at least two
    distinct numeric values and no range the script already pins. Plotly is
    excluded until its capture exposes computed auto-ranges, and figures that
    are statically known to contain several axes are excluded because the V2
    injector deliberately targets one-axis figures only. This is still only a
    cheap prefilter: the binding V2 planner reads the GT capture and proves that
    a real endpoint is outside the flawed range while data stays unchanged. A
    merely raised zero baseline is no longer a valid flaw.
  * dropped-series and swapped-legend-labels need real evidence of >=2 series
    and series names the summary or the table actually mentions, otherwise the
    flaw is undetectable and the instance is worthless.

  * the four plotly fallbacks below each need the feature they perturb to be
    written literally in the script, so a pair that passes really does render a
    title / an axis title / a legend / a swappable mark.

Categories that already clear the 100-instance target with room to spare are not
loosened at all: they delegate to the deterministic candidate check so nothing
regresses.

The one exception, and the reason this module is not purely a three-category
widening: the delegation is library-blind but the transforms are not. Four of
the seven delegated categories match only the matplotlib/seaborn spelling of
their feature -- `_title_candidates` wants a positional string on
title/suptitle/set_title or a `title=` on `set`/`update`, `_label_candidates`
the same for xlabel/ylabel, `t_missing_legend` an `ax.legend()` statement or a
seaborn `hue=`, `t_wrong_chart_type` a name from the mpl/sns swap tables. Plotly
spells all four as layout keywords, so all four scored 0 plotly jobs out of 382
renderable plotly scripts while the other six scored 1345. That is a
library-correlated hole in a benchmark that claims to cover plotly, not a
property of the data: measured over those 382 scripts, 382 carry a literal
title, 256 an axis title or an express `labels={}` mapping, 292 a legend over
>=2 series the summary or table names, and 342 a mark with an honest swap
partner. `_PLOTLY_FALLBACK` closes that hole, and only that hole:

  * it runs ONLY after the deterministic answer was already "no target", so
    every pair the transforms propose is still proposed, unchanged;
  * it runs ONLY on a script that imports plotly. No matplotlib or seaborn
    sample in the corpus does (383/383 plotly, 0/617 others), so those verdicts
    are unchanged by construction, not by luck.
"""
import ast
import csv
import io
import random
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transforms

CATEGORIES = (
    "wrong-title",
    "wrong-axis-label",
    "missing-legend",
    "swapped-legend-labels",
    "color-contradicts-summary",
    "wrong-chart-type",
    "dropped-series",
    "axis-truncation",
    "value-perturbation",
    "swapped-series",
)

# Every one of these has >=180 deterministic candidates today, far past the 100
# we need, so they keep exactly the behaviour they have.
_DELEGATED = {
    "wrong-title": transforms.t_wrong_title,
    "wrong-axis-label": transforms.t_wrong_axis_label,
    "missing-legend": transforms.t_missing_legend,
    "color-contradicts-summary": transforms.t_color_contradicts_summary,
    "wrong-chart-type": transforms.t_wrong_chart_type,
    "value-perturbation": transforms.t_value_perturbation,
    "swapped-series": transforms.t_swapped_series,
}
LOOSE_CATEGORIES = ("swapped-legend-labels", "dropped-series", "axis-truncation")
# Delegated categories that get a second, plotly-only look when the transform
# finds nothing. Filled in below, once the checks exist.
PLOTLY_FALLBACK_CATEGORIES = ("wrong-title", "wrong-axis-label", "missing-legend",
                              "wrong-chart-type")

# t_wrong_title needs a donor title only to fill in the replacement string, and
# build_jobs always hands it ~1000 of them. One placeholder gives the same
# yes/no answer without forcing every caller to collect the corpus first.
_PLACEHOLDER_DONORS = ["Untitled Chart"]

# Marks whose captured coordinates support the V2 planner. Bars/areas contribute
# their value axis; lines and scatters contribute both continuous axes because
# V2 deliberately clips real endpoints instead of merely raising a baseline.
_VBAR_COMPS = {"bar", "barplot", "countplot"}
_AREA_COMPS = {"fill_between", "stackplot", "area", "stem", "funnel", "waterfall"}
_POINT_COMPS = {"plot", "line", "lineplot", "scatter", "scatterplot"}
_VBAR_KINDS = {"bar", "column"}
_HBAR_KINDS = {"barh"}
# Zero-baselined too, but the bar heights are counts the renderer computes, so
# they stand in neither the script nor table.csv. Worse, the only numbers we can
# read statically are the samples on the OTHER axis, so the spread test would
# measure the wrong axis. Excluded until the evidence model covers derived values.
_HIST_COMPS = {"hist", "histplot", "histogram", "hist2d"}

# Chart families with no continuous value axis at all: a raised floor is either
# meaningless or invisible there.
_NO_VALUE_AXIS = ("pie", "donut", "ring", "treemap", "sunburst", "wordcloud", "word cloud")

# Constructs that already pin an axis range, per axis. Matching on the AST and
# not on substrings matters: `DataFrame.set_axis` only renames columns, and
# pinning xlim leaves a vertical bar chart's value axis free.
_LIMIT_CALLS = {"y": {"set_ylim", "ylim", "set_rlim"}, "x": {"set_xlim", "xlim"}}
_LIMIT_KWARGS = {"y": {"ylim", "yaxis_range"}, "x": {"xlim", "xaxis_range"}}
_LIMIT_DICTS = {"y": {"yaxis", "radialaxis"}, "x": {"xaxis"}}
_LIMIT_UPDATERS = {"y": {"update_yaxes", "update_polars"}, "x": {"update_xaxes"}}

# Keyword arguments that map a column onto a visual channel, i.e. that turn one
# call into several series (seaborn and plotly express).
_GROUP_KWARGS = ("hue", "style", "col", "row", "color", "symbol", "line_dash",
                 "facet_col", "facet_row", "pattern_shape", "size")

_MAX_CSV_ROWS = 500

# --------------------------------------------------------- plotly spellings
# The keyword spellings below were read off the 382 renderable plotly scripts
# (see the module docstring), not off the plotly docs, which is what keeps the
# near-misses out: `title_font=`, `title_x=`, `title_pad=` style a title but are
# not one, and `legend_title_text=` titles the legend, not the chart. The two
# swap tables at the end are the exception and say so: they cover the marks the
# corpus draws plus their obvious siblings, so a later shard does not silently
# lose a chart family.

# Layout keywords that ARE the chart title. `title=` also takes a styling dict,
# in which case only its "text" key is the title (14 scripts pass a dict that
# only positions the title set by an express `title=`).
_PLOTLY_TITLE_KWARGS = ("title", "title_text")
# Axis-title spellings, per axis. `xaxis`/`yaxis` (and the `xaxis2`... of a
# subplot figure) take a dict whose "title" is either the string or {"text": s}.
_PLOTLY_AXIS_KWARGS = {"x": ("xaxis_title", "xaxis_title_text"),
                       "y": ("yaxis_title", "yaxis_title_text")}
_PLOTLY_AXIS_UPDATERS = {"update_xaxes": "x", "update_yaxes": "y"}
# Express kinds with no cartesian x/y axis: their `labels={}` renames legend and
# hover entries, never an axis title.
_PX_NO_XY = {"pie", "treemap", "sunburst", "icicle", "funnel_area",
             "line_polar", "scatter_polar", "bar_polar", "line_ternary",
             "scatter_ternary", "scatter_geo", "line_geo", "choropleth"}
# Legend evidence, layout level and trace level.
_PLOTLY_LEGEND_KWARGS = ("legend", "legend_title", "legend_title_text")
# Express channels that map a column onto a discrete legend.
_PX_LEGEND_CHANNELS = ("color", "symbol", "line_dash", "pattern_shape")

# Swap partners: a different mark that plots the SAME data, so the chart lies
# about the encoding while table and summary still describe the original. The
# plotly analogue of transforms.MPL_TYPE_SWAP / SNS_TYPE_SWAP, and deliberately
# as narrow: treemap, sunburst, imshow, Heatmap, Sankey, Indicator and friends
# are absent because a hierarchical or matrix encoding has no honest same-data
# alternative, and proposing one would just buy a decline from the model. That
# exclusion is what the 40 rejected plotly scripts are; they are all treemaps,
# sunbursts and heatmaps.
_PX_TYPE_SWAP = {
    "scatter": "line", "line": "scatter", "bar": "line", "area": "line",
    "histogram": "box", "box": "violin", "violin": "box", "strip": "box",
    "density_contour": "scatter", "density_heatmap": "scatter",
    "pie": "bar", "funnel": "bar", "funnel_area": "bar", "ecdf": "line",
    "line_polar": "line", "scatter_polar": "scatter", "bar_polar": "bar",
}
_GO_TYPE_SWAP = {
    "Scatter": "Bar", "Scattergl": "Bar", "Bar": "Scatter", "Histogram": "Bar",
    "Box": "Violin", "Violin": "Box", "Pie": "Bar", "Funnel": "Bar",
    "Waterfall": "Bar", "Scatterpolar": "Barpolar", "Barpolar": "Scatterpolar",
}


# ------------------------------------------------------------------- public API

def applies(category: str, code: str, meta: dict, summary: str, csv_text: str,
            *, donor_titles=None) -> bool:
    """True if this (sample, category) pair is worth one API call."""
    return bool(reasons(category, code, meta, summary, csv_text,
                        donor_titles=donor_titles)["applies"])


def reasons(category: str, code: str, meta: dict, summary: str, csv_text: str,
            *, donor_titles=None) -> dict:
    """The same verdict as `applies`, plus why, for reporting and debugging.

    Never raises. A sample we cannot even parse is simply not a candidate, and
    saying so beats killing a 1000-sample sweep.
    """
    out = {"category": category, "applies": False, "mode": "loose",
           "why": [], "why_not": [], "signals": {}}
    try:
        if category not in CATEGORIES:
            out["why_not"].append(f"unknown category {category!r}")
            return out
        if category in _DELEGATED:
            out["mode"] = "delegated"
            if _delegate(category, code, meta, summary, csv_text, donor_titles):
                out["applies"] = True
                out["why"].append("deterministic transform found a target")
                return out
            out["why_not"].append("deterministic transform found no target")
            check = _PLOTLY_FALLBACK.get(category)
            if check is None:
                return out
            pf = _plotly_facts(code)
            if not pf["is_plotly"]:
                return out  # matplotlib/seaborn keep the delegated verdict verbatim
            out["mode"] = "plotly"
            check(
                pf,
                _facts(
                    code,
                    summary,
                    csv_text,
                    str(meta.get("chart_type", "")),
                    str(meta.get("library", "")),
                ),
                out,
            )
            return out
        facts = _facts(
            code,
            summary,
            csv_text,
            str(meta.get("chart_type", "")),
            str(meta.get("library", "")),
        )
        if facts["error"]:
            out["why_not"].append(facts["error"])
            return out
        {"axis-truncation": _check_axis_truncation,
         "dropped-series": _check_dropped_series,
         "swapped-legend-labels": _check_swapped_legend_labels}[category](facts, out)
    except Exception as exc:  # noqa: BLE001 - a precondition must never kill the sweep
        out["applies"] = False
        out["why_not"].append(f"precondition_error: {exc!r}")
    return out


def _delegate(category, code, meta, summary, csv_text, donor_titles):
    """The unchanged deterministic answer, for the categories we do not widen."""
    try:
        ctx = transforms.build_ctx(code, meta, summary, csv_text)
    except (SyntaxError, ValueError):
        return False
    rng = random.Random(f"{meta.get('id', '?')}:{category}")
    try:
        return _DELEGATED[category](ctx, rng, donor_titles or _PLACEHOLDER_DONORS) is not None
    except Exception:  # noqa: BLE001 - transforms are an external prefilter boundary
        return False


# ---------------------------------------------------------------- category checks

def _check_axis_truncation(facts, out):
    sig = out["signals"]
    sig["chart_type"] = facts["chart_type"]
    sig["library"] = facts["library"]
    sig["detected_backend"] = facts["detected_backend"]
    sig["library_conflict"] = facts["library_conflict"]
    sig["multi_axes"] = facts["multi_axes"]
    sig["value_axis"] = facts["value_axis"]
    sig["zero_baseline_marks"] = sorted(facts["zero_baseline_marks"])
    sig["range_set"] = facts["range_set"]
    sig["value_span"] = facts["value_span"]

    if facts["library_conflict"]:
        out["why_not"].append(facts["library_conflict"])
        return
    # Plotly's JSON contains a range only when the source pins it explicitly.
    # The free auto-range required by this precondition is therefore absent
    # from the GT capture, so the binding planner cannot prove clipping yet.
    if (
        _library_family(facts["library"]) == "plotly"
        or facts["detected_backend"] == "plotly"
    ):
        out["why_not"].append(
            "Plotly GT capture does not expose computed auto-ranges"
        )
        return
    if facts["multi_axes"]:
        out["why_not"].append(
            f"V2 axis injection requires one axis; {facts['multi_axes']}"
        )
        return
    if any(w in facts["chart_type"].lower() for w in _NO_VALUE_AXIS):
        out["why_not"].append("chart family has no continuous value axis")
        return
    if not facts["value_axis"]:
        out["why_not"].append("no supported continuous value axis")
        return
    if facts["range_set"]:
        out["why_not"].append(f"script already pins the {facts['value_axis']} range "
                              f"via {facts['range_set']}")
        return
    if facts["value_span"] is None:
        out["why_not"].append("fewer than two distinct truth-backed numeric values")
        return
    lo, hi = facts["value_span"]
    out["why"].append(f"{'/'.join(sorted(facts['zero_baseline_marks']))} on a free "
                      f"{facts['value_axis']} axis, values {lo:g}..{hi:g}; the "
                      "capture-time planner can place one endpoint outside the range")
    out["applies"] = True


def _check_dropped_series(facts, out):
    sig = out["signals"]
    sig["series_evidence"] = list(facts["series_evidence"])
    sig["named_series"] = facts["named_series"][:6]

    if not facts["series_evidence"]:
        out["why_not"].append("no evidence of a second series")
        return
    if not facts["named_series"]:
        out["why_not"].append("no series name occurs in summary.txt or table.csv, so a "
                              "dropped series would be undetectable")
        return
    out["why"].append(f"{len(facts['series_evidence'])} multi-series signal(s), "
                      f"{len(facts['named_series'])} named series")
    out["applies"] = True


def _check_swapped_legend_labels(facts, out):
    sig = out["signals"]
    sig["series_evidence"] = list(facts["series_evidence"])
    sig["legend_evidence"] = list(facts["legend_evidence"])
    sig["named_series"] = facts["named_series"][:6]

    if not facts["legend_evidence"]:
        out["why_not"].append("chart draws no legend, so swapped names would be invisible")
        return
    if not facts["series_evidence"]:
        out["why_not"].append("no evidence of a second series")
        return
    if len(facts["named_series"]) < 2:
        out["why_not"].append("fewer than two series names occur in summary.txt or table.csv")
        return
    out["why"].append(f"legend ({', '.join(facts['legend_evidence'])}) over "
                      f"{len(facts['named_series'])} named series")
    out["applies"] = True


# ------------------------------------------------------- plotly fallback checks
# Each mirrors the bar its delegated transform sets for matplotlib, so the two
# libraries are admitted on the same evidence rather than on the same spelling.
# All four take (plotly facts, shared facts, out) so the dispatch table below
# stays uniform; the two textual-metadata checks read the script only.

def _check_plotly_title(pf, facts, out):
    """`t_wrong_title` asks for one literal title string. So does this."""
    out["signals"]["plotly_titles"] = pf["titles"][:3]
    if not pf["titles"]:
        out["why_not"].append("no literal chart title in any layout or express call")
        return
    out["why"].append(f"plotly title {pf['titles'][0]!r} via {pf['title_via'][0]}")
    out["applies"] = True


def _check_plotly_axis_label(pf, facts, out):
    """`t_wrong_axis_label` asks for an xlabel or ylabel literal. Same here, in
    whichever way the script spells it; the express `labels={}` form counts only
    when a key it renames is the column bound to x= or y=, because that same
    dict also renames legend and hover fields, and on a pie or a polar chart it
    renames nothing else."""
    out["signals"]["plotly_axis_titles"] = pf["axis_titles"][:4]
    if not pf["axis_titles"]:
        out["why_not"].append("no axis title: neither an axis layout keyword nor an "
                              "express labels={} entry for the x= or y= column")
        return
    axes = sorted({a for a, _, _ in pf["axis_titles"]})
    out["why"].append(f"{'/'.join(axes)}-axis title(s) via "
                      f"{', '.join(sorted({v for _, _, v in pf['axis_titles']}))}")
    out["applies"] = True


def _check_plotly_missing_legend(pf, facts, out):
    """`t_missing_legend` asks for a legend the script draws on purpose over >=2
    series. Plotly draws one automatically, so the >=2-named-series bar from
    swapped-legend-labels does the work: without two names the summary or the
    table actually carries, deleting the legend costs the reader nothing and the
    flaw is unprovable."""
    sig = out["signals"]
    sig["plotly_legend_evidence"] = pf["legend_evidence"]
    sig["legend_suppressed"] = pf["legend_suppressed"]
    sig["named_series"] = facts["named_series"][:6]

    if pf["legend_suppressed"]:
        out["why_not"].append("script already sets showlegend=False, so there is no "
                              "legend left to remove")
        return
    if not pf["legend_evidence"]:
        out["why_not"].append("chart draws no legend")
        return
    if not facts["series_evidence"]:
        out["why_not"].append("no evidence of a second series")
        return
    if len(facts["named_series"]) < 2:
        out["why_not"].append("fewer than two series names occur in summary.txt or table.csv")
        return
    out["why"].append(f"legend ({', '.join(pf['legend_evidence'])}) over "
                      f"{len(facts['named_series'])} named series")
    out["applies"] = True


def _check_plotly_chart_type(pf, facts, out):
    """`t_wrong_chart_type` asks for a name in its swap table. Same here."""
    out["signals"]["plotly_type_swaps"] = [f"{a} -> {b}" for a, b, _ in pf["type_swaps"][:3]]
    if not pf["type_swaps"]:
        out["why_not"].append("no plotly mark with an honest same-data swap partner")
        return
    old, new, n = pf["type_swaps"][0]
    out["why"].append(f"{old} -> {new} at {n} call site(s)")
    out["applies"] = True


_PLOTLY_FALLBACK = {
    "wrong-title": _check_plotly_title,
    "wrong-axis-label": _check_plotly_axis_label,
    "missing-legend": _check_plotly_missing_legend,
    "wrong-chart-type": _check_plotly_chart_type,
}


# ------------------------------------------------------------------ fact gathering

@lru_cache(maxsize=64)
def _facts(
    code: str,
    summary: str,
    csv_text: str,
    chart_type: str,
    library: str = "",
) -> dict:
    """Everything the loose checks need, extracted once per sample.

    Cached because build_jobs asks about several categories per sample while the
    extraction walks the whole AST and parses the CSV.
    """
    f = {"error": "", "chart_type": chart_type, "library": library,
         "detected_backend": "", "library_conflict": "", "multi_axes": "",
         "value_axis": "", "range_set": "",
         "zero_baseline_marks": set(), "value_span": None,
         "series_evidence": [], "legend_evidence": [], "named_series": []}
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError) as exc:
        f["error"] = f"unparseable script: {exc.__class__.__name__}"
        return f

    f["detected_backend"] = _ast_backend(tree)
    metadata_backend = _library_family(library)
    if f["detected_backend"] == "mixed":
        f["library_conflict"] = "AST contains multiple rendering backends"
    elif metadata_backend and f["detected_backend"] \
            and metadata_backend != f["detected_backend"]:
        f["library_conflict"] = (
            f"library metadata {library!r} contradicts AST backend "
            f"{f['detected_backend']!r}"
        )
    f["multi_axes"] = _known_multi_axes(tree, chart_type)

    columns = _csv_columns(csv_text)
    numeric = _numeric_columns(columns)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]

    marks = _zero_baseline_marks(calls, numeric)
    # A vertical mark wins: when a script draws both, y is the axis the reader
    # reads values off.
    axis = "y" if marks["y"] else ("x" if marks["x"] else "")
    f["value_axis"] = axis
    if axis:
        f["zero_baseline_marks"] = marks[axis]
        f["range_set"] = _range_pinned(calls, axis)
        f["value_span"] = _clippable_span(tree, numeric)

    names, f["series_evidence"] = _series_signals(calls, columns, numeric)
    f["legend_evidence"] = _legend_signals(code, calls)
    f["named_series"] = _named_in_evidence(names, summary, csv_text)
    return f


def _positive_int(node) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) \
            and not isinstance(node.value, bool) and node.value > 0:
        return node.value
    return None


def _library_family(library: str) -> str:
    normalized = library.strip().casefold()
    if "plotly" in normalized:
        return "plotly"
    if normalized in {"matplotlib", "mpl", "seaborn"}:
        return "matplotlib"
    return ""


def _ast_backend(tree: ast.AST) -> str:
    """Infer only backend evidence that is explicit in imports or output calls."""
    writers = set()
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            component = _comp(node)
            if component == "write_image":
                writers.add("plotly")
            elif component == "savefig":
                writers.add("matplotlib")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                family = _library_family(alias.name.split(".", 1)[0])
                if family:
                    imported.add(family)
        elif isinstance(node, ast.ImportFrom):
            family = _library_family((node.module or "").split(".", 1)[0])
            if family:
                imported.add(family)
    evidence = writers or imported
    if len(evidence) > 1:
        return "mixed"
    return next(iter(evidence), "")


_ADDITIONAL_AXIS_CALLS = {
    "inset_axes",
    "secondary_xaxis",
    "secondary_yaxis",
    "twinx",
    "twiny",
}


def _known_multi_axes(tree: ast.AST, chart_type: str) -> str:
    """Return only static evidence that guarantees a multi-axis figure."""
    normalized = " ".join(chart_type.casefold().replace("-", " ").split())
    if "multi axes" in normalized or "multiple axes" in normalized:
        return f"chart_type={chart_type!r}"

    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        component = _comp(call)
        if component in _ADDITIONAL_AXIS_CALLS:
            return f"{component}() creates an additional axis"
        if component not in {"subplots", "make_subplots"}:
            continue
        row_names = {"nrows", "rows"}
        column_names = {"ncols", "cols"}
        rows = next(
            (_positive_int(kw.value) for kw in call.keywords if kw.arg in row_names),
            None,
        )
        columns = next(
            (_positive_int(kw.value) for kw in call.keywords if kw.arg in column_names),
            None,
        )
        if rows is None and call.args:
            rows = _positive_int(call.args[0])
        if columns is None and len(call.args) > 1:
            columns = _positive_int(call.args[1])
        # The unspecified dimension defaults to one for both APIs. A known
        # dimension greater than one therefore suffices; variable dimensions
        # remain permissive and are left to the real capture-time planner.
        if (rows is not None and rows > 1) or (columns is not None and columns > 1):
            return f"{component}() declares a {rows or 1}x{columns or 1} grid"
    return ""


def _comp(call) -> str:
    return transforms.call_name(call).split(".")[-1].lower()


def _str_kw(call, name):
    for kw in call.keywords:
        if kw.arg == name and isinstance(kw.value, ast.Constant) \
                and isinstance(kw.value.value, str):
            return kw.value.value
    return None


def _list_kw(call, name):
    for kw in call.keywords:
        if kw.arg == name and isinstance(kw.value, (ast.List, ast.Tuple)):
            return kw.value.elts
    return None


def _has_kw(call, name) -> bool:
    return any(kw.arg == name for kw in call.keywords)


def _strings(nodes):
    return [n.value for n in nodes
            if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def _is_horizontal(call, numeric_cols) -> bool:
    """Bars drawn sideways. Their value axis is x, which is exactly why the
    ChartNet label is useless here: several of these are filed as "Bar Chart"."""
    o = _str_kw(call, "orientation") or _str_kw(call, "orient")
    if o and o.lower() in {"h", "horizontal"}:
        return True
    x, y = _str_kw(call, "x"), _str_kw(call, "y")
    return bool(x and y and x in numeric_cols and y not in numeric_cols)


def _zero_baseline_marks(calls, numeric_cols) -> dict:
    """Return capturable marks grouped by axes whose endpoints can be clipped."""
    marks = {"y": set(), "x": set()}
    for c in calls:
        comp, kind = _comp(c), (_str_kw(c, "kind") or "").lower()
        sideways = False
        if comp == "barh" or kind in _HBAR_KINDS:
            mark, sideways = "barh", True
        elif comp in _VBAR_COMPS or kind in _VBAR_KINDS:
            mark = comp if comp in _VBAR_COMPS else "bar"
        elif comp in _AREA_COMPS or kind == "area":
            mark = comp if comp in _AREA_COMPS else "area"
        elif comp in _POINT_COMPS or kind in {"line", "scatter"}:
            mark = comp if comp in _POINT_COMPS else kind
            marks["x"].add(mark)
            marks["y"].add(mark)
            continue
        else:
            continue  # includes _HIST_COMPS, see there
        marks["x" if sideways or _is_horizontal(c, numeric_cols) else "y"].add(mark)
    return marks


def _dict_has_key(node, key) -> bool:
    if isinstance(node, ast.Dict):
        return any(isinstance(k, ast.Constant) and k.value == key for k in node.keys)
    if isinstance(node, ast.Call) and _comp(node) == "dict":
        return any(kw.arg == key for kw in node.keywords)
    return False


def _range_pinned(calls, axis: str) -> str:
    """What already fixes this axis, or "" if nothing does."""
    for c in calls:
        comp = _comp(c)
        if comp in _LIMIT_CALLS[axis]:
            return f"{comp}()"
        for name in _LIMIT_KWARGS[axis]:
            if _has_kw(c, name):
                return f"{name}="
        if comp in _LIMIT_UPDATERS[axis] and _has_kw(c, "range"):
            return f"{comp}(range=)"
        for kw in c.keywords:
            if kw.arg in _LIMIT_DICTS[axis] and _dict_has_key(kw.value, "range"):
                return f"{kw.arg}=dict(range=)"
        if comp == "axis" and c.args and isinstance(c.args[0], (ast.List, ast.Tuple)) \
                and len(c.args[0].elts) == 4:
            return "axis([xmin, xmax, ymin, ymax])"
    return ""


def _year_like(nums) -> bool:
    """A run of calendar years is an axis, not a measured quantity. Floats count:
    table.csv hands us 2020.0, and a year column would sail through every spread
    test with a min/max of 0.99."""
    return all(float(v).is_integer() and 1900 <= v <= 2100 for v in nums)


def _value_lists(tree):
    """Inline numeric series, read the way transforms._numeric_lists reads them.
    Three elements is enough here: a four-bar chart is truncatable, it just is
    not spliceable by a deterministic transform."""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.List, ast.Tuple)) or len(node.elts) < 3:
            continue
        nums = [transforms.num_value(el) for el in node.elts]
        nums = [v for v in nums if v is not None]
        if len(nums) < max(3, int(0.9 * len(node.elts))) or _year_like(nums):
            continue
        out.append(nums)
    return out


def _clippable_span(tree, numeric_cols):
    """A distinct numeric span from which the runtime planner can clip a value.

    This is only a cheap API-call prefilter. The binding V2 postcondition reads
    the captured figure, proves that a real endpoint is outside the flawed
    limits, and proves that the complete series data stayed unchanged.
    """
    for nums in list(_value_lists(tree)) + list(numeric_cols.values()):
        if len(nums) < 3 or _year_like(nums):
            continue
        lo, hi = min(nums), max(nums)
        if hi > lo:
            return (lo, hi)
    return None


def _series_signals(calls, columns, numeric_cols):
    """(candidate series names, signal names). Any one signal is enough: the
    question is only whether a second series plausibly exists."""
    names, signals = [], []

    labels = sorted({v for c in calls for v in [_str_kw(c, "label")] if v})
    if len(labels) >= 2:
        signals.append("labelled_plot_calls")
    names += labels

    for c in calls:
        if _comp(c) != "legend":
            continue
        entries = _list_kw(c, "labels")
        if entries is None and c.args and isinstance(c.args[0], (ast.List, ast.Tuple)):
            entries = c.args[0].elts
        strings = _strings(entries or [])
        if len(strings) >= 2:
            signals.append("legend_label_list")
            names += strings

    group_cols = [col for c in calls for kw in _GROUP_KWARGS
                  for col in [_str_kw(c, kw)] if col and col in columns]
    if group_cols:
        signals.append("grouping_column")
        for col in group_cols:
            uniq = sorted({v for v in columns[col] if v})
            if 2 <= len(uniq) <= 40:
                names += uniq
    elif any(_str_kw(c, kw) for c in calls for kw in ("hue", "style", "col", "row")):
        # A grouping column we could not resolve against the CSV still splits
        # that one call into several series.
        signals.append("grouping_column")

    if sum(1 for c in calls if _comp(c) in {"add_trace", "add_scatter", "add_bar"}) >= 2:
        signals.append("multiple_add_trace")
    trace_names = sorted({v for c in calls for v in [_str_kw(c, "name")] if v})
    if len(trace_names) >= 2:
        signals.append("named_traces")
    names += trace_names

    if any(_comp(c) in {"melt", "pivot", "pivot_table"} for c in calls):
        signals.append("reshaped_frame")

    wide = [h for h in numeric_cols if not _year_like(numeric_cols[h])]
    if len(wide) >= 2:
        signals.append("multiple_numeric_csv_columns")
        names += wide

    for c in calls:
        ys = _list_kw(c, "y") if _comp(c) == "plot" else None
        if ys and len(ys) >= 2:
            signals.append("dataframe_multi_y")
            names += _strings(ys)

    if not signals and any(_comp(c) == "legend" for c in calls):
        signals.append("legend_call")

    seen, uniq_names = set(), []
    for n in names:
        if n and n not in seen:
            seen.add(n)
            uniq_names.append(n)
    return uniq_names, sorted(set(signals))


def _legend_signals(code, calls):
    """Does the rendered chart carry a legend? Swapping two names is only a flaw
    if the reader can see the names."""
    ev = []
    if any(_comp(c) == "legend" for c in calls):
        ev.append("legend_call")
    if any(_has_kw(c, "hue") for c in calls):
        ev.append("seaborn_hue")
    if any(transforms.call_name(c).startswith("px.")
           and (_str_kw(c, "color") or _str_kw(c, "symbol") or _str_kw(c, "line_dash"))
           for c in calls):
        ev.append("plotly_express_color")
    if sum(1 for c in calls if _str_kw(c, "name")) >= 2:
        ev.append("named_traces")
    if "showlegend=True" in code or "legend_title" in code:
        ev.append("legend_layout")
    # An explicit switch-off beats every automatic legend above; only a legend
    # the script asks for by name survives it.
    if "legend=False" in code or "showlegend=False" in code:
        ev = [e for e in ev if e == "legend_call"]
    return sorted(set(ev))


def _named_in_evidence(names, summary, csv_text):
    """Series names the agent can actually read somewhere truthful. Without one
    the flaw is unprovable and the instance is dead weight."""
    hay = (summary + "\n" + csv_text).lower()
    return [n for n in names if len(n) >= 2 and n.lower() in hay]


def _const_str(node):
    return node.value if isinstance(node, ast.Constant) \
        and isinstance(node.value, str) else None


def _kw_node(call, name):
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _dict_value(node, key):
    """Value under `key` of a `{...}` literal or a `dict(...)` call, else None."""
    if isinstance(node, ast.Dict):
        for k, v in zip(node.keys, node.values):
            if isinstance(k, ast.Constant) and k.value == key:
                return v
    elif isinstance(node, ast.Call) and _comp(node) == "dict":
        for kw in node.keywords:
            if kw.arg == key:
                return kw.value
    return None


def _text_of(node):
    """The string a plotly text slot renders. Every one of them takes either the
    bare string or a styling dict whose "text" key carries it, and a dict with
    no "text" (14 scripts pass `title=dict(x=0.5)` to centre a title set by the
    express call) renders no text of its own."""
    s = _const_str(node)
    return s if s is not None else _const_str(_dict_value(node, "text"))


def _axis_of(kwarg) -> str:
    """"x"/"y" for xaxis, yaxis and the xaxis2... a subplot figure numbers."""
    if kwarg and kwarg[:5] in ("xaxis", "yaxis") \
            and (len(kwarg) == 5 or kwarg[5:].isdigit()):
        return kwarg[0]
    return ""


def _plotly_aliases(tree):
    """{module: local name} for plotly.express / plotly.graph_objects. Read from
    the imports rather than hardcoding px/go: the alias is what the call sites
    actually spell, and a `call_name` prefix means nothing without it."""
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name in ("plotly.express", "plotly.graph_objects"):
                    out[a.name] = a.asname or a.name
        elif isinstance(n, ast.ImportFrom) and n.module == "plotly":
            for a in n.names:
                if a.name in ("express", "graph_objects"):
                    out[f"plotly.{a.name}"] = a.asname or a.name
    return out


@lru_cache(maxsize=64)
def _plotly_facts(code: str) -> dict:
    """The four features the plotly fallback needs, extracted once per sample.

    Separate from `_facts` because it is keyed on the script alone: nothing here
    consults the summary or the table, and it must stay cheap enough to run for
    every delegated miss.
    """
    pf = {"is_plotly": False, "titles": [], "title_via": [], "axis_titles": [],
          "legend_evidence": [], "legend_suppressed": False, "type_swaps": []}
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return pf
    alias = _plotly_aliases(tree)
    pf["is_plotly"] = any(
        (isinstance(n, ast.Import) and any(a.name.split(".")[0] == "plotly" for a in n.names))
        or (isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[0] == "plotly")
        for n in ast.walk(tree))
    if not pf["is_plotly"]:
        return pf

    px_name = alias.get("plotly.express")
    go_name = alias.get("plotly.graph_objects")
    legend, swaps, named_traces = [], {}, 0

    for c in [n for n in ast.walk(tree) if isinstance(n, ast.Call)]:
        parts = transforms.call_name(c).split(".")
        comp, root = parts[-1], (parts[0] if len(parts) > 1 else "")
        express = bool(px_name) and root == px_name

        if comp == "update_layout" or express:
            for name in _PLOTLY_TITLE_KWARGS:
                text = _text_of(_kw_node(c, name))
                if text is not None:
                    pf["titles"].append(text)
                    pf["title_via"].append(f"{'px' if express else 'update_layout'}({name}=)")

        axis = _PLOTLY_AXIS_UPDATERS.get(comp, "")
        if axis:
            for name in ("title_text", "title"):
                text = _text_of(_kw_node(c, name))
                if text is not None:
                    pf["axis_titles"].append((axis, text, f"{comp}({name}=)"))
        if comp == "update_layout":
            for ax, names in _PLOTLY_AXIS_KWARGS.items():
                for name in names:
                    text = _text_of(_kw_node(c, name))
                    if text is not None:
                        pf["axis_titles"].append((ax, text, f"{name}="))
            for kw in c.keywords:
                ax = _axis_of(kw.arg)
                text = _text_of(_dict_value(kw.value, "title")) if ax else None
                if text is not None:
                    pf["axis_titles"].append((ax, text, f"{kw.arg}=dict(title=)"))
        if express and comp not in _PX_NO_XY:
            labels = _kw_node(c, "labels")
            bound = {ax: _const_str(_kw_node(c, ax)) for ax in ("x", "y")}
            if isinstance(labels, ast.Dict):
                for k, v in zip(labels.keys, labels.values):
                    key, text = _const_str(k), _const_str(v)
                    ax = next((a for a, col in bound.items() if col and col == key), "")
                    # A labels= entry for a column that is NOT on x or y renames a
                    # legend or a hover field, so it is not an axis title.
                    if ax and text is not None:
                        pf["axis_titles"].append((ax, text, "px(labels=)"))

        show = _kw_node(c, "showlegend")
        if comp == "update_layout":
            if isinstance(show, ast.Constant):
                # Layout level switches the whole legend off; a trace-level False
                # only hides that one trace.
                if show.value:
                    legend.append("layout_showlegend")
                else:
                    pf["legend_suppressed"] = True
            for name in _PLOTLY_LEGEND_KWARGS:
                if _kw_node(c, name) is not None:
                    legend.append("legend_layout")
        elif isinstance(show, ast.Constant) and show.value:
            legend.append("trace_showlegend")
        if _const_str(_kw_node(c, "name")) is not None:
            named_traces += 1
        if express and any(_const_str(_kw_node(c, ch)) for ch in _PX_LEGEND_CHANNELS):
            legend.append("express_legend_channel")

        swap = _PX_TYPE_SWAP.get(comp) if express else (
            _GO_TYPE_SWAP.get(comp) if go_name and root == go_name else None)
        if swap:
            key = (f"{root}.{comp}", f"{root}.{swap}")
            swaps[key] = swaps.get(key, 0) + 1

    if named_traces >= 2:
        legend.append("named_traces")
    pf["legend_evidence"] = sorted(set(legend))
    pf["type_swaps"] = [(old, new, n) for (old, new), n in
                        sorted(swaps.items(), key=lambda kv: (-kv[1], kv[0]))]
    return pf


def _csv_columns(csv_text: str) -> dict:
    if not csv_text.strip():
        return {}
    rows = list(csv.reader(io.StringIO(csv_text)))
    if not rows:
        return {}
    header = [h.strip() for h in rows[0]]
    cols = {h: [] for h in header if h}
    for row in rows[1:_MAX_CSV_ROWS + 1]:
        for h, cell in zip(header, row):
            if h:
                cols[h].append(cell.strip())
    return cols


def _numeric_columns(columns: dict) -> dict:
    """Columns that are at least 80% parseable numbers, as floats."""
    out = {}
    for h, cells in columns.items():
        vals, filled = [], 0
        for cell in cells:
            if not cell:
                continue
            filled += 1
            try:
                vals.append(float(cell.replace(",", "").replace("%", "").replace("$", "")))
            except ValueError:
                pass
        if filled >= 3 and len(vals) >= 0.8 * filled:
            out[h] = vals
    return out
