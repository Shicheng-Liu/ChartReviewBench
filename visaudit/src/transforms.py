"""Deterministic code perturbations for ChartRepairBench v0.

Design: the AST locates the exact node to change (no false positives from
comments or repeated literals), then a minimal splice is applied to the ORIGINAL
source string — so gt_code -> flawed_code is a tiny, human-readable diff that
doubles as the ground-truth edit for evaluation.

v0 policy: exactly ONE injected flaw per instance; perturbation metadata stores
a list of edits so multi-flaw stacking stays cheap later. Only the rendered
chart lies — table.csv and summary.txt are never touched; they are the agent's
evidence.
"""
import ast
import difflib
import json
import random
import re
from pathlib import Path

LLM_LABELS_PATH = Path(__file__).resolve().parent.parent / "data/v0/llm_axis_labels.json"
LLM_TITLES_PATH = Path(__file__).resolve().parent.parent / "data/v0/llm_titles.json"
_LLM_LABELS = None
_LLM_TITLES = None


def _llm_labels():
    """Cached plausible-but-wrong axis labels proposed by Claude (see llm_labels.py).
    Missing/unreadable cache degrades gracefully to the rule-based fallback."""
    global _LLM_LABELS
    if _LLM_LABELS is None:
        try:
            _LLM_LABELS = json.loads(LLM_LABELS_PATH.read_text())
        except (OSError, ValueError):
            _LLM_LABELS = {}
    return _LLM_LABELS


def _llm_titles():
    """Cached plausible-but-wrong titles proposed by Claude (see llm_titles.py).
    Missing/unreadable cache degrades gracefully to the donor-title fallback."""
    global _LLM_TITLES
    if _LLM_TITLES is None:
        try:
            _LLM_TITLES = json.loads(LLM_TITLES_PATH.read_text())
        except (OSError, ValueError):
            _LLM_TITLES = {}
    return _LLM_TITLES

# ---------------------------------------------------------------- span helpers

def line_starts(src):
    starts = [0]
    for line in src.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    return starts


def span(node, starts):
    return (starts[node.lineno - 1] + node.col_offset,
            starts[node.end_lineno - 1] + node.end_col_offset)


def stmt_line_span(node, starts):
    """Whole-line span of a statement, including the trailing newline."""
    s = starts[node.lineno - 1]
    e = starts[node.end_lineno] if node.end_lineno < len(starts) else starts[-1]
    return s, e


def call_name(node):
    parts, f = [], node.func
    while isinstance(f, ast.Attribute):
        parts.append(f.attr)
        f = f.value
    if isinstance(f, ast.Name):
        parts.append(f.id)
    return ".".join(reversed(parts))


def is_plain_str(node, src, starts):
    """True if this Constant str's source span starts with a quote (i.e. it is
    a standalone literal we can safely replace — excludes f-string innards)."""
    if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
        return False
    s, _ = span(node, starts)
    ch = src[s]
    if ch in "'\"":
        return True
    return ch in "rRuU" and s + 1 < len(src) and src[s + 1] in "'\""


def num_value(node):
    """Numeric value of Constant or -Constant, else None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = num_value(node.operand)
        return None if inner is None else -inner
    return None


def apply_edits(src, edits):
    for s, e, new in sorted(edits, key=lambda x: x[0], reverse=True):
        src = src[:s] + new + src[e:]
    return src


def _kwarg_deletion_span(src, s, e):
    """Extend a keyword node's span over its separating comma."""
    j = e
    while j < len(src) and src[j] in " \t\n":
        j += 1
    if j < len(src) and src[j] == ",":
        j += 1
        while j < len(src) and src[j] in " \t":
            j += 1
        return s, j
    i = s - 1
    while i >= 0 and src[i] in " \t\n":
        i -= 1
    if i >= 0 and src[i] == ",":
        return i, e
    return s, e


def strip_kwarg(src, name):
    """Remove every `name=...` keyword argument from every call. Returns new
    source, or None if nothing was removed / result no longer parses."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    starts = line_starts(src)
    edits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == name:
                    s = starts[kw.lineno - 1] + kw.col_offset
                    e = starts[kw.end_lineno - 1] + kw.end_col_offset
                    edits.append(_kwarg_deletion_span(src, s, e) + ("",))
    if not edits:
        return None
    new_src = apply_edits(src, edits)
    try:
        ast.parse(new_src)
    except SyntaxError:
        return None
    return new_src


def unified(gt, new):
    return "".join(difflib.unified_diff(
        gt.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile="gt/code.py", tofile="flawed/code.py"))

# ------------------------------------------------------------------- vocab

COLOR_WORDS = {
    "blue", "red", "green", "orange", "purple", "brown", "pink", "gray", "grey",
    "yellow", "black", "cyan", "magenta", "teal", "gold", "crimson", "navy",
    "salmon", "olive", "maroon", "violet", "turquoise", "coral", "indigo",
}
NAMED_CONTRAST = {
    "blue": "red", "red": "blue", "green": "orange", "orange": "green",
    "purple": "yellow", "yellow": "purple", "brown": "teal", "teal": "brown",
    "pink": "navy", "navy": "pink", "gray": "crimson", "grey": "crimson",
    "black": "gold", "gold": "black", "cyan": "maroon", "magenta": "olive",
}
HEX_CYCLE = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]

TITLE_COMPS = {"title", "suptitle", "set_title"}
XLABEL_COMPS = {"xlabel", "set_xlabel"}
YLABEL_COMPS = {"ylabel", "set_ylabel"}
SNS_LEGEND_OK = {"lineplot", "scatterplot", "barplot", "boxplot", "violinplot",
                 "stripplot", "swarmplot", "kdeplot", "histplot", "pointplot",
                 "countplot"}
MPL_TYPE_SWAP = {"bar": "plot", "barh": "plot", "scatter": "plot", "plot": "scatter"}
SNS_TYPE_SWAP = {"barplot": "lineplot", "lineplot": "scatterplot",
                 "scatterplot": "lineplot", "boxplot": "violinplot",
                 "violinplot": "boxplot", "stripplot": "swarmplot"}

# --------------------------------------------------------------- ctx builders

def build_ctx(src, meta, summary, csv_text):
    tree = ast.parse(src)
    starts = line_starts(src)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    headers = []
    first = csv_text.strip().splitlines()[0] if csv_text.strip() else ""
    headers = [h.strip() for h in first.split(",") if h.strip()]
    return {
        "src": src, "tree": tree, "starts": starts, "calls": calls,
        "meta": meta, "summary": summary.lower(), "csv_headers": headers,
    }


def _title_candidates(ctx):
    out = []
    for c in ctx["calls"]:
        comp = call_name(c).split(".")[-1]
        if comp in TITLE_COMPS and c.args and is_plain_str(c.args[0], ctx["src"], ctx["starts"]):
            out.append(c.args[0])
        if comp in {"set", "update"}:
            for kw in c.keywords:
                if kw.arg == "title" and is_plain_str(kw.value, ctx["src"], ctx["starts"]):
                    out.append(kw.value)
    return out


def extract_title(src):
    try:
        ctx = build_ctx(src, {}, "", "")
    except SyntaxError:
        return None
    cands = _title_candidates(ctx)
    return cands[0].value if cands else None

# ----------------------------------------------------------------- transforms

def t_wrong_title(ctx, rng, donor_titles):
    cands = _title_candidates(ctx)
    if not cands:
        return None
    node = cands[0]
    starts = ctx["starts"]

    # Preferred mechanism: LLM-proposed plausible-but-wrong title (content from
    # the model, injection deterministic) — same hybrid as t_wrong_axis_label.
    llm = _llm_titles().get(str(ctx["meta"].get("id", "")))
    if llm and llm.get("new") and llm.get("old") == node.value:
        s, e = span(node, starts)
        return {
            "category": "wrong-title", "group": "textual-metadata",
            "detectable_from": ["summary"], "difficulty": "easy",
            "precondition": "a title call with a plain string literal exists; LLM proposed a plausible-but-wrong title",
            "evidence": {"summary": [node.value]},
            "generator": {"type": "llm", "model": llm.get("model", "?"), "cached": True},
            "edits": [(s, e, repr(llm["new"]))],
            "description": f"title {node.value!r} -> {llm['new']!r} (LLM-generated plausible-but-wrong title)",
        }

    donors = [t for t in donor_titles if t and t != node.value]
    if not donors:
        return None
    new = rng.choice(donors)
    s, e = span(node, starts)
    return {
        "category": "wrong-title", "group": "textual-metadata",
        "detectable_from": ["summary"], "difficulty": "easy",
        "precondition": "a title call with a plain string literal exists",
        "evidence": {"summary": [node.value]},
        "edits": [(s, e, repr(new))],
        "description": f"title {node.value!r} -> {new!r} (donor title from another sample)",
    }


def _label_candidates(ctx, comps, kwarg_name):
    out = []
    for c in ctx["calls"]:
        comp = call_name(c).split(".")[-1]
        if comp in comps and c.args and is_plain_str(c.args[0], ctx["src"], ctx["starts"]):
            out.append(c.args[0])
        if comp in {"set", "update"}:
            for kw in c.keywords:
                if kw.arg == kwarg_name and is_plain_str(kw.value, ctx["src"], ctx["starts"]):
                    out.append(kw.value)
    return out


def t_wrong_axis_label(ctx, rng, donor_titles):
    xs = _label_candidates(ctx, XLABEL_COMPS, "xlabel")
    ys = _label_candidates(ctx, YLABEL_COMPS, "ylabel")
    starts = ctx["starts"]

    # Preferred mechanism: LLM-proposed plausible-but-wrong label (content from
    # the model, injection deterministic). Falls back to the rule-based swap
    # below when no cached proposal exists or the label no longer matches.
    llm = _llm_labels().get(str(ctx["meta"].get("id", "")))
    if llm and llm.get("new"):
        pool = ys if llm.get("axis") == "y" else xs
        node = next((n for n in pool if n.value == llm.get("old")), None)
        if node is not None:
            s, e = span(node, starts)
            return {
                "category": "wrong-axis-label", "group": "textual-metadata",
                "detectable_from": ["table", "summary"], "difficulty": "medium",
                "precondition": "axis-label literal exists; LLM proposed a plausible-but-wrong label",
                "evidence": {"either": [llm["old"]]},
                "generator": {"type": "llm", "model": llm.get("model", "?"), "cached": True},
                "edits": [(s, e, repr(llm["new"]))],
                "description": (f"{llm['axis']}-label {llm['old']!r} -> {llm['new']!r} "
                                "(LLM-generated plausible-but-wrong label, deterministic AST injection)"),
            }

    if xs and ys and xs[0].value != ys[0].value:
        (xs_s, xs_e), (ys_s, ys_e) = span(xs[0], starts), span(ys[0], starts)
        return {
            "category": "wrong-axis-label", "group": "textual-metadata",
            "detectable_from": ["table", "summary"], "difficulty": "easy",
            "precondition": "distinct xlabel and ylabel string literals exist",
            "evidence": {"either": [xs[0].value, ys[0].value]},
            "edits": [(xs_s, xs_e, repr(ys[0].value)), (ys_s, ys_e, repr(xs[0].value))],
            "description": f"swapped axis labels: x={xs[0].value!r} <-> y={ys[0].value!r}",
        }
    pool = ys or xs
    if pool:
        node = pool[0]
        donors = [h for h in ctx["csv_headers"] if h and h.lower() != node.value.lower()]
        if donors:
            new = rng.choice(donors)
            s, e = span(node, starts)
            axis = "y" if pool is ys else "x"
            return {
                "category": "wrong-axis-label", "group": "textual-metadata",
                "detectable_from": ["table", "summary"], "difficulty": "easy",
                "precondition": "an axis-label literal and a differing CSV header exist",
                "evidence": {"either": [node.value]},
                "edits": [(s, e, repr(new))],
                "description": f"{axis}-label {node.value!r} -> {new!r} (wrong CSV column)",
            }
    return None


def t_swapped_legend_labels(ctx, rng, donor_titles):
    labels = []
    for c in ctx["calls"]:
        for kw in c.keywords:
            if kw.arg == "label" and is_plain_str(kw.value, ctx["src"], ctx["starts"]):
                labels.append(kw.value)
    distinct = {}
    for n in labels:
        distinct.setdefault(n.value, n)
    if len(distinct) >= 2:
        a, b = rng.sample(sorted(distinct), 2)
        na, nb = distinct[a], distinct[b]
        (sa, ea), (sb, eb) = span(na, ctx["starts"]), span(nb, ctx["starts"])
        return {
            "category": "swapped-legend-labels", "group": "textual-metadata",
            "detectable_from": ["table", "summary"], "difficulty": "hard",
            "precondition": ">=2 distinct label= string kwargs",
            "evidence": {"either": [a, b]},
            "edits": [(sa, ea, repr(b)), (sb, eb, repr(a))],
            "description": f"swapped series labels {a!r} <-> {b!r} (geometry unchanged, naming lies)",
        }
    for c in ctx["calls"]:
        if call_name(c).split(".")[-1] == "legend" and c.args:
            arg = c.args[0]
            if isinstance(arg, (ast.List, ast.Tuple)):
                elems = [el for el in arg.elts if is_plain_str(el, ctx["src"], ctx["starts"])]
                if len(elems) >= 2:
                    i, j = rng.sample(range(len(elems)), 2)
                    ni, nj = elems[i], elems[j]
                    (si, ei), (sj, ej) = span(ni, ctx["starts"]), span(nj, ctx["starts"])
                    return {
                        "category": "swapped-legend-labels", "group": "textual-metadata",
                        "detectable_from": ["table", "summary"], "difficulty": "hard",
                        "precondition": "legend([...]) with >=2 string entries",
                        "evidence": {"either": [ni.value, nj.value]},
                        "edits": [(si, ei, repr(nj.value)), (sj, ej, repr(ni.value))],
                        "description": f"swapped legend entries {ni.value!r} <-> {nj.value!r}",
                    }
    return None


def t_missing_legend(ctx, rng, donor_titles):
    src, starts = ctx["src"], ctx["starts"]
    label_vals = sorted({kw.value.value for c in ctx["calls"] for kw in c.keywords
                         if kw.arg == "label" and is_plain_str(kw.value, src, starts)})
    if len(label_vals) >= 2:
        evidence = {"summary": label_vals[:3], "min_summary": 2}
    else:
        evidence = {"summary": ["legend"]}
    has_hue = any(
        call_name(c).startswith("sns.")
        and call_name(c).split(".")[-1] in SNS_LEGEND_OK
        and any(kw.arg == "hue" for kw in c.keywords)
        for c in ctx["calls"])

    explicit = []
    for stmt in ast.walk(ctx["tree"]):
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            if call_name(stmt.value).split(".")[-1] == "legend":
                if len(label_vals) >= 2 or has_hue or stmt.value.args:
                    s, e = stmt_line_span(stmt, starts)
                    explicit.append((s, e, ""))

    # Seaborn creates its legend inside every axes-level hue call. Deleting a
    # later plt.legend()/ax.legend() merely removes custom styling and exposes
    # that automatic legend again (the exact leak measured in 0047 and 0142).
    # Suppress every hue producer first, then remove any explicit replacement.
    hue_calls, hue_edits = [], []
    for c in ctx["calls"]:
        name = call_name(c)
        if not (name.startswith("sns.") and name.split(".")[-1] in SNS_LEGEND_OK
                and any(kw.arg == "hue" for kw in c.keywords)):
            continue
        hue_calls.append(name)
        legend_kw = next((kw for kw in c.keywords if kw.arg == "legend"), None)
        if legend_kw is None:
            _, fe = span(c.func, starts)
            insert_at = src.index("(", fe) + 1
            hue_edits.append((insert_at, insert_at, "legend=False, "))
        elif not (isinstance(legend_kw.value, ast.Constant)
                  and legend_kw.value.value is False):
            s, e = span(legend_kw.value, starts)
            hue_edits.append((s, e, "False"))

    if hue_calls and (hue_edits or explicit):
        return {
            "category": "missing-legend", "group": "textual-metadata",
            "detectable_from": ["summary"], "difficulty": "easy-medium",
            "precondition": "seaborn axes-level hue call with automatic legend",
            "evidence": evidence,
            "edits": hue_edits + explicit,
            "description": ("suppressed every seaborn hue legend at its source via "
                            "legend=False and removed explicit legend replacements"),
        }

    if explicit:
        return {
            "category": "missing-legend", "group": "textual-metadata",
            "detectable_from": ["summary"], "difficulty": "easy-medium",
            "precondition": "explicit legend() call on a multi-series chart",
            "evidence": evidence,
            "edits": explicit[:1],
            "description": "removed the legend() call — series become unidentifiable",
        }
    return None


def _color_value_nodes(ctx):
    nodes = []
    for c in ctx["calls"]:
        for kw in c.keywords:
            if kw.arg in {"color", "c", "edgecolor", "facecolor"} and \
                    is_plain_str(kw.value, ctx["src"], ctx["starts"]):
                nodes.append(kw.value)
            if kw.arg in {"palette", "colors", "color"} and isinstance(kw.value, (ast.List, ast.Tuple)):
                nodes.extend(el for el in kw.value.elts
                             if is_plain_str(el, ctx["src"], ctx["starts"]))
    return nodes


def t_color_contradicts_summary(ctx, rng, donor_titles):
    if not any(w in ctx["summary"] for w in COLOR_WORDS):
        return None
    nodes = _color_value_nodes(ctx)
    distinct = {}
    for n in nodes:
        distinct.setdefault(n.value, n)
    starts = ctx["starts"]
    base = {
        "category": "color-contradicts-summary", "group": "visual-encoding",
        "detectable_from": ["summary"], "difficulty": "medium",
        "precondition": "summary names colors AND code carries color literals",
    }
    if len(distinct) >= 2:
        a, b = rng.sample(sorted(distinct), 2)
        na, nb = distinct[a], distinct[b]
        (sa, ea), (sb, eb) = span(na, starts), span(nb, starts)
        named = [w for w in (a.lower(), b.lower()) if w in COLOR_WORDS]
        evidence = {"summary": named, "min_summary": 1} if named else {"summary_any_color": True}
        return {**base,
                "evidence": evidence,
                "edits": [(sa, ea, repr(b)), (sb, eb, repr(a))],
                "description": f"swapped colors {a!r} <-> {b!r} while the summary names them"}
    if len(distinct) == 1:
        val, node = next(iter(distinct.items()))
        low = val.lower()
        if low.startswith("#"):
            new = next(h for h in HEX_CYCLE if h.lower() != low)
        else:
            new = NAMED_CONTRAST.get(low)
        if not new or new == low:
            return None
        s, e = span(node, starts)
        evidence = {"summary": [low], "min_summary": 1} if low in COLOR_WORDS \
            else {"summary_any_color": True}
        return {**base,
                "evidence": evidence,
                "edits": [(s, e, repr(new))],
                "description": f"color {val!r} -> {new!r}, contradicting the summary"}
    return None


def _numeric_lists(ctx):
    """Candidate value lists: >=5 numeric elements, not year-like."""
    out = []
    for node in ast.walk(ctx["tree"]):
        if isinstance(node, (ast.List, ast.Tuple)) and len(node.elts) >= 5:
            vals = [num_value(el) for el in node.elts]
            nums = [v for v in vals if v is not None]
            if len(nums) < max(5, int(0.9 * len(node.elts))):
                continue
            if all(isinstance(v, int) and 1900 <= v <= 2100 for v in nums):
                continue  # year axis, not a value series
            out.append((node, vals))
    return out


def t_value_perturbation(ctx, rng, donor_titles):
    lists = _numeric_lists(ctx)
    if not lists:
        return None
    node, vals = lists[rng.randrange(len(lists))]
    idxs = [i for i, v in enumerate(vals) if v is not None and v != 0]
    if not idxs:
        return None
    i = rng.choice(idxs)
    factor = rng.choice([0.45, 1.6, 2.25])
    old = vals[i]
    new = round(old * factor) if isinstance(old, int) else round(old * factor, 3)
    if new == old:
        new = old + (1 if isinstance(old, int) else 0.5)
    s, e = span(node.elts[i], ctx["starts"])
    return {
        "category": "value-perturbation", "group": "data-level",
        "detectable_from": ["table"], "difficulty": "medium",
        "precondition": "inline numeric data series (non-year) with >=5 elements",
        "evidence": {"table_nums": [old]},
        "edits": [(s, e, repr(new))],
        "description": f"data value #{i} changed {old!r} -> {new!r} (x{factor}); table.csv still holds the truth",
    }


def t_ylim_truncation(ctx, rng, donor_titles):
    meta = ctx["meta"]
    if "Bar" not in meta.get("chart_type", ""):
        return None
    src, starts = ctx["src"], ctx["starts"]
    if "barh(" in src:
        return None
    for c in ctx["calls"]:
        if call_name(c).split(".")[-1] in {"ylim", "set_ylim"}:
            return None
    if "import matplotlib.pyplot as plt" not in src:
        return None
    nums = [v for _, vals in _numeric_lists(ctx) for v in vals if v is not None]
    if not nums:
        return None
    lo, hi = min(nums), max(nums)
    if lo <= 0 or hi <= 0 or lo / hi < 0.25:
        return None
    for stmt in ast.walk(ctx["tree"]):
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) and \
                call_name(stmt.value).split(".")[-1] == "savefig":
            line_start = starts[stmt.lineno - 1]
            indent = src[line_start:line_start + stmt.col_offset]
            if indent.strip():
                continue
            new_line = f"{indent}plt.ylim({lo * 0.88:.6g}, {hi * 1.05:.6g})\n"
            return {
                "category": "axis-truncation", "group": "structural",
                "detectable_from": ["table", "summary"], "difficulty": "hard",
                "precondition": "vertical bar chart, no existing ylim, positive values with min/max>=0.25",
                "evidence": {"table_nums": [lo, hi]},
                "edits": [(line_start, line_start, new_line)],
                "description": f"injected plt.ylim({lo * 0.88:.6g}, {hi * 1.05:.6g}) — truncated baseline exaggerates differences",
            }
    return None


def t_wrong_chart_type(ctx, rng, donor_titles):
    src, starts = ctx["src"], ctx["starts"]
    options = {}
    for c in ctx["calls"]:
        name = call_name(c)
        parts = name.split(".")
        comp = parts[-1]
        prefix = parts[0] if len(parts) > 1 else ""
        if prefix == "sns" and comp in SNS_TYPE_SWAP:
            options.setdefault(name, []).append(c)
        elif prefix in {"plt", "ax", "axs", "axes", "ax1", "ax2"} and comp in MPL_TYPE_SWAP:
            if any(kw.arg in {"x", "y"} for kw in c.keywords):
                continue  # plt.plot/scatter reject x=/y= keywords
            options.setdefault(name, []).append(c)
    if not options:
        return None
    name = sorted(options, key=lambda k: (-len(options[k]), k))[0]
    calls = options[name]
    parts = name.split(".")
    comp = parts[-1]
    new_comp = SNS_TYPE_SWAP[comp] if parts[0] == "sns" else MPL_TYPE_SWAP[comp]
    new_name = ".".join(parts[:-1] + [new_comp])
    edits = []
    for c in calls:
        s, e = span(c.func, starts)
        edits.append((s, e, new_name))
    type_word = {"bar": "bar", "barh": "bar", "plot": "line", "scatter": "scatter",
                 "barplot": "bar", "lineplot": "line", "scatterplot": "scatter",
                 "boxplot": "box", "violinplot": "violin", "stripplot": "strip"}.get(comp, comp)
    return {
        "category": "wrong-chart-type", "group": "structural",
        "detectable_from": ["summary"], "difficulty": "easy",
        "precondition": f"signature-compatible swap pair exists ({name} -> {new_name})",
        "evidence": {"summary": [type_word]},
        "edits": edits,
        "description": f"{name} -> {new_name} at {len(edits)} call site(s); summary/chart_type still say {comp}",
    }


PLOT_COMPS = {"plot", "bar", "barh", "scatter", "fill_between", "step", "stackplot",
              "lineplot", "scatterplot", "barplot", "kdeplot", "hist", "histplot"}


def t_dropped_series(ctx, rng, donor_titles):
    """Taxonomy, Structural: 'Missing data series' — remove one whole series."""
    src, starts = ctx["src"], ctx["starts"]
    cands = []
    for stmt in ast.walk(ctx["tree"]):
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            if call_name(call).split(".")[-1] not in PLOT_COMPS:
                continue
            for kw in call.keywords:
                if kw.arg == "label" and is_plain_str(kw.value, src, starts):
                    cands.append((stmt, kw.value.value))
                    break
    if len(cands) < 2:
        return None
    stmt, label = cands[rng.randrange(len(cands))]
    s, e = stmt_line_span(stmt, starts)
    return {
        "category": "dropped-series", "group": "structural",
        "detectable_from": ["table", "summary"], "difficulty": "easy-medium",
        "precondition": ">=2 labeled plot calls exist; one entire series is removed",
        "evidence": {"either": [label]},
        "edits": [(s, e, "")],
        "description": f"removed the plot call for series {label!r} — table and summary still describe it",
    }


def t_swapped_series(ctx, rng, donor_titles):
    """Taxonomy, Semantic: 'Swapped series' — exchange two series' value arrays."""
    lists = _numeric_lists(ctx)
    by_len = {}
    for node, vals in lists:
        by_len.setdefault(len(node.elts), []).append((node, vals))
    options = [g for g in by_len.values() if len(g) >= 2]
    if not options:
        return None
    group = options[rng.randrange(len(options))]
    (na, va), (nb, vb) = rng.sample(group, 2)
    if va == vb:
        return None
    starts = ctx["starts"]
    (sa, ea), (sb, eb) = span(na, starts), span(nb, starts)
    txt_a, txt_b = ctx["src"][sa:ea], ctx["src"][sb:eb]
    nums_a = [v for v in va if v is not None]
    nums_b = [v for v in vb if v is not None]
    return {
        "category": "swapped-series", "group": "data-level",
        "detectable_from": ["table", "summary"], "difficulty": "hard",
        "precondition": "two equal-length numeric series exist; their value arrays are exchanged",
        "evidence": {"table_nums": [max(nums_a), max(nums_b)]},
        "edits": [(sa, ea, txt_b), (sb, eb, txt_a)],
        "description": "swapped the value arrays of two series — labels and colors unchanged, every point lies",
    }


TRANSFORMS = [
    t_wrong_title,
    t_wrong_axis_label,
    t_swapped_legend_labels,
    t_missing_legend,
    t_color_contradicts_summary,
    t_value_perturbation,
    t_ylim_truncation,
    t_wrong_chart_type,
    t_dropped_series,
    t_swapped_series,
]


def perturb_all(src, meta, summary, csv_text, donor_titles):
    """Apply every applicable transform independently (one flaw per instance)."""
    ctx = build_ctx(src, meta, summary, csv_text)
    results = []
    for t in TRANSFORMS:
        rng = random.Random(f"{meta.get('id', '?')}:{t.__name__}")
        try:
            cand = t(ctx, rng, donor_titles)
        except Exception as exc:  # a transform must never kill the pipeline
            results.append({"category": t.__name__, "status": "transform_error",
                            "error": repr(exc)})
            continue
        if not cand:
            continue
        cand.setdefault("generator", {"type": "rule"})
        new_src = apply_edits(src, cand["edits"])
        try:
            ast.parse(new_src)
        except SyntaxError:
            cand["status"] = "syntax_error_after_edit"
            results.append(cand)
            continue
        cand["new_src"] = new_src
        cand["diff"] = unified(src, new_src)
        cand["status"] = "candidate"
        results.append(cand)
    return results
