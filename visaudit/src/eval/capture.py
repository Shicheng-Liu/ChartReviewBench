"""Run a plotting script and read out what it actually drew.

The script executes in the existing sandbox (own temp cwd, pinned libraries,
timeout); `_runner.py` and `_extract.py` travel with it and leave `figure.json`
behind, which we collect and normalise into a property map.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import harness  # noqa: E402

RUNNER = (HERE / "_runner.py").read_text(encoding="utf-8")
EXTRACT = (HERE / "_extract.py").read_text(encoding="utf-8")

SCALAR_KEYS = (
    "title", "xlabel", "ylabel", "xscale", "yscale", "suptitle",
    "title_color", "xlabel_color", "ylabel_color",
)
LIST_KEYS = (
    "xlim", "ylim", "xticks", "yticks", "xticklabels", "yticklabels",
    "annotations", "arrow_colors",
)
TICK_POLICY_KEYS = ("xtick_policy", "ytick_policy")


def capture(code: str, extra_files: dict | None = None, timeout: int = 90,
            *, isolate: bool = False) -> dict:
    """Execute code; isolate=True requires OS confinement, with no fallback."""
    files = {
        "chart_code.py": code,
        "_runner.py": RUNNER,
        "_extract.py": EXTRACT,
    }
    for name, text in (extra_files or {}).items():
        files.setdefault(name, text)

    launcher = None
    if isolate:
        from .kernel_isolation import isolated_python
        launcher = isolated_python
    res = harness.run_in_sandbox(
        files, "_runner.py", timeout=timeout,
        collect=("figure.json", "capture_error.txt"), python_launcher=launcher,
    )
    raw = res["files"].get("figure.json")
    spec = None
    if raw:
        try:
            spec = json.loads(raw)
        except json.JSONDecodeError:
            spec = None
    if spec is None and res["status"] == "ok":
        res["status"] = "no_figure"
    return {
        "status": res["status"],
        "duration": res["duration"],
        "stderr": res["stderr"],
        "capture_error": res["files"].get("capture_error.txt", ""),
        "png": res["png"],
        "spec": spec,
        "flat": flatten(spec) if spec else {},
    }


def capture_path(path, extra_files=None, timeout=90, *, isolate=False) -> dict:
    p = Path(path)
    return capture(p.read_text(encoding="utf-8"), extra_files, timeout, isolate=isolate)


# --------------------------------------------------------------------------- property map

def _escape_series_label(label: str) -> str:
    """Keep occurrence suffixes unambiguous without changing ordinary labels."""
    return "".join({"%": "%25", "#": "%23", "[": "%5B", "]": "%5D"}.get(
        char, char,
    ) for char in label)


def _series_key(axis_idx: int, series: list, i: int) -> str:
    raw_label = (series[i].get("label") or "s%d" % i).strip()
    twins = [j for j, s in enumerate(series) if (s.get("label") or "") == series[i].get("label")]
    label = _escape_series_label(raw_label)
    if len(twins) > 1:  # disambiguate duplicate labels by position
        label = "%s#%d" % (label, twins.index(i))
    return "axes[%d].series[%s]" % (axis_idx, label)


def flatten(spec: dict) -> dict:
    """Nested spec -> {property_path: value}. The substrate of all three metrics."""
    flat = {}
    if not spec:
        return flat
    if spec.get("suptitle"):
        flat["suptitle"] = spec["suptitle"]
    if spec.get("figlegend"):
        flat["figlegend.labels"] = spec["figlegend"]
    if spec.get("figlegend_entries"):
        flat["figlegend.entries"] = spec["figlegend_entries"]
    for key in ("figure_size_inches", "figure_dpi", "canvas_size_px"):
        if spec.get(key) is not None:
            flat[key] = spec[key]

    for ai, ax in enumerate(spec.get("axes") or []):
        base = "axes[%d]" % ai
        for key in SCALAR_KEYS:
            if ax.get(key) is not None:
                flat["%s.%s" % (base, key)] = ax[key]
        for key in LIST_KEYS:
            if ax.get(key):
                flat["%s.%s" % (base, key)] = ax[key]
        for key in TICK_POLICY_KEYS:
            if isinstance(ax.get(key), dict):
                flat["%s.%s" % (base, key)] = ax[key]
        leg = ax.get("legend")
        if leg:
            if leg.get("labels"):
                flat["%s.legend.labels" % base] = leg["labels"]
            if leg.get("entries"):
                flat["%s.legend.entries" % base] = leg["entries"]
            if leg.get("title"):
                flat["%s.legend.title" % base] = leg["title"]

        series = ax.get("series") or []
        for si, s in enumerate(series):
            key = _series_key(ai, series, si)
            flat["%s.kind" % key] = s.get("kind")
            for extra in (
                "color", "edgecolor", "colors", "sizes", "clim", "rendered_colors",
                "color_alpha", "marker", "linestyle", "linewidth", "markersize",
                "markerfacecolor", "markerfacecolor_alpha", "markeredgecolor",
                "markeredgecolor_alpha", "markeredgewidth", "fillstyle", "drawstyle",
                "alpha", "zorder",
            ):
                if s.get(extra) is not None:
                    flat["%s.%s" % (key, extra)] = s[extra]
            if s.get("values"):
                flat["%s.values" % key] = s["values"]
    return flat


def series_of(spec: dict) -> list:
    """All series across axes, tagged with their axis index (for Data Fidelity)."""
    out = []
    for ai, ax in enumerate(spec.get("axes") or []):
        for s in ax.get("series") or []:
            item = dict(s)
            item["axis"] = ai
            out.append(item)
    return out
