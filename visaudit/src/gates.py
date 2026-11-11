"""Quality gates: is a perturbed render valid and visibly different from GT?

The visibility comparison runs on downsampled RGB images (max over the three
channels) — grayscale would hide colour-only flaws, because e.g. #E15759 and
#59A14F differ by only 0.006 in luminance but by 0.53 in the red channel.
Grayscale is used only for blank detection and the soft agreement number.

This is deliberately a coarse, honest signal (not SSIM): it exists to catch
silent no-op perturbations and degenerate/blank renders, and to give a soft
agreement number between the local GT re-render and the dataset's stored PNG.
"""
import io

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

DOWNSAMPLE = 448          # max side length for comparisons
PIXEL_DELTA = 0.04        # per-pixel, per-channel change that counts as "different"
VISIBLE_FRAC = 0.001      # >=0.1% of pixels changed => visibly different
                          # (safe because the pinned env renders deterministically:
                          #  true no-ops measure 0.0, real text edits >= ~0.002)
BLANK_FRAC = 0.995        # >=99.5% of pixels near the modal value => blank


def _rgb(png_bytes: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(png_bytes)) as im:
        im = im.convert("RGB")
        im.thumbnail((DOWNSAMPLE, DOWNSAMPLE), Image.BILINEAR)
        return np.asarray(im, dtype=np.float32) / 255.0


def _gray(png_bytes: bytes, force_square: bool = False) -> np.ndarray:
    with Image.open(io.BytesIO(png_bytes)) as im:
        im = im.convert("L")
        if force_square:
            im = im.resize((DOWNSAMPLE, DOWNSAMPLE), Image.BILINEAR)
        else:
            im.thumbnail((DOWNSAMPLE, DOWNSAMPLE), Image.BILINEAR)
        return np.asarray(im, dtype=np.float32) / 255.0


def is_blank(png_bytes: bytes) -> bool:
    a = _gray(png_bytes)
    hist, edges = np.histogram(a, bins=64, range=(0.0, 1.0))
    mode = (edges[hist.argmax()] + edges[hist.argmax() + 1]) / 2
    return float((np.abs(a - mode) < 0.02).mean()) >= BLANK_FRAC


def compare(gt_png: bytes, other_png: bytes) -> dict:
    """RGB comparison — color-only flaws (e.g. red<->blue swaps) stay visible
    even when the two colors have near-identical luminance."""
    a, b = _rgb(gt_png), _rgb(other_png)
    if a.shape != b.shape:
        return {"pixel_diff_frac": 1.0, "size_changed": True}
    frac = float((np.abs(a - b).max(axis=2) > PIXEL_DELTA).mean())
    return {"pixel_diff_frac": round(frac, 5), "size_changed": False}


def visibility_gate(gt_png: bytes, pert_png: bytes) -> dict:
    cmp = compare(gt_png, pert_png)
    blank = is_blank(pert_png)
    visible = (not blank) and (cmp["size_changed"] or cmp["pixel_diff_frac"] >= VISIBLE_FRAC)
    return {"visibly_different": visible, "blank": blank, **cmp}


def _norm(s: str) -> str:
    return " ".join(str(s).lower().split())


def _num_variants(v) -> list:
    """String forms a number may take inside table.csv (24 vs 24.0)."""
    out = [str(v)]
    if isinstance(v, float) and v == int(v):
        out.append(str(int(v)))
    if isinstance(v, int):
        out.append(f"{v}.0")
    return out


def _found(needle: str, hay: str) -> bool:
    n = _norm(needle)
    if not n:
        return False
    if n in hay:
        return True
    words = [w for w in n.split() if len(w) > 3]
    if len(words) >= 4:  # long titles: tolerate reformulation via word overlap
        hits = sum(1 for w in words if w in hay)
        return hits / len(words) >= 0.7
    return False


# Must stay in sync with transforms.COLOR_WORDS — a colour the generator
# recognises but the gate does not would reject valid instances.
COLOR_WORDS_FALLBACK = {
    "blue", "red", "green", "orange", "purple", "brown", "pink", "gray", "grey",
    "yellow", "black", "cyan", "magenta", "teal", "gold", "crimson", "navy",
    "salmon", "olive", "maroon", "violet", "turquoise", "coral", "indigo",
}


def check_solvable(evidence: dict, summary: str, csv_text: str):
    """Per-instance solvability: does the untouched evidence (summary/table)
    actually carry the information that the perturbation contradicts?
    This is the creation-time quality check — instances whose flaw is
    undetectable from the given modalities must not count as valid."""
    if not evidence:
        return False, "no evidence declared by transform"
    s_hay, t_hay = _norm(summary), _norm(csv_text)
    both_hay = s_hay + "\n" + t_hay

    if evidence.get("summary_any_color"):
        ok = any(w in s_hay for w in COLOR_WORDS_FALLBACK)
        return ok, "summary names colors" if ok else "summary names no colors"

    needles_s = evidence.get("summary", [])
    hits_s = sum(1 for n in needles_s if _found(n, s_hay))
    need_s = evidence.get("min_summary", len(needles_s))
    if need_s > len(needles_s):
        # A contract demanding more witnesses than it declares is malformed,
        # not vacuously satisfied. The old min() silently weakened it: a
        # two-witness contract with one declared needle was happy with one.
        return False, (f"malformed evidence: min_summary {need_s} exceeds "
                       f"{len(needles_s)} declared needles")
    if hits_s < need_s:
        return False, f"summary evidence missing ({hits_s}/{need_s} needles found)"

    for n in evidence.get("table", []):
        if not _found(n, t_hay):
            return False, f"table evidence missing: {n!r}"

    for v in evidence.get("table_nums", []):
        if not any(var in t_hay for var in _num_variants(v)):
            return False, f"value {v!r} not present in table.csv"

    for n in evidence.get("either", []):
        if not _found(n, both_hay):
            return False, f"evidence {n!r} in neither summary nor table"

    return True, "all evidence present"


def coarse_agreement(local_png: bytes, dataset_png: bytes) -> float:
    """Soft similarity between local re-render and stored dataset image (0..1).

    Forced-square grayscale mean absolute agreement — fonts/version drift make
    pixel equality meaningless, so this is reported as a sanity number only.
    """
    a = _gray(local_png, force_square=True)
    b = _gray(dataset_png, force_square=True)
    return round(1.0 - float(np.abs(a - b).mean()), 4)


# =====================================================================
# Gates for LLM-GENERATED perturbation code.
#
# When an LLM rewrites the whole script (instead of us splicing one span),
# the diff is no longer minimal by construction — it has to be *verified*.
# These gates enforce, per category, that the model changed only what it
# was asked to change. That is what keeps diff.patch usable as ground truth
# for later Recovery / Preservation scoring.
# =====================================================================
import ast as _ast
import difflib as _difflib
import io as _io
import tokenize as _tokenize
from collections import Counter as _Counter

# Max changed lines per category. NOT guessed — calibrated by running
# src/calibrate_gates.py over all 2810 rule-based edits, whose minimality is
# guaranteed by construction; these are the observed maxima plus headroom.
# (A multi-line legend() call or a data array spanning 24 lines is a legitimate
# single edit, so a naive "one line" budget rejects correct perturbations.)
LINE_BUDGET = {
    "wrong-title": 2, "value-perturbation": 2, "axis-truncation": 2,
    "wrong-axis-label": 3, "swapped-legend-labels": 3,
    "color-contradicts-summary": 3, "wrong-chart-type": 6,
    "missing-legend": 12, "dropped-series": 12, "swapped-series": 26,
}
HARD_LINE_CAP = 30
# Relative cap: an edit may never touch more than this share of the file.
# This is what actually catches "the model reformatted everything" — for the
# large-budget categories the literal-integrity gates below carry the strictness.
MAX_CHANGED_FRACTION = 0.4
# allowed change in total line count (min, max); default is exactly 0
LINE_DELTA = {
    "missing-legend": (-12, 0), "dropped-series": (-12, 0),
    "axis-truncation": (0, 2), "swapped-series": (-4, 4), "wrong-chart-type": (-2, 2),
}
# what may happen to the numeric data literals
LITERAL_POLICY = {
    "value-perturbation": "one_changed", "swapped-series": "permutation",
    # deleting a legend()/plot call legitimately removes its numeric kwargs
    # (e.g. bbox_to_anchor=(1, 1.05)) — numbers may vanish, never appear
    "dropped-series": "subset", "missing-legend": "subset",
    "axis-truncation": "two_added",
}  # everything else: "identical"

BANNED_CALLS = {"exec", "eval", "compile", "__import__", "input"}


def _numbers(src):
    """All numeric literals in SOURCE order (ast.walk is breadth-first, so its
    order does not reflect the file and would make a real swap look unchanged)."""
    try:
        tree = _ast.parse(src)
    except SyntaxError:
        return None
    found = []
    for n in _ast.walk(tree):
        if isinstance(n, _ast.Constant) and isinstance(n.value, (int, float)) \
                and not isinstance(n.value, bool):
            found.append((getattr(n, "lineno", 0), getattr(n, "col_offset", 0), n.value))
    return [v for _, _, v in sorted(found, key=lambda t: (t[0], t[1]))]


def _strings(src):
    try:
        tree = _ast.parse(src)
    except SyntaxError:
        return None
    return [n.value for n in _ast.walk(tree)
            if isinstance(n, _ast.Constant) and isinstance(n.value, str)]


def _imports(src):
    try:
        tree = _ast.parse(src)
    except SyntaxError:
        return set()
    mods = set()
    for n in _ast.walk(tree):
        if isinstance(n, _ast.Import):
            mods.update(a.name.split(".")[0] for a in n.names)
        elif isinstance(n, _ast.ImportFrom) and n.module:
            mods.add(n.module.split(".")[0])
    return mods


def _output_targets(src):
    """Constant filenames passed to savefig / write_image / imsave."""
    try:
        tree = _ast.parse(src)
    except SyntaxError:
        return None
    out = []
    for n in _ast.walk(tree):
        if not isinstance(n, _ast.Call):
            continue
        fn = n.func
        name = fn.attr if isinstance(fn, _ast.Attribute) else getattr(fn, "id", "")
        if name not in {"savefig", "write_image", "imsave"}:
            continue
        args = list(n.args) + [kw.value for kw in n.keywords if kw.arg in ("fname", "file")]
        for a in args:
            if isinstance(a, _ast.Constant) and isinstance(a.value, str):
                out.append(a.value)
    return sorted(out)


def _comments(src):
    try:
        toks = _tokenize.generate_tokens(_io.StringIO(src).readline)
        return sorted(t.string.strip() for t in toks if t.type == _tokenize.COMMENT)
    except (_tokenize.TokenError, IndentationError, SyntaxError):
        return []


def diff_stats(gt_src, new_src):
    a, b = gt_src.splitlines(), new_src.splitlines()
    sm = _difflib.SequenceMatcher(None, a, b, autojunk=False)
    added = removed = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "delete"):
            removed += i2 - i1
        if tag in ("replace", "insert"):
            added += j2 - j1
    return {"added": added, "removed": removed,
            "changed": max(added, removed), "line_delta": len(b) - len(a)}


def check_llm_edit(gt_src, new_src, category):
    """Verify an LLM-produced script changed ONLY what the category allows.

    Returns (ok, note, metrics). A failure note is written back to the model
    on retry, so it is phrased as actionable feedback.
    """
    if not new_src or not new_src.strip():
        return False, "empty code returned", {}
    try:
        new_tree = _ast.parse(new_src)
        gt_tree = _ast.parse(gt_src)
    except SyntaxError as exc:
        return False, f"perturbed code does not parse: {exc}", {}

    d = diff_stats(gt_src, new_src)
    budget = LINE_BUDGET.get(category, 2)

    # 1 — semantic no-op: AST identical means only comments/whitespace moved
    if _ast.dump(gt_tree) == _ast.dump(new_tree):
        return False, "no semantic change (AST identical to ground truth)", d

    # 2 — edit locality
    if d["changed"] > HARD_LINE_CAP:
        return False, (f"changed {d['changed']} lines — you must not reformat the file; "
                       f"change at most {budget} line(s)"), d
    if d["changed"] > budget:
        return False, (f"changed {d['changed']} lines, allowed for '{category}' is {budget}"), d
    n_lines = max(1, len(gt_src.splitlines()))
    if d["changed"] / n_lines > MAX_CHANGED_FRACTION:
        return False, (f"changed {d['changed']} of {n_lines} lines — that is a rewrite, "
                       "not a targeted edit"), d
    lo, hi = LINE_DELTA.get(category, (0, 0))
    if not lo <= d["line_delta"] <= hi:
        return False, (f"line count changed by {d['line_delta']}, allowed range is [{lo}, {hi}]"), d

    # 3 — comments must be untouched (models like to annotate their own edit)
    if _comments(gt_src) != _comments(new_src):
        return False, "comments were added, removed or altered — leave all comments untouched", d

    # 4 — output path preserved (otherwise the harness can't find the render)
    if _output_targets(gt_src) != _output_targets(new_src):
        return False, "the savefig/write_image output filename was changed — keep it byte-identical", d

    # 5 — no new imports, no dangerous calls
    new_imports = _imports(new_src) - _imports(gt_src)
    if new_imports:
        return False, f"introduced new imports {sorted(new_imports)} — none are allowed", d
    # only NEW dangerous calls count — a few ChartNet ground-truth scripts
    # legitimately contain eval(), and inheriting them is not an attack
    def _banned(src):
        try:
            t = _ast.parse(src)
        except SyntaxError:
            return _Counter()
        return _Counter(n.func.id for n in _ast.walk(t)
                        if isinstance(n, _ast.Call) and getattr(n.func, "id", "") in BANNED_CALLS)
    introduced = _banned(new_src) - _banned(gt_src)
    if introduced:
        return False, f"introduced forbidden call(s) {sorted(introduced)}", d

    # 6 — data-literal integrity, per category policy
    gt_nums, new_nums = _numbers(gt_src), _numbers(new_src)
    gt_strs, new_strs = _strings(gt_src), _strings(new_src)
    policy = LITERAL_POLICY.get(category, "identical")
    cg, cn = _Counter(gt_nums), _Counter(new_nums)
    if policy == "identical":
        if cg != cn:
            gone = sorted((cg - cn).elements())[:3]
            return False, (f"numeric data literals changed ({gone}…) — this category must not "
                           "touch any number"), d
    elif policy == "one_changed":
        if not (sum((cg - cn).values()) == 1 and sum((cn - cg).values()) == 1):
            return False, "exactly one numeric literal must change — no more, no less", d
        if _Counter(gt_strs) != _Counter(new_strs):
            return False, "changing a value must not change any text literal", d
    elif policy == "permutation":
        if cg != cn:
            return False, "series values must be swapped, not modified — multiset must stay equal", d
        if gt_nums == new_nums:
            return False, "numbers are in the same order — nothing was actually swapped", d
    elif policy == "subset":
        if cn - cg:
            return False, "removing a series must not introduce new numbers", d
    elif policy == "two_added":
        if sum((cg - cn).values()) != 0 or sum((cn - cg).values()) > 2:
            return False, "only the two new axis-limit numbers may be added", d

    return True, "edit within category budget", d
