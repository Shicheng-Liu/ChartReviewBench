"""The evaluation dataset: a named, reproducible slice of the instance pool.

This module selects instances from `<root>/samples` and records their identities,
selection method and category quotas in a dataset manifest.

The selected validity version determines which gates bind. In v0, `exec_ok`
binds; solvability and edit-budget diagnostics are recorded without excluding
instances. The versioned validity notes describe additional gates.

Selection within a category uses round robin sampling over library and chart
type to avoid favouring whichever stratum sorts first. Each category records
requested, available, selected and missing counts; `build` warns on shortfalls.

    .venv/bin/python -m src.eval.dataset --name v0-1000 --n 100
    .venv/bin/python -m src.eval.dataset --stats v0-1000
"""
import argparse
import datetime
import json
import random
import sys
from collections import Counter, deque
from pathlib import Path

VERSION = "1.0"
DEFAULT_ROOT = Path("data/v0")

# The v0 taxonomy, kept explicit: a category that has vanished from the pool must
# show up as available 0, not silently drop out of the dataset.
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

# The files an item cannot be run without. `load` refuses a dataset whose items
# are not backed by these.
REQUIRED_FILES = ("flawed_code", "gt_code")


# --------------------------------------------------------------------------- paths

def flaw_dir(item: dict) -> str:
    """The folder under perturbed_llm/ that holds this item's flawed script.

    Usually the category itself. It differs when a category is REPORTED as more
    than one population — axis-truncation splits into a pinned and a free case,
    which are the same injected flaw judged by different rules, so they share one
    generation run and one folder while counting as two categories everywhere
    else. Keeping the split at the reporting layer means no instance is
    duplicated on disk and no earlier dataset changes meaning.
    """
    return item.get("flaw_dir") or item["category"]


def item_paths(item: dict, root=DEFAULT_ROOT) -> dict:
    """Where one dataset item lives on disk. Existence is not checked here."""
    inst = Path(root) / "samples" / item["instance"]
    flawed = inst / "perturbed_llm" / flaw_dir(item)
    return {
        "instance_dir": inst,
        "flawed_code": flawed / "code.py",
        "flawed_render": flawed / "render.png",
        "gt_code": inst / "gt" / "code.py",
        "table": inst / "gt" / "table.csv",
        "summary": inst / "gt" / "summary.txt",
    }


def dataset_path(name_or_path, root=DEFAULT_ROOT, out_dir=None) -> Path:
    p = Path(name_or_path)
    if p.suffix == ".json":
        return p
    return (Path(out_dir) if out_dir is not None else Path(root) / "datasets") / (p.name + ".json")


# --------------------------------------------------------------------------- the pool

def _manifest(root: Path, include_single=True, include_multi=True):
    """Every instance the pipeline knows about, single-flaw and multi-flaw.

    Multi-flaw instances are recorded in their own manifest because they are
    produced by a different path, but they live in the same sample tree and must
    pass through the same validity handling. Reading only the single-flaw
    manifest made them 'not in manifest', so they could only enter a dataset via
    --include-invalid — that is, by skipping every gate.
    """
    rows, sources = {}, []
    names = []
    if include_single:
        names.append("manifest_llm.jsonl")
    if include_multi:
        names.append("manifest_multiflaw.jsonl")
    for name in names:
        path = root / name
        if not path.exists():
            continue
        sources.append(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows[(r["sample_folder"], r["category"])] = r
    return rows, sources


def _index(root: Path):
    path = root / "index.json"
    if not path.exists():
        return {}, None
    return {d["folder"]: d for d in json.loads(path.read_text(encoding="utf-8"))}, path


def _judge_votes(root: Path):
    """Cross-family judge verdicts per instance_id, 'current' prompt variant only.

    Binding votes come from ENDPOINT judge families distinct from the generator
    family to avoid self-preference bias. The Anthropic opus verdict is loaded
    as a diagnostic only.
    """
    votes, opus, sources = {}, {}, []
    endpoint_path = root / "judge_results_endpoint.json"
    if endpoint_path.exists():
        sources.append(endpoint_path)
        for r in json.loads(endpoint_path.read_text(encoding="utf-8")):
            if (r.get("prompt_variant", "current") != "current"
                    or "llm_detectable" not in r):
                continue
            v = votes.setdefault(r["instance_id"], {"n": 0, "detectable": 0})
            v["n"] += 1
            v["detectable"] += bool(r["llm_detectable"])
    anth_path = root / "judge_results.json"
    if anth_path.exists():
        sources.append(anth_path)
        for r in json.loads(anth_path.read_text(encoding="utf-8")):
            if "llm_detectable" in r:
                opus[r["instance_id"]] = bool(r["llm_detectable"])
    return votes, opus, sources


def _multiflaw_validity(row: dict, votes: dict, validity: str):
    """(judge record, None) when a multi-flaw row passes the level, else (None, reason).

    A multi-flaw instance is a chain of single-flaw edits, so its validity is the
    conjunction over its components: the chain witness (each component's own
    figure-object footprint survives into the final chart, recorded by
    multiflaw.py when the chain was built), the postconditions the level binds,
    evaluated on the FINAL chart, and a cross-family judge majority for every
    component (endpoint_judge.py --multiflaw, keyed instance_id::category). The
    combined artifact has no entry of its own in footprints.json; the chain
    witness is that footprint.
    """
    gates = row.get("gates") or {}
    if gates.get("survival_witness") is not True:
        return None, "multiflaw_witness_missing"
    components = list(row.get("flaw_categories") or [])
    if not components:
        return None, "multiflaw_components_missing"
    survival = row.get("survival") or {}
    for component in components:
        if component in BOUND_POSTCONDITIONS[validity]:
            # The stage gate binds: the component's postcondition held when it
            # was injected, and the chain witness above proves its paths survive
            # into the final chart. The final-chart gate is recorded as a
            # diagnostic only, because it also rejects the partner flaws' edits.
            detail = survival.get(component)
            stage = detail.get("semantic_gate") if isinstance(detail, dict) else None
            if stage is None:
                return None, "semantic_gate_missing"
            if stage.get("ok") is not True:
                return None, "semantic_gate_rejected"
    per_component = {}
    for component in components:
        v = votes.get("%s::%s" % (row.get("instance_id"), component))
        if v is None or v["n"] < V1_MIN_VOTES:
            return None, "judge_coverage_missing"
        if v["detectable"] * 2 <= v["n"]:
            return None, "judge_rejected"
        per_component[component] = {"families": v["n"], "detectable": v["detectable"]}
    return {"families": min(c["families"] for c in per_component.values()),
            "detectable": min(c["detectable"] for c in per_component.values()),
            "components": per_component}, None


def _footprints(root: Path, footprints_path=None):
    path = Path(footprints_path) if footprints_path else root / "footprints.json"
    if not path.exists():
        return {}, path
    return json.loads(path.read_text(encoding="utf-8")), path


def _axis_case(root: Path, folder: str, fp_rec: dict):
    """Use the scorer's text/geometry binding; None means unclassified.

    Tick endpoints and axis sides require a reference capture. Limits alone
    cannot distinguish a tick span from bounds, or identify a secondary axis.
    Missing captures and ambiguous claims must not be reported as 'free'.
    """
    from . import axis_rule
    from .capture import capture_path
    import re

    lims = (fp_rec or {}).get("gt_lims") or {}
    if not lims:
        return None
    directory = root / "samples" / folder / "gt"
    if not all((directory / name).is_file() for name in ("code.py", "table.csv", "summary.txt")):
        return None
    extra = {name: (directory / name).read_text(encoding="utf-8", errors="replace")
             for name in ("table.csv", "summary.txt")}
    capture = capture_path(directory / "code.py", extra, 120)
    if capture.get("status") != "ok" or not (capture.get("spec") or {}).get("axes"):
        return None
    spec, flat = capture["spec"], dict(capture.get("flat") or {})
    decisions = []
    for path, value in lims.items():
        match = re.fullmatch(r"axes\[(\d+)\]\.([xy])lim", path)
        if match is None:
            continue
        axes_index, letter = int(match[1]), match[2]
        if axes_index >= len(spec["axes"]):
            return None
        if flat.get(path) is None and value is not None:
            flat[path] = value
        decisions.append(axis_rule.axis_constraint(
            extra["summary.txt"], axes_index, letter, spec, flat)["status"])
    if not decisions or "ambiguous" in decisions:
        return None
    return "pinned" if "bound" in decisions else "free"


def _data_extent(root: Path, folder: str, path: str):
    """What the plotted data spans on the axis named by `path`, or None."""
    from . import axis_rule
    import re
    m = re.search(r"axes\[(\d+)\]\.([xy])lim$", path)
    if not m:
        return None
    from .capture import capture_path
    d = root / "samples" / folder / "gt"
    extra = {"table.csv": (d / "table.csv").read_text(encoding="utf-8", errors="replace")}
    if (d / "summary.txt").exists():
        extra["summary.txt"] = (d / "summary.txt").read_text(encoding="utf-8", errors="replace")
    cap = capture_path(d / "code.py", extra, 120)
    if cap["status"] != "ok":
        return None
    return axis_rule.data_extent(cap.get("spec") or {}, int(m.group(1)), m.group(2), None)


# How many cross-family verdicts a v1 instance needs before the majority binds,
# and the majority itself: strictly more than half of the votes cast.
V1_MIN_VOTES = 3


def _pool(root: Path, only_valid: bool, validity: str = "v0", split_axis: bool = False,
          footprints_path=None, categories=None):
    """Every flawed instance on disk, joined to the metadata that describes it.

    The disk is the pool and the manifest decides validity, so an instance the
    manifest does not list is not valid, and an instance the manifest lists but
    that is not on disk is not in the pool at all.

    validity "v0" binds exec_ok alone. validity "v1" additionally requires a
    cross-family LLM-judge majority (the flaw is witnessed by summary or table),
    a non-empty figure-object footprint, and any category-specific semantic
    postcondition. For missing-legend that final check proves concrete GT legend
    labels actually disappeared instead of merely moving or being restyled.
    These extra gates follow the QC pipeline: the RULE check still only confirms,
    it never rejects.
    """
    requested = set(categories if categories is not None else CATEGORIES)
    physical_requested = {
        "axis-truncation" if c in ("axis-truncation-pinned", "axis-truncation-free") else c
        for c in requested
    }
    wants_multi = any(c.startswith("multi") for c in physical_requested)
    wants_single = any(not c.startswith("multi") for c in physical_requested)
    manifest, manifest_paths = _manifest(
        root, include_single=wants_single, include_multi=wants_multi)
    index, index_path = _index(root)
    sources = list(manifest_paths) + ([index_path] if index_path is not None else [])
    votes, opus, judge_sources = _judge_votes(root) if validity in V1_FAMILY else ({}, {}, [])
    sources.extend(judge_sources)
    fps, fp_path = _footprints(root, footprints_path) if validity in V1_FAMILY else ({}, None)
    if fp_path is not None:
        sources.append(fp_path)

    entries, excluded = [], Counter()
    for code in sorted((root / "samples").glob("*/perturbed_llm/*/code.py")):
        folder, category = code.parents[2].name, code.parent.name
        reported_category = (category.split("__", 1)[0]
                             if category.startswith("multi") and "__" in category
                             else category)
        if reported_category not in physical_requested:
            continue
        if not (root / "samples" / folder / "gt" / "code.py").exists():
            excluded["gt_code_missing"] += 1
            continue
        row = manifest.get((folder, category))
        if only_valid:
            if row is None:
                excluded["not_in_manifest"] += 1
                continue
            if not row.get("gates", {}).get("exec_ok"):
                excluded["exec_failed"] += 1
                continue
        judge = None
        is_multiflaw_row = category.startswith("multi") and "__" in category
        if validity in V1_FAMILY and only_valid and is_multiflaw_row:
            judge, reason = _multiflaw_validity(row or {}, votes, validity)
            if reason:
                excluded[reason] += 1
                continue
        elif validity in V1_FAMILY and only_valid:
            iid = (row or {}).get("instance_id")
            fp = fps.get(iid)
            if fp is None or fp.get("status") != "ok":
                excluded["footprint_missing"] += 1
                continue
            if fp.get("empty"):
                excluded["footprint_empty"] += 1
                continue
            if category in BOUND_POSTCONDITIONS[validity]:
                semantic = fp.get("semantic_gate")
                if semantic is None:
                    excluded["semantic_gate_missing"] += 1
                    continue
                if semantic.get("ok") is not True:
                    excluded["semantic_gate_rejected"] += 1
                    continue
            v = votes.get(iid)
            if v is None or v["n"] < V1_MIN_VOTES:
                excluded["judge_coverage_missing"] += 1
                continue
            if v["detectable"] * 2 <= v["n"]:
                excluded["judge_rejected"] += 1
                continue
            judge = {"families": v["n"], "detectable": v["detectable"],
                     "opus_diagnostic": opus.get(iid)}
        meta = index.get(folder, {})
        # Multi-flaw folders carry their combination in the name so that one
        # source chart can host several of them. What is REPORTED is the arity
        # (multi2, multi3); the folder stays reachable through flaw_dir, the
        # same split the axis cases use.
        reported, fdir = category, None
        if category.startswith("multi") and "__" in category:
            reported, fdir = category.split("__", 1)[0], category
        entry = {
            "instance": folder,
            "category": reported,
            **({"flaw_dir": fdir} if fdir else {}),
            "library": (row or {}).get("library") or meta.get("library") or "unknown",
            "chart_type": (row or {}).get("chart_type") or meta.get("chart_type") or "unknown",
            "gates": (row or {}).get("gates") or {},
            "judge": judge,
            "verification_method": (row or {}).get("verification_method"),
        }
        if validity in V1_FAMILY and category == "axis-truncation":
            entry["axis_case"] = _axis_case(root, folder,
                                            fps.get((row or {}).get("instance_id")))
            if split_axis and entry["axis_case"]:
                # Split at the POOL, not after drawing: otherwise one quota is
                # divided between the two cases and each reports a phantom
                # shortfall. They are separate populations, so they draw
                # separately and report their own availability.
                entry["flaw_dir"] = category
                entry["category"] = "%s-%s" % (category, entry["axis_case"])
        entries.append(entry)
    return entries, excluded, sources


# --------------------------------------------------------------------------- selection

def _spread(entries: list, n: int, rng: random.Random) -> list:
    """Draw n entries by round robin over library, then over chart type.

    Taking the first n would hand a category to whichever library sorts first;
    rotating one library at a time, and inside it one chart type at a time,
    keeps the draw spread while an exhausted stratum simply drops out of the
    rotation. The returned order is the draw order, so even a truncated run of
    the dataset stays spread.
    """
    by_lib: dict = {}
    for e in entries:
        by_lib.setdefault(e["library"], {}).setdefault(e["chart_type"], []).append(e)

    for lib in sorted(by_lib):  # sorted first, so the shuffles are reproducible
        for chart in sorted(by_lib[lib]):
            by_lib[lib][chart].sort(key=lambda e: e["instance"])
            rng.shuffle(by_lib[lib][chart])

    libs = sorted(by_lib)
    rng.shuffle(libs)
    charts = {}
    for lib in libs:
        order = sorted(by_lib[lib])
        rng.shuffle(order)
        charts[lib] = deque(order)

    queue, picked = deque(libs), []
    while queue and len(picked) < n:
        lib = queue.popleft()
        rotation, taken = charts[lib], None
        while rotation:
            chart = rotation.popleft()
            bucket = by_lib[lib][chart]
            if bucket:
                taken = bucket.pop()
                rotation.append(chart)
                break
        if taken is not None:
            picked.append(taken)
            queue.append(lib)  # an exhausted library is not put back
    return picked


def _spread_multiflaw(entries: list, n: int, rng: random.Random) -> list:
    """Round-robin over flaw combinations, then library and chart type.

    Multi-flaw items report one category (for example ``multi2``), while their
    physical ``flaw_dir`` carries the pair. Treating all pairs as one pool made
    the old 500-item artifact range from one to 39 examples per pair and omitted
    an allowed pair from the first 300. Combination is therefore the primary
    stratum; the ordinary spread is applied inside each bucket.
    """
    by_combo = {}
    for entry in entries:
        fdir = entry.get("flaw_dir") or ""
        combo = fdir.split("__", 1)[1] if "__" in fdir else "unknown"
        by_combo.setdefault(combo, []).append(entry)

    queues = {}
    for combo in sorted(by_combo):
        local_rng = random.Random("%.17g|%s" % (rng.random(), combo))
        queues[combo] = deque(_spread(by_combo[combo], len(by_combo[combo]), local_rng))
    order = sorted(queues)
    rng.shuffle(order)
    rotation, picked = deque(order), []
    while rotation and len(picked) < n:
        combo = rotation.popleft()
        if queues[combo]:
            picked.append(queues[combo].popleft())
        if queues[combo]:
            rotation.append(combo)
    return picked


def _category_stats(requested: int, available: int, picked: list) -> dict:
    strata = Counter("%s | %s" % (e["library"], e["chart_type"]) for e in picked)
    gates = [e["gates"] for e in picked]
    result = {
        "requested": requested,
        "available": available,
        "selected": len(picked),
        "short": max(0, requested - len(picked)),
        "libraries": dict(sorted(Counter(e["library"] for e in picked).items())),
        "chart_types": dict(sorted(Counter(e["chart_type"] for e in picked).items())),
        "strata": dict(sorted(strata.items())),
        # Recorded, not applied: see the module docstring on why neither binds.
        "non_binding_gates": {
            "rule_gate_rejected": sum(1 for g in gates if g.get("solvable") is False),
            "edit_over_budget": sum(1 for g in gates if g.get("edit_within_budget") is False),
        },
    }
    methods = Counter(e.get("verification_method") for e in picked
                      if e.get("verification_method"))
    if methods:
        result["verification_methods"] = dict(sorted(methods.items()))
    return result


# --------------------------------------------------------------------------- public API

# Validity levels that add the judge, footprint and semantic gates on top of
# exec_ok, and which category postconditions each level binds. v1.3
# additionally binds the clipping-based AxisTruncation postcondition, whose
# target (limits and clipped truth points) the deterministic transform records.
V1_FAMILY = frozenset({"v1", "v1.3"})
BOUND_POSTCONDITIONS = {
    "v0": frozenset(),
    "v1": frozenset({"missing-legend"}),
    "v1.3": frozenset({"missing-legend", "axis-truncation"}),
}

VALIDITY_NOTES = {
    "v0": {
        "binding": ["gates.exec_ok"],
        "recorded_not_binding": ["gates.solvable", "gates.edit_within_budget"],
        "note": ("the rule solvability gate may confirm but not reject; "
                 "the edit budget is measured, not enforced"),
    },
    "v1": {
        "binding": ["gates.exec_ok", "judge_majority_cross_family", "footprint_nonempty",
                    "category_semantic_postcondition"],
        "recorded_not_binding": ["gates.solvable", "gates.edit_within_budget",
                                 "judge.opus_diagnostic"],
        "note": ("exec_ok: the flawed script renders. judge majority: more than half of at "
                 "least %d ENDPOINT judge families (a different family than the claude-sonnet-5 "
                 "generator) call the flaw detectable from summary/table. footprint_nonempty: "
                 "the gt-vs-flawed figure-object diff is non-empty, so the flaw reaches the "
                 "rendered chart. The category-specific semantic postcondition is currently "
                 "implemented for missing-legend only: it must remove at least one concrete "
                 "GT legend-label group rather than merely restyle or move it. The rule gate "
                 "only confirms, never rejects." % V1_MIN_VOTES),
    },
    "v1.3": {
        "binding": ["gates.exec_ok", "judge_majority_cross_family", "footprint_nonempty",
                    "category_semantic_postcondition",
                    "axis_truncation_clipping_postcondition"],
        "recorded_not_binding": ["gates.solvable", "gates.edit_within_budget",
                                 "judge.opus_diagnostic"],
        "note": ("v1 plus the clipping-based AxisTruncation definition: an axis-truncation instance "
                 "is valid only if its recorded limits actually clip at least one "
                 "truth-backed plotted point (deterministic transform, target persisted "
                 "in perturbation.json). Axis instances without a target fail this gate. The generator is "
                 "Mistral Medium 3.5 on ENDPOINT; judge families exclude the generator family. "
                 "Multi-flaw instances are valid only as the conjunction of their "
                 "components: chain witness (every component's footprint survives into "
                 "the final chart), the bound postconditions evaluated on the final "
                 "chart, and a cross-family judge majority for every component."),
    },
}


def _chart_reuse(items: list) -> dict:
    per_chart = Counter(i["instance"] for i in items)
    reuse = Counter(per_chart.values())
    return {
        "unique_source_charts": len(per_chart),
        "max_categories_per_chart": max(per_chart.values(), default=0),
        "charts_by_reuse_count": {str(k): v for k, v in sorted(reuse.items())},
        "note": ("one source chart can host several perturbation instances by design; "
                 "category-level summaries remain defined, but cross-instance independence "
                 "is not given"),
    }


def _entry_key(item: dict):
    return item.get("instance"), item.get("category"), item.get("flaw_dir")


def build(name: str, n_per_category: int = 100, categories=None, seed: int = 42,
          only_valid: bool = True, root=DEFAULT_ROOT, out_dir=None,
          write: bool = True, validity: str = "v0", split_axis: bool = False,
          footprints_path=None, refill_from=None, subset_of=None) -> dict:
    """Draw a dataset from the pool and write it to `<root>/datasets/<name>.json`.

    Deterministic: the same seed against the same pool produces the same file
    byte for byte. Each category is drawn from its own seeded stream, so adding
    or removing a category does not disturb the others.
    """
    root = Path(root)
    if validity not in VALIDITY_NOTES:
        raise ValueError("unknown validity %r (know: %s)" % (validity, sorted(VALIDITY_NOTES)))
    entries, excluded, sources = _pool(
        root, only_valid, validity, split_axis, footprints_path=footprints_path,
        categories=categories)

    subset_path, subset_keys = None, None
    if subset_of:
        subset_path = dataset_path(subset_of, root)
        parent = json.loads(subset_path.read_text(encoding="utf-8"))
        subset_keys = {_entry_key(item) for item in parent.get("items", [])}
        entries = [entry for entry in entries if _entry_key(entry) in subset_keys]
        sources.append(subset_path)

    refill_path, refill_items = None, {}
    if refill_from:
        refill_path = dataset_path(refill_from, root)
        prior = json.loads(refill_path.read_text(encoding="utf-8"))
        for item in prior.get("items", []):
            refill_items.setdefault(item["category"], []).append(item)
        sources.append(refill_path)

    by_cat: dict = {}
    for e in entries:
        by_cat.setdefault(e["category"], []).append(e)
    if categories is None:
        known = set(CATEGORIES)
        if split_axis:
            known = (known - {"axis-truncation"}) | {"axis-truncation-pinned",
                                                     "axis-truncation-free"}
        categories = sorted(known)
    else:
        categories = list(categories)

    items, stats, refill_stats = [], {}, {}
    for category in categories:
        rng = random.Random("%s|%s" % (seed, category))
        pool = by_cat.get(category, [])
        is_multiflaw = category.startswith("multi") and any(e.get("flaw_dir") for e in pool)
        spread = _spread_multiflaw if is_multiflaw else _spread
        if refill_path:
            available = {_entry_key(e): e for e in pool}
            retained, removed_items, seen = [], [], set()
            for old in refill_items.get(category, []):
                key = _entry_key(old)
                if key in available and key not in seen and len(retained) < n_per_category:
                    retained.append(available[key])
                    seen.add(key)
                else:
                    removed_items.append(old)
            remaining = [e for e in pool if _entry_key(e) not in seen]
            remaining.sort(key=_entry_key)
            rng.shuffle(remaining)
            added = []

            # Replace a rejected item from the same library/chart stratum when
            # possible. A blind refill preserved 986/1000 instances but turned
            # missing-legend from a balanced 33/33/34 library split into
            # 38/37/25 because every rejected leak happened to be seaborn.
            # Matching the removed stratum keeps both the old measurement base
            # and its designed coverage while changing only invalid items.
            for old in removed_items:
                if len(retained) + len(added) >= n_per_category or not remaining:
                    break
                preferences = (
                    lambda e: e.get("library") == old.get("library")
                              and e.get("chart_type") == old.get("chart_type"),
                    lambda e: e.get("library") == old.get("library"),
                    lambda e: e.get("chart_type") == old.get("chart_type"),
                    lambda e: True,
                )
                index = next(i for match in preferences
                             for i, entry in enumerate(remaining) if match(entry))
                added.append(remaining.pop(index))
            if len(retained) + len(added) < n_per_category:
                added.extend(spread(
                    remaining, n_per_category - len(retained) - len(added), rng))
            picked = retained + added
            refill_stats[category] = {
                "retained": len(retained),
                "removed": [old["instance"] for old in removed_items],
                "added": [e["instance"] for e in added],
            }
        else:
            picked = spread(pool, n_per_category, rng)
        stats[category] = _category_stats(n_per_category, len(pool), picked)
        if is_multiflaw:
            stats[category]["flaw_combinations"] = dict(sorted(Counter(
                ((e.get("flaw_dir") or "unknown").split("__", 1)[-1]
                 for e in picked)).items()))
        if validity == "v0":
            # byte-compatible with every dataset built before validity existed —
            # except that flaw_dir must travel when the folder is not the
            # category, or nothing downstream finds the files.
            items.extend({"instance": e["instance"], "category": e["category"],
                          **({"flaw_dir": e["flaw_dir"]} if e.get("flaw_dir") else {})}
                         for e in picked)
        else:
            for e in picked:
                item = {"instance": e["instance"], "category": e["category"],
                        "library": e["library"], "chart_type": e["chart_type"],
                        "judge": e["judge"]}
                if e.get("axis_case"):
                    item["axis_case"] = e["axis_case"]
                # Two reported categories, one folder on disk: the same injected
                # flaw judged by different rules is a different question to an
                # agent, but duplicating the instances would be a lie.
                if e.get("flaw_dir"):
                    item["flaw_dir"] = e["flaw_dir"]
                items.append(item)
            if picked and any(e.get("axis_case") for e in picked):
                stats[category]["axis_cases"] = dict(
                    Counter(e.get("axis_case") or "unknown" for e in picked))

    short = [c for c in categories if stats[c]["short"] > 0]
    mtime = max((p.stat().st_mtime for p in sources if p.exists()), default=None)

    def source_label(path):
        try:
            return str(path.relative_to(root))
        except ValueError:
            return str(path)

    ds = {
        "name": name,
        "version": VERSION,
        "seed": seed,
        "n_per_category": n_per_category,
        "only_valid": only_valid,
        "validity": {**VALIDITY_NOTES[validity], "version": validity},
        "categories": list(categories),
        "source": {
            "root": str(root),
            "files": [source_label(p) for p in sources],
            "mtime": (datetime.datetime.fromtimestamp(mtime, datetime.timezone.utc)
                      .replace(microsecond=0).isoformat() if mtime else None),
            "excluded_from_pool": dict(sorted(excluded.items())),
        },
        "total": len(items),
        "complete": not short,
        "short_categories": short,
        "category_stats": stats,
        "chart_reuse": _chart_reuse(items),
        "items": items,
    }
    if refill_path:
        ds["refill"] = {"from": str(refill_path), "categories": refill_stats}
    if subset_path:
        selected = {_entry_key(item) for item in items}
        outside = selected - subset_keys
        if outside:
            raise RuntimeError("dataset %s selected %d items outside parent %s"
                               % (name, len(outside), subset_path))
        ds["subset_of"] = {"dataset": str(subset_path),
                           "parent_total": len(subset_keys),
                           "verified": True}

    if short:
        _warn_short(ds)
    if write:
        path = dataset_path(name, root, out_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(ds, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return ds


def load(name_or_path, root=DEFAULT_ROOT, check: bool = True) -> dict:
    """Read a dataset back and prove every item is still on disk."""
    path = dataset_path(name_or_path, root)
    if not path.exists():
        raise FileNotFoundError("no such dataset: %s" % path)
    ds = json.loads(path.read_text(encoding="utf-8"))
    if check:
        missing = []
        for item in ds.get("items", []):
            paths = item_paths(item, root)
            for key in REQUIRED_FILES:
                if not paths[key].exists():
                    missing.append("%s / %s: %s missing at %s"
                                   % (item["instance"], item["category"], key, paths[key]))
        if missing:
            shown = "\n  ".join(missing[:10])
            more = "\n  ... and %d more" % (len(missing) - 10) if len(missing) > 10 else ""
            raise FileNotFoundError(
                "dataset %s: %d of %d items are not backed by files under %s:\n  %s%s"
                % (ds.get("name", path.name), len(missing), len(ds.get("items", [])), root, shown, more))
    return ds


def items(ds) -> list:
    """The item list of a dataset, which may be given as a dict, a name or a path."""
    if not isinstance(ds, dict):
        ds = load(ds)
    return list(ds.get("items", []))


# --------------------------------------------------------------------------- reporting

def _warn_short(ds: dict) -> None:
    n = ds["n_per_category"]
    print("warning: dataset '%s' could not fill the quota of %d in %d of %d categories:"
          % (ds["name"], n, len(ds["short_categories"]), len(ds["categories"])), file=sys.stderr)
    width = max(len(c) for c in ds["short_categories"])
    for c in ds["short_categories"]:
        s = ds["category_stats"][c]
        print("  %-*s  %4d of %4d  (%d short)" % (width, c, s["selected"], s["requested"], s["short"]),
              file=sys.stderr)
    print("  a mean over a short category is not comparable to a mean over a full one.",
          file=sys.stderr)


def format_stats(ds: dict) -> str:
    rows = ds["category_stats"]
    width = max([len(c) for c in rows] + [8])
    head = "%-*s  %9s %9s %8s %6s  %s" % (width, "category", "requested", "available",
                                          "selected", "short", "libraries")
    body = []
    for category in ds["categories"]:
        c = rows[category]
        libs = ", ".join("%s %d" % (k, v) for k, v in c["libraries"].items()) or "-"
        body.append("%-*s  %9d %9d %8d %6d  %s"
                    % (width, category, c["requested"], c["available"], c["selected"], c["short"], libs))
    rule = "-" * max(len(line) for line in [head] + body)
    out = [
        "dataset   %s  (v%s, seed %s, %s)"
        % (ds["name"], ds.get("version", "?"), ds["seed"],
           "valid only" if ds.get("only_valid", True) else "all instances"),
        "source    %s  %s" % (ds["source"]["root"], ds["source"].get("mtime") or ""),
        "total     %d items across %d categories%s"
        % (ds["total"], len(ds["categories"]), "" if ds["complete"] else "  (INCOMPLETE)"),
        "",
        head, rule, *body, rule,
        "%-*s  %9s %9s %8d %6d" % (width, "total", "", "",
                                   sum(c["selected"] for c in rows.values()),
                                   sum(c["short"] for c in rows.values())),
    ]
    if ds["short_categories"]:
        out += ["", "short: " + ", ".join(ds["short_categories"]),
                "means over these categories rest on fewer instances and are not comparable."]
    over = {c: rows[c]["non_binding_gates"] for c in ds["categories"]}
    flagged = ["%s (rule %d, edit %d)" % (c, g["rule_gate_rejected"], g["edit_over_budget"])
               for c, g in over.items() if g["rule_gate_rejected"] or g["edit_over_budget"]]
    if flagged:
        out += ["", "recorded but not applied: " + "; ".join(flagged)]
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", help="build a dataset under this name")
    ap.add_argument("--stats", metavar="NAME", help="print what a built dataset contains")
    ap.add_argument("--n", type=int, default=100, help="instances per category (default 100)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--categories", help="comma separated, defaults to the whole taxonomy")
    ap.add_argument("--include-invalid", action="store_true",
                    help="do not filter by the manifest gates")
    ap.add_argument("--validity", default="v0", choices=sorted(VALIDITY_NOTES),
                    help="v0: exec_ok binds. v1: exec_ok + cross-family judge majority "
                         "+ non-empty footprint + semantic category gates bind")
    ap.add_argument("--split-axis", action="store_true",
                    help="report axis-truncation as two categories, -pinned and -free: "
                         "the same injected flaw judged by different rules. One folder "
                         "on disk, two rows in every report (v1 only)")
    ap.add_argument("--footprints",
                    help="alternate footprint JSON, used for corrected rebuilds without "
                         "overwriting the historical footprint artifact")
    ap.add_argument("--refill-from", metavar="DATASET",
                    help="retain still-valid items from this dataset and replace only those "
                         "the current gates reject")
    ap.add_argument("--subset-of", metavar="DATASET",
                    help="restrict the draw to items in this parent dataset and record a "
                         "hard subset invariant (for example panel within release)")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--dry-run", action="store_true", help="report the draw, write nothing")
    args = ap.parse_args(argv)

    if bool(args.name) == bool(args.stats):
        ap.error("give exactly one of --name (build) or --stats (report)")

    if args.stats:
        ds = load(args.stats, Path(args.root), check=False)
        print(format_stats(ds))
        return 0

    ds = build(
        args.name,
        n_per_category=args.n,
        categories=[c.strip() for c in args.categories.split(",")] if args.categories else None,
        seed=args.seed,
        only_valid=not args.include_invalid,
        root=Path(args.root),
        out_dir=args.out_dir,
        write=not args.dry_run,
        validity=args.validity,
        split_axis=args.split_axis,
        footprints_path=args.footprints,
        refill_from=args.refill_from,
        subset_of=args.subset_of,
    )
    print(format_stats(ds))
    if args.dry_run:
        print("\ndry run: nothing written.")
    else:
        print("\nwrote %s" % dataset_path(args.name, Path(args.root), args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
