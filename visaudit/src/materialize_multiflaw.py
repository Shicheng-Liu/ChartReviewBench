"""Turn verified multi-flaw records into instance folders the pipeline can run.

`multiflaw.py` produces and verifies the chained scripts but keeps them in one
cache file. Everything downstream — task building, the runner, the metric —
addresses an instance as `samples/<folder>/perturbed_llm/<category>/`, so the
verified records are written into that shape under a physical folder such as
`multi2__missing-legend+wrong-title`. The dataset reports the logical arity
(`multi2`, `multi3`) and retains the physical folder as `flaw_dir`; this lets one
source chart carry several distinct combinations without overwriting any.

Only records whose verdict is "verified" are written.  The current release
boundary is an independent stored-single-flaw footprint witness for every
component (plus the MissingLegend semantic postcondition where applicable), not
a cross-family judgment of the combined artifact.

The metadata keeps the edits as a LIST, as reviewer asked for, and names both
categories so the evaluation can attribute a repaired property to the flaw it
belongs to.

    .venv/bin/python src/materialize_multiflaw.py --dry-run
    .venv/bin/python src/materialize_multiflaw.py
"""
import argparse
import datetime
import difflib
import itertools
import json
import statistics
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness  # noqa: E402
import multiflaw  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
V0 = ROOT / "data/v0"
SAMPLES = V0 / "samples"
CACHE = V0 / "multiflaw.json"
MANIFEST = V0 / "manifest_multiflaw.jsonl"
QC_REPORT = V0 / "qc/multiflow_release_v12.json"


# The survival checks whose verdicts may enter a dataset: the collision-free
# replay of the historical (Sonnet) records, and the chain check that verifies
# every record built since 01.09. at generation time (multiflaw.CHAIN_BASIS).
ADMITTED_BASES = ("independent-single-flaw-replay-v12", multiflaw.CHAIN_BASIS)


def admitted(rec):
    """Whether a cached record belongs to the declared pilot design.

    The cache is append-only evidence and retains early pilots. Materialization
    is the release boundary: a record from a pair later excluded from the design
    remains auditable in the cache but must not leak into a dataset, and neither
    may a record whose survival check is not one this module knows.
    """
    cats = rec.get("categories") or []
    basis = (rec.get("survival_detail") or {}).get("verification_basis")
    return (basis in ADMITTED_BASES
            and all(multiflaw.compatible(a, b)
                    for a, b in itertools.combinations(cats, 2)))


def _render(item):
    """Run one chained script and write its render; return status and whether a PNG exists."""
    out, code, timeout = item
    res = harness.run_script(code, timeout=timeout)
    ok = res.get("status") == "ok" and bool(res.get("png"))
    if ok:
        out.mkdir(parents=True, exist_ok=True)
        (out / "render.png").write_bytes(res["png"])
    return {"status": res.get("status"), "png": ok}


def verification_method(rec):
    """Name which of the two historical survival checks produced the verdict."""
    per_flaw = (rec.get("survival_detail") or {}).get("per_flaw") or {}
    fields = {key for value in per_flaw.values() if isinstance(value, dict)
              for key in value}
    if "own_paths" in fields:
        return "chain-stages"
    if "solo_paths" in fields:
        return "stored-single-flaw"
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reuse-renders", action="store_true",
                    help="reuse an existing render when its code is byte-identical; "
                         "useful for rebuilding release metadata without 2700 executions")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--workers", type=int, default=8, help="parallel renders")
    args = ap.parse_args()

    records = json.loads(CACHE.read_text())
    all_verified = {k: r for k, r in records.items() if r.get("verdict") == "verified"}
    verified = {k: r for k, r in all_verified.items() if admitted(r)}
    print(f"{len(records)} records, {len(all_verified)} verified, "
          f"{len(all_verified) - len(verified)} excluded by the current pilot design")
    by_arity = Counter(len(r["categories"]) for r in verified.values())
    print("flaws per instance:", dict(by_arity))
    if args.dry_run:
        print("dry run: nothing written")
        return

    # The folder is named after the COMBINATION, not just the number of flaws.
    # Naming it `multi2` alone meant one instance per source chart, so a sample
    # that supports several pairs kept only the first — 1746 of 2736 verified
    # (and paid for) instances had nowhere to live, and the pairs that survived
    # were whichever sorted first, which skewed the coverage from 1 to 74 per
    # combination. The dataset builder maps these folders back to one reported
    # category, so nothing downstream sees 36 categories where it expects one.
    # Render in parallel before the sequential bookkeeping below: at 100 per
    # combination the stock is ~12k scripts, and one process at ~3 s each is
    # ten hours where eight workers need about one.  The worker writes
    # render.png itself and hands back only the status, so nothing large sits
    # in memory; the loop keeps the exact order and dedupe it always had.
    ordered, seen = [], {}
    for key, rec in sorted(verified.items()):
        category = "multi%d__%s" % (len(rec["categories"]),
                                    "+".join(sorted(rec["categories"])))
        if (rec["folder"], category) in seen:
            continue
        seen[(rec["folder"], category)] = key
        ordered.append((key, category))
    to_render = []
    for key, category in ordered:
        rec = verified[key]
        out = SAMPLES / rec["folder"] / "perturbed_llm" / category
        old = (out / "code.py").read_text(encoding="utf-8") if (out / "code.py").exists() else None
        if not (args.reuse_renders and old == rec["perturbed_code"]
                and (out / "render.png").exists()):
            to_render.append((key, out))
    rendered = {}
    print(f"rendering {len(to_render)} of {len(ordered)} instances with {args.workers} workers",
          flush=True)
    harness.warm_cache()
    with Pool(args.workers) as pool:
        for (key, _), res in zip(to_render, pool.imap(
                _render, [(out, verified[key]["perturbed_code"], args.timeout)
                          for key, out in to_render], chunksize=2)):
            rendered[key] = res

    rows, written, failed, seen = [], 0, 0, {}
    for key, rec in sorted(verified.items()):
        folder = rec["folder"]
        category = "multi%d__%s" % (len(rec["categories"]),
                                    "+".join(sorted(rec["categories"])))
        if (folder, category) in seen:
            continue
        seen[(folder, category)] = key
        d = SAMPLES / folder
        out = d / "perturbed_llm" / category
        out.mkdir(parents=True, exist_ok=True)
        gt_code = (d / "gt/code.py").read_text(encoding="utf-8")
        old_code = ((out / "code.py").read_text(encoding="utf-8")
                    if (out / "code.py").exists() else None)
        (out / "code.py").write_text(rec["perturbed_code"], encoding="utf-8")
        (out / "diff.patch").write_text("".join(difflib.unified_diff(
            gt_code.splitlines(keepends=True),
            rec["perturbed_code"].splitlines(keepends=True),
            fromfile="gt/code.py", tofile="code.py")), encoding="utf-8")

        # The render has to exist: the chart modality is a file on disk, and a
        # task whose chart is missing is silently dropped a layer further down.
        reuse = (args.reuse_renders and old_code == rec["perturbed_code"]
                 and (out / "render.png").exists())
        res = {"status": "ok", "png": True} if reuse else rendered[key]
        ok = bool(res.get("status") == "ok" and res.get("png"))
        meta = json.loads((d / "meta.json").read_text())
        row = {
            "instance_id": "%s__%s__llm" % (folder.split("_", 1)[1], category),
            "sample_id": meta["id"], "sample_folder": folder, "category": category,
            "library": meta["library"], "chart_type": meta["chart_type"],
            "description": " + ".join("%s: %s" % (e["category"], e["description"])
                                      for e in rec["edits"])[:600],
            "edits": rec["edits"],                     # the list reviewer asked for
            "flaw_categories": rec["categories"],
            "evidence": {"summary": sorted({n for e in rec["edits"]
                                            for n in (e.get("evidence") or {}).get("summary", [])})},
            "generator": {"type": "llm-chained", "model": rec.get("model")},
            "v1_policy": rec.get("v1_policy", "multiple injected flaws per instance"),
            "survival": rec.get("survival_detail", {}).get("per_flaw", {}),
            "verification_method": verification_method(rec),
            "gates": {
                "exec_ok": ok,
                "exec_status": res.get("status"),
                "solvable": None,
                "survival_witness": rec.get("verdict") == "verified",
                "cross_family_judged": False,
                "solvability_note": (
                    "independent single-flaw footprint witnesses present; "
                    "combined artifact has not passed a cross-family judge"
                ),
            },
            "files": {"gt_code": "../../gt/code.py", "flawed_code": "code.py",
                      "gt_render": "../../gt/render.png", "flawed_render": "render.png",
                      "diff": "diff.patch"},
        }
        (out / "perturbation.json").write_text(json.dumps(row, indent=1, ensure_ascii=False),
                                               encoding="utf-8")
        rows.append(row)
        written += ok
        failed += not ok
        if len(rows) % 25 == 0:
            print(f"  {len(rows)}/{len(verified)}", flush=True)

    MANIFEST.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                        encoding="utf-8")
    coverage = {}
    for arity in sorted(by_arity):
        combo_counts = Counter(
            "+".join(sorted(rec["categories"]))
            for rec in verified.values() if len(rec["categories"]) == arity
        )
        values = list(combo_counts.values())
        coverage[str(arity)] = {
            "admitted_verified": sum(values),
            "represented_combinations": len(values),
            "min_per_combination": min(values, default=0),
            "median_per_combination": statistics.median(values) if values else 0,
            "max_per_combination": max(values, default=0),
            "with_missing_legend": sum(
                n for combo, n in combo_counts.items()
                if "missing-legend" in combo.split("+")
            ),
            "combination_counts": dict(sorted(combo_counts.items())),
        }
    before_path = V0 / "multiflaw.pre_race_reverify.json"
    before = json.loads(before_path.read_text()) if before_path.exists() else {}
    bases = Counter((rec.get("survival_detail") or {}).get("verification_basis")
                    for rec in verified.values())
    report = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).replace(
            microsecond=0).isoformat(),
        "verification_basis": dict(sorted(bases.items(), key=lambda kv: str(kv[0]))),
        "claim": ("a footprint witness for every component (stored single-flaw replay "
                  "for the historical records, per-stage chain check for records since "
                  "01.09.); the cross-family judge of the components runs afterwards "
                  "(endpoint_judge.py --multiflaw) and binds in dataset.py, not here"),
        "cache_records": len(records),
        "old_verified_before_race_remediation": sum(
            rec.get("verdict") == "verified" for rec in before.values()),
        "admitted_verified_after_remediation": len(verified),
        "excluded_by_pair_policy": len(all_verified) - len(verified),
        "materialized": len(rows),
        "render_ok": written,
        "render_failed": failed,
        "coverage": coverage,
    }
    QC_REPORT.parent.mkdir(parents=True, exist_ok=True)
    QC_REPORT.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n{written} instances materialized and rendered, {failed} without a render")
    print(f"-> {MANIFEST}")
    print(f"-> {QC_REPORT}")
    print("Next: .venv/bin/python -m src.eval.dataset --name multi2-v1.2-balanced --n 500 "
          "--categories multi2")


if __name__ == "__main__":
    main()
