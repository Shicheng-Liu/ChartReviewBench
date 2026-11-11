"""Compute the human alignment number from exported annotations.

Reads one or more annotation exports (downloaded from annotate_llm.html /
annotate_judged.html; later files win on overlap), joins them with the LLM
judge verdicts and the recorded rule verdicts, and reports accuracy plus
Cohen's kappa (human vs judge, human vs rule) with confusion matrices --
the human alignment number for the QC pipeline.

'unsure' annotations are excluded from accuracy/kappa but counted. Mind the
prevalence caveat: at a ~95% valid base rate kappa is deflated even when
accuracy is high -- always report accuracy, kappa and the raw confusion
matrix together.

Usage:
    .venv/bin/python src/score_annotations.py --annotations data/v0/annotations_judged.json
    .venv/bin/python src/score_annotations.py --annotations a.json b.json \
        --judge-model claude-sonnet-5
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V0 = ROOT / "data/v0"


def score(pairs):
    """pairs: [(human_valid, other_valid)] -> (accuracy, kappa, confusion)."""
    n = len(pairs)
    if not n:
        return None, None, {}
    a = sum(1 for h, o in pairs if h and o)
    b = sum(1 for h, o in pairs if h and not o)
    c = sum(1 for h, o in pairs if not h and o)
    d = sum(1 for h, o in pairs if not h and not o)
    po = (a + d) / n
    pe = ((a + b) * (a + c) + (c + d) * (b + d)) / n ** 2
    kappa = (po - pe) / (1 - pe) if pe < 1 else None
    return po, kappa, {"both_valid": a, "human_valid_other_invalid": b,
                       "human_invalid_other_valid": c, "both_invalid": d}


def score_v11(annotations):
    """The three error rates of the v1.1 judge gate, per block.

    One block can only measure one direction. The rejected block gives the
    dangerous rate — good instances the gate threw away (reviewer: a false
    positive is the one that hurts, because it is invisible). The clean block
    gives the survivable one — bad instances that slipped into the benchmark.
    The contested block says how much a split vote is worth.
    """
    key = json.loads((V0 / "annotation_key_v11.json").read_text())
    blocks, votes = key["blocks"], key["judge_votes"]
    print(f"dataset {key['dataset']} · {sum(len(v) for v in blocks.values())} instances "
          f"in the sheet, {len(annotations)} annotated\n")

    pairs, rows = [], []
    for name, ids in blocks.items():
        seen = [(i, annotations.get(i, {}).get("verdict")) for i in ids]
        done = [(i, v) for i, v in seen if v in ("valid", "invalid")]
        unsure = sum(1 for _, v in seen if v == "unsure")
        if not done:
            rows.append((name, len(ids), 0, unsure, None, ""))
            continue
        n_valid = sum(1 for _, v in done if v == "valid")
        # what the gate decided for this block, by construction
        gate_valid = name != "rejected"
        pairs += [(v == "valid", gate_valid) for _, v in done]
        if name == "rejected":
            rate = n_valid / len(done)
            note = f"FALSE POSITIVES of the gate: {n_valid}/{len(done)} = {rate:.1%} thrown away wrongly"
        elif name == "clean":
            rate = 1 - n_valid / len(done)
            note = f"FALSE NEGATIVES: {len(done) - n_valid}/{len(done)} = {rate:.1%} slipped through"
        else:
            rate = n_valid / len(done)
            note = f"humans back the majority in {n_valid}/{len(done)} = {rate:.1%} of split votes"
        rows.append((name, len(ids), len(done), unsure, rate, note))

    width = max(len(r[0]) for r in rows)
    for name, n, done, unsure, rate, note in rows:
        print(f"{name:<{width}}  sheet {n:3d}  annotated {done:3d}  unsure {unsure:2d}  {note}")

    acc, kappa, conf = score(pairs)
    if acc is not None:
        print(f"\nHuman vs the judge gate over all annotated blocks (n={len(pairs)}): "
              f"accuracy {acc:.1%}" + (f", Cohen's kappa {kappa:.3f}" if kappa is not None else ""))
        print(f"  {conf}")
        print("  Note: this sample is deliberately NOT the dataset's base rate — the "
              "rejected block is oversampled so the dangerous direction is measurable "
              "at all. Report the per-block rates, not this accuracy, as the headline.")
    fams = [votes[i]["families"] for ids in blocks.values() for i in ids if i in votes]
    if fams:
        print(f"  judge families per instance: min {min(fams)}, median {sorted(fams)[len(fams)//2]}")


def score_gate_annotations(annotations, key):
    """Score a blind accepted/rejected gate sheet without inventing labels.

    ``valid`` means the human would retain the benchmark instance.  Rejected
    items therefore report the good-instance rejection rate; accepted items
    report the bad-instance acceptance rate.  Those names avoid the ambiguous
    false-positive convention that depends on whether "positive" means valid
    or rejected.
    """
    decisions = key.get("gate_decisions") or {}
    rows, all_pairs = {}, []
    for block, ids in key["blocks"].items():
        pairs, unsure, missing = [], 0, 0
        for iid in ids:
            verdict = (annotations.get(iid) or {}).get("verdict")
            if verdict == "unsure":
                unsure += 1
                continue
            if verdict not in ("valid", "invalid"):
                missing += 1
                continue
            gate_valid = decisions.get(iid, block != "rejected")
            pairs.append((verdict == "valid", bool(gate_valid)))
        accuracy, kappa, confusion = score(pairs)
        n_human_valid = sum(human for human, _ in pairs)
        gate_valid = all(decisions.get(iid, block != "rejected") for iid in ids)
        rows[block] = {
            "sheet_n": len(ids),
            "n": len(pairs),
            "unsure": unsure,
            "missing": missing,
            "gate_decision": "accept" if gate_valid else "reject",
            "human_valid": n_human_valid,
            "human_valid_rate": n_human_valid / len(pairs) if pairs else None,
            "accuracy": accuracy,
            "kappa": kappa,
            "confusion": confusion,
            "good_instance_rejection_rate": (
                n_human_valid / len(pairs) if pairs and not gate_valid else None),
            "bad_instance_acceptance_rate": (
                (len(pairs) - n_human_valid) / len(pairs)
                if pairs and gate_valid else None),
        }
        all_pairs.extend(pairs)

    rejection_strata = defaultdict(list)
    for iid, reasons in (key.get("rejection_reasons") or {}).items():
        label = "+".join(reasons)
        rejection_strata[label].append(iid)
    reason_rows = {}
    for label, ids in sorted(rejection_strata.items()):
        decided = [(annotations.get(iid) or {}).get("verdict") for iid in ids]
        valid = sum(value == "valid" for value in decided)
        n = sum(value in ("valid", "invalid") for value in decided)
        reason_rows[label] = {
            "sheet_n": len(ids),
            "n": n,
            "unsure": sum(value == "unsure" for value in decided),
            "missing": sum(value not in ("valid", "invalid", "unsure")
                           for value in decided),
            "human_valid": valid,
            "good_instance_rejection_rate": valid / n if n else None,
        }

    accuracy, kappa, confusion = score(all_pairs)
    return {
        "schema": 2,
        "dataset": key.get("dataset"),
        "panel": key.get("panel"),
        "representative": bool(key.get("representative")),
        "design": key.get("design"),
        "n_sheet": sum(len(ids) for ids in key["blocks"].values()),
        "n_with_any_annotation": sum(
            iid in annotations for ids in key["blocks"].values() for iid in ids),
        "overall": {"n": len(all_pairs), "accuracy": accuracy, "kappa": kappa,
                    "confusion": confusion},
        "blocks": rows,
        "rejected_by_reason": reason_rows,
        "unknown_annotation_ids": sorted(
            set(annotations) - {iid for ids in key["blocks"].values() for iid in ids}),
    }


def print_gate_report(report):
    print(f"dataset {report['dataset']} · {report['n_with_any_annotation']}/"
          f"{report['n_sheet']} rows touched")
    print("DESIGN CAVEAT: the rejected block is deliberately stratified and the "
          "accepted side is the fixed panel. Report per-block/stratum rates; "
          "pooled agreement is descriptive only.\n")
    for name, row in report["blocks"].items():
        detail = (f"good-instance rejection {row['good_instance_rejection_rate']:.1%}"
                  if row["good_instance_rejection_rate"] is not None else
                  f"bad-instance acceptance {row['bad_instance_acceptance_rate']:.1%}"
                  if row["bad_instance_acceptance_rate"] is not None else "no decisions")
        print(f"{name:<10} gate={row['gate_decision']:<6} sheet={row['sheet_n']:3d} "
              f"decided={row['n']:3d} unsure={row['unsure']:2d} missing={row['missing']:3d} "
              f"{detail}")
    overall = report["overall"]
    if overall["accuracy"] is not None:
        kappa = (f" · kappa {overall['kappa']:.3f}"
                 if overall["kappa"] is not None else "")
        print(f"\npooled human↔gate n={overall['n']}: accuracy "
              f"{overall['accuracy']:.1%}{kappa} · {overall['confusion']}")
    if report["rejected_by_reason"]:
        print("\nRejected block by hidden reason:")
        for reason, row in report["rejected_by_reason"].items():
            rate = row["good_instance_rejection_rate"]
            shown = "n/a" if rate is None else f"{rate:.1%}"
            print(f"  {reason:<40} decided={row['n']:2d}/{row['sheet_n']:2d} "
                  f"good-instance rejection={shown}")


def score_v12(annotations):
    return _score_frozen(annotations, "v12")


def score_v13(annotations):
    return _score_frozen(annotations, "v13")


def _score_frozen(annotations, tag):
    key = json.loads((V0 / f"annotation_key_{tag}.json").read_text(encoding="utf-8"))
    report = score_gate_annotations(annotations, key)
    out = V0 / f"annotation_scores_{tag}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print_gate_report(report)
    print(f"-> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", nargs="+", required=True,
                    help="exported annotation JSON file(s); later files win")
    ap.add_argument("--judge-model", default="claude-opus-4-8")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--v11", action="store_true",
                      help="score against annotation_key_v11.json: per-block error rates "
                           "of the cross-family judge gate")
    mode.add_argument("--v12", action="store_true",
                      help="score against annotation_key_v12.json: the frozen panel plus "
                           "stratified composite-gate rejections")
    mode.add_argument("--v13", action="store_true",
                      help="score against annotation_key_v13.json (same design, v1.3 freeze)")
    args = ap.parse_args()

    if args.v11:
        merged = {}
        for path in args.annotations:
            merged.update(json.loads(Path(path).read_text()))
        return score_v11(merged)
    if args.v12 or args.v13:
        merged = {}
        for path in args.annotations:
            merged.update(json.loads(Path(path).read_text()))
        return (score_v13 if args.v13 else score_v12)(merged)

    annotations = {}
    for path in args.annotations:
        annotations.update(json.loads(Path(path).read_text()))

    manifest = {m["instance_id"]: m for m in
                (json.loads(l) for l in (V0 / "manifest_llm.jsonl").read_text().splitlines())}
    judge = {r["instance_id"]: r for r in json.loads((V0 / "judge_results.json").read_text())
             if r.get("judge_model", "claude-sonnet-5") == args.judge_model
             and "error" not in r}

    decided, unsure = {}, []
    for iid, rec in annotations.items():
        if rec.get("verdict") == "unsure":
            unsure.append(iid)
        elif rec.get("verdict") in ("valid", "invalid"):
            decided[iid] = rec["verdict"] == "valid"

    vs_judge, vs_rule, disagreements = [], [], []
    per_cat = defaultdict(Counter)
    for iid, human in decided.items():
        m = manifest.get(iid)
        if m is None:
            continue
        cat = m["category"]
        per_cat[cat]["n"] += 1
        per_cat[cat]["human_valid"] += human
        rule = m["gates"]["solvable"]
        vs_rule.append((human, rule))
        j = judge.get(iid)
        if j is not None:
            vs_judge.append((human, j["llm_detectable"]))
            per_cat[cat]["judge_agree"] += human == j["llm_detectable"]
            if human != j["llm_detectable"]:
                disagreements.append({
                    "instance_id": iid, "category": cat,
                    "human": "valid" if human else "invalid",
                    "judge": "valid" if j["llm_detectable"] else "invalid",
                    "rule": "valid" if rule else "invalid",
                    "note": annotations[iid].get("note", ""),
                    "judge_reasoning": j.get("reasoning", "")})

    acc_j, kap_j, conf_j = score(vs_judge)
    acc_r, kap_r, conf_r = score(vs_rule)
    out = {
        "judge_model": args.judge_model,
        "n_annotated": len(decided) + len(unsure),
        "n_decided": len(decided),
        "n_unsure": len(unsure),
        "human_valid_rate": (sum(1 for v in decided.values() if v) / len(decided)
                             if decided else None),
        "human_vs_judge": {"n": len(vs_judge), "accuracy": acc_j, "kappa": kap_j,
                           "confusion": conf_j},
        "human_vs_rule": {"n": len(vs_rule), "accuracy": acc_r, "kappa": kap_r,
                          "confusion": conf_r},
        "per_category": {c: dict(cnt) for c, cnt in sorted(per_cat.items())},
        "disagreements": disagreements,
    }
    (V0 / "annotation_scores.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    fmt = lambda x: "n/a" if x is None else f"{x:.3f}"
    print(f"{out['n_annotated']} annotiert ({len(decided)} entschieden, "
          f"{len(unsure)} unsure)")
    print(f"Human↔Judge ({args.judge_model}, n={len(vs_judge)}): "
          f"Accuracy {fmt(acc_j)} · Kappa {fmt(kap_j)} · {conf_j}")
    print(f"Human↔Regel (n={len(vs_rule)}): "
          f"Accuracy {fmt(acc_r)} · Kappa {fmt(kap_r)} · {conf_r}")
    print("Caveat: bei ~95% valider Basisrate ist Kappa auch bei hoher Accuracy "
          "gedrückt -- immer beide Zahlen plus Konfusionsmatrix berichten.")
    if disagreements:
        print(f"\n{len(disagreements)} Disagreements Human↔Judge:")
        for d in disagreements:
            print(f"  {d['instance_id']} ({d['category']}): human={d['human']} "
                  f"judge={d['judge']} rule={d['rule']}")
    print(f"-> {V0 / 'annotation_scores.json'}")


if __name__ == "__main__":
    main()
