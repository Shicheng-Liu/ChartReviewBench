"""Offline clean-chart recognition analysis; never modifies source runs.

Final script equality is an observable proxy, not a recognition annotation:
an agent can modify output.png through execute_python without editing chart.py.
Explicit CCR therefore requires reviewed, source-bound initial-chart verdicts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

MODELS = (
    "google-gemma-4-31b-it-a100", "qwen3-vl-235b-a22b-instruct",
    "ministral-3-8b-2512", "mistral-large-3-675b-2512-fp4",
    "granite-vision-4-1-4b",
)
SCHEMA = "clean-chart-recognition-offline-v2"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rate(n: int, denominator: int) -> float | None:
    return n / denominator if denominator else None


def strict_ccr_outcome(row: dict) -> bool | None:
    """Recognition without unnecessary repair, with unknowns left unresolved.

    A changed final script or involuntary termination fails the explicit Track C
    instruction. For an unchanged voluntary submission, a reviewed correct
    initial-chart verdict AND reviewed absence of unnecessary repair are needed.
    Pixel differences alone do not label semantic repairs.
    """
    if not row["completed"]:
        return None
    if row["no_final_code_edit"] is False or row["voluntary_finish"] is False:
        return False
    if row["explicit_initial_verdict"] == "incorrect" or row["unnecessary_repair_annotated"] is True:
        return False
    if (row["no_final_code_edit"] is True and row["voluntary_finish"] is True
            and row["explicit_initial_verdict"] == "correct"
            and row["unnecessary_repair_annotated"] is False):
        return True
    return None


def apply_annotation(row: dict, annotation: dict) -> None:
    if not row["completed"]:
        raise ValueError("do not label an incomplete attempt as a completed recognition outcome")
    if annotation.get("initial_chart_verdict") not in {"correct", "incorrect", "undetermined"}:
        raise ValueError("annotation needs an initial-chart verdict")
    if not annotation.get("reviewer") or not annotation.get("evidence"):
        raise ValueError("annotation needs reviewer and evidence")
    repair = annotation.get("unnecessary_repair")
    if repair is not None and type(repair) is not bool:
        raise ValueError("unnecessary_repair must be true, false, or null")
    row["explicit_initial_verdict"] = annotation["initial_chart_verdict"]
    row["unnecessary_repair_annotated"] = repair


def analyze_episode(episode: dict, trajectory: list[dict] | None = None,
                    annotation: dict | None = None) -> dict:
    """Extract behavior independently from an optional reviewed semantic verdict.

    An annotation must concern the *initial* chart, not whether a repaired final
    chart is correct. No-edit behavior or preservation never supplies this label.
    The CLI additionally binds annotations to the episode and trajectory hashes.
    """
    if episode.get("clean") is not True or episode.get("track") != "C":
        raise ValueError("CCR only applies to initially correct Track C charts")
    complete = episode.get("status") == "ok"
    edited = episode.get("edited") if complete else None
    if edited is not True and edited is not False:
        edited = None
    origin = episode.get("finish_origin")
    voluntary = (origin == "agent" and episode.get("stop_reason") == "finished")
    origin_known = bool(origin) and bool(episode.get("stop_reason"))
    no_edit = None if edited is None else not edited
    proxy = (no_edit and voluntary) if complete and edited is not None and origin_known else None
    result = {
        "task_id": episode["task_id"], "instance": episode.get("instance"),
        "status": episode.get("status"), "completed": complete,
        "stop_reason": episode.get("stop_reason"), "finish_origin": origin,
        "executability": episode.get("executability"), "produced_png": episode.get("produced_png"),
        "no_final_code_edit": no_edit, "voluntary_finish": voluntary if origin_known else None,
        "ccr_behavior_proxy": proxy, "explicit_initial_verdict": None,
        "unnecessary_repair_annotated": None,
        "preservation_strict": episode.get("metrics", {}).get("preservation", {}).get("strict"),
        "preservation_binary_strict": episode.get("metrics", {}).get("preservation_binary", {}).get("strict"),
        "trajectory_complete": False, "observed_chart_write": None,
        "finish_message": "",
    }
    if trajectory is not None:
        result["trajectory_complete"] = (
            len(trajectory) == episode.get("steps") and
            [r.get("step") for r in trajectory] == list(range(len(trajectory)))
        )
        writes = False
        for call in trajectory:
            observation, args = call.get("observation") or {}, call.get("args") or {}
            if call.get("tool") == "finish" and observation.get("finished") is True:
                result["finish_message"] = args.get("message", "")
            changed = observation.get("files_changed") or []
            changed_chart = any(str(x).removeprefix("./") == "chart.py" for x in changed)
            direct_write = (call.get("tool") in {"write_file", "edit_file", "patch_file"}
                            and str(args.get("path", "")).removeprefix("./") == "chart.py"
                            and observation.get("ok") is True)
            writes |= changed_chart or direct_write
        result["observed_chart_write"] = writes if result["trajectory_complete"] else (True if writes else None)
    if annotation is not None:
        apply_annotation(result, annotation)
    result["ccr_success"] = strict_ccr_outcome(result)
    return result


def summarize(rows: list[dict]) -> dict:
    complete = [r for r in rows if r["completed"]]
    known_edits = [r for r in complete if r["no_final_code_edit"] is not None]
    known_proxy = [r for r in complete if r["ccr_behavior_proxy"] is not None]
    no_edit = sum(r["no_final_code_edit"] is True for r in known_edits)
    proxy = sum(r["ccr_behavior_proxy"] is True for r in known_proxy)
    recognized = sum(r["explicit_initial_verdict"] == "correct" for r in complete)
    unrecognized = sum(r["explicit_initial_verdict"] == "incorrect" for r in complete)
    judged = recognized + unrecognized
    ccr_success = sum(strict_ccr_outcome(r) is True for r in complete)
    ccr_failure = sum(strict_ccr_outcome(r) is False for r in complete)
    ccr_unknown = len(complete) - ccr_success - ccr_failure
    ccr_unknown_total = len(rows) - ccr_success - ccr_failure
    return {
        "target_clean_n": len(rows), "completed_clean_n": len(complete),
        "completion_coverage": rate(len(complete), len(rows)),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "no_final_code_edit_n": no_edit, "final_code_edit_known_n": len(known_edits),
        "no_final_code_edit_rate": rate(no_edit, len(known_edits)),
        "ccr_behavior_proxy_n": proxy, "ccr_behavior_proxy_known_n": len(known_proxy),
        "ccr_behavior_proxy_rate": rate(proxy, len(known_proxy)),
        "ccr_behavior_proxy_unknown_completed_n": len(complete) - len(known_proxy),
        "no_edit_without_voluntary_finish_n": sum(r["no_final_code_edit"] is True and r["voluntary_finish"] is False for r in complete),
        "no_final_edit_with_observed_chart_write_n": sum(r["no_final_code_edit"] is True and r["observed_chart_write"] is True for r in complete),
        "trajectory_complete_n": sum(r["trajectory_complete"] for r in complete),
        "nonempty_finish_message_n": sum(bool(r["finish_message"]) for r in complete),
        "recognition_correct_n": recognized, "recognition_incorrect_n": unrecognized,
        "recognition_unknown_completed_n": len(complete) - judged,
        "recognition_annotation_coverage": rate(judged, len(complete)),
        "ccr_explicit_rate": rate(recognized, len(complete)) if judged == len(complete) else None,
        "ccr_explicit_rate_known_only": rate(recognized, judged),
        "ccr_success_n": ccr_success, "ccr_failure_n": ccr_failure,
        "ccr_unknown_completed_n": ccr_unknown,
        "ccr_unknown_total_n": ccr_unknown_total,
        "ccr_known_n": ccr_success + ccr_failure,
        "ccr_label_coverage": rate(ccr_success + ccr_failure, len(rows)),
        "ccr_rate": rate(ccr_success, len(rows)) if not ccr_unknown_total else None,
        "ccr_confirmed_lower_bound": rate(ccr_success, len(rows)),
        "ccr_possible_upper_bound": rate(ccr_success + ccr_unknown_total, len(rows)),
        "ccr_completed_only_rate": rate(ccr_success, len(complete)) if not ccr_unknown else None,
        "perfect_preservation_n": sum(r["preservation_binary_strict"] == 1 for r in complete),
        "preservation_known_n": sum(r["preservation_binary_strict"] is not None for r in complete),
        "proxy_with_imperfect_preservation_n": sum(r["ccr_behavior_proxy"] is True and r["preservation_binary_strict"] == 0 for r in complete),
    }


def annotate_snapshot(snapshot: dict, annotations: dict) -> dict:
    """Apply source-bound reviewed labels to frozen rows, without reading live runs."""
    report = json.loads(json.dumps(snapshot))
    used = set()
    for row in report["rows"]:
        label = annotations.get(row["model"], {}).get(row["instance"])
        if label is not None:
            for field in ("episode_sha256", "trajectory_sha256"):
                if not label.get(field) or label[field] != row.get(field):
                    raise ValueError("annotation does not match frozen " + field)
            apply_annotation(row, label)
            used.add((row["model"], row["instance"]))
        row["ccr_success"] = strict_ccr_outcome(row)
    if used != {(model, instance) for model, labels in annotations.items() for instance in labels}:
        raise ValueError("annotation contains an unknown model or instance")
    report["summaries"] = {model: summarize([r for r in report["rows"] if r["model"] == model])
                           for model in report["summaries"]}
    report["annotation_applied_at"] = datetime.now(timezone.utc).isoformat()
    report["schema"] = SCHEMA
    report.setdefault("definitions", {})["strict_ccr"] = (
        "Self-finish with final source unchanged, a reviewed declaration that the initial chart is correct, "
        "and reviewed absence of unnecessary intervention. Outcome is unknown if required evidence is "
        "unresolved and no failure condition is established. Rates and bounds use every target clean chart."
    )
    return report


def normalize_annotations(snapshot: dict, labels: list[dict]) -> dict:
    """Bind a compact assistant-review export to the frozen evidence population.

    The returned source hashes identify the exact evidence reviewed. A supplied
    mismatching hash is rejected; no current live run is consulted.
    """
    indexed = {(r["model"], r["task_id"]): r for r in snapshot["rows"]}
    normalized = {}
    for label in labels:
        key = label["model"], label["task_id"]
        if key not in indexed:
            raise ValueError("annotation contains an unknown model or task")
        row = indexed[key]
        recognition = label.get("recognition")
        if recognition is not None and type(recognition) is not bool:
            raise ValueError("recognition must be true, false, or null")
        target = normalized.setdefault(key[0], {})
        if row["instance"] in target:
            raise ValueError("duplicate annotation")
        target[row["instance"]] = {
            "initial_chart_verdict": "undetermined" if recognition is None else "correct" if recognition else "incorrect",
            "unnecessary_repair": label.get("unnecessary_intervention"),
            "reviewer": label.get("reviewer"), "evidence": label.get("evidence_short"),
        }
        for field in ("episode_sha256", "trajectory_sha256"):
            if label.get(field) and label[field] != row.get(field):
                raise ValueError("annotation does not match frozen " + field)
            target[row["instance"]][field] = row.get(field)
    return normalized


def analyze_campaign(campaign: Path, annotations: dict | None = None,
                     recovery_receipt: Path | None = None) -> dict:
    annotations = annotations or {}
    sources, reports, all_rows = {}, {}, []

    def read(path: Path) -> bytes:
        content = path.read_bytes()
        sources[str(path)] = digest(content)
        return content

    population = json.loads(read(campaign / "runtime/C.json"))
    clean = {i["instance"] for i in population["items"] if i.get("clean") is True}
    if len(clean) != sum(i.get("clean") is True for i in population["items"]):
        raise ValueError("duplicate clean population entries")
    overlay = None
    if recovery_receipt:
        receipt = json.loads(read(recovery_receipt))
        old, new = Path(receipt["original_error"]), Path(receipt["new_episode"])
        old_bytes, new_bytes = read(old), read(new)
        if digest(old_bytes) != receipt["original_error_sha256"] or digest(new_bytes) != receipt["new_episode_sha256"]:
            raise ValueError("recovery receipt hash mismatch")
        overlay = json.loads(new_bytes)
        if overlay["status"] != "ok":
            raise ValueError("recovery must be a regular result")
    for model in MODELS:
        run = campaign / "runs" / model / "C"
        episodes = {}
        for path in sorted((run / "episodes").glob("*clean*.json")):
            episode = json.loads(read(path))
            if episode.get("instance") not in clean or episode.get("model") != model:
                raise ValueError("unexpected clean result or model")
            if episode["instance"] in episodes:
                raise ValueError("duplicate clean result")
            episodes[episode["instance"]] = (episode, path)
        if overlay and overlay.get("model") == model and overlay.get("clean") is True:
            episodes[overlay["instance"]] = (overlay, Path(receipt["new_episode"]))
        rows = []
        for instance in sorted(clean):
            episode, path = episodes.get(instance, ({"task_id": instance + "__clean__t-Cclean__s-none__w-none",
                "instance": instance, "track": "C", "clean": True, "status": "missing"}, None))
            trajectory, trajectory_path = None, None
            if episode.get("status") == "ok":
                trajectory_path = path.parent.parent / "runs" / episode["run_slug"] / "trajectory.jsonl"
                if trajectory_path.exists():
                    trajectory = [json.loads(line) for line in read(trajectory_path).splitlines() if line.strip()]
            annotation = annotations.get(model, {}).get(instance)
            if annotation:
                if not path or annotation.get("episode_sha256") != sources[str(path)] or not trajectory_path or annotation.get("trajectory_sha256") != sources.get(str(trajectory_path)):
                    raise ValueError("annotation does not match episode and trajectory hashes")
            row = analyze_episode(episode, trajectory, annotation)
            row.update(model=model, episode_path=str(path) if path else None,
                       trajectory_path=str(trajectory_path) if trajectory_path else None,
                       episode_sha256=sources.get(str(path)), trajectory_sha256=sources.get(str(trajectory_path)),
                       final_code_comparison_verified=None)
            if episode.get("status") == "ok":
                initial = campaign / "tasks/C" / episode["task_id"] / "workspace/chart.py"
                final = path.parent.parent / "runs" / episode["run_slug"] / "workspace/chart.py"
                if initial.is_file() and final.is_file() and not initial.is_symlink() and not final.is_symlink():
                    equals = read(initial) == read(final)
                    if row["no_final_code_edit"] is not None and row["no_final_code_edit"] != equals:
                        raise ValueError("stored edited flag disagrees with saved scripts")
                    row["final_code_comparison_verified"] = True
            rows.append(row)
        reports[model] = summarize(rows)
        all_rows.extend(rows)
    return {
        "schema": SCHEMA, "source_of_truth": "SOURCE_OF_TRUTH.md",
        "generated_at": datetime.now(timezone.utc).isoformat(), "campaign": str(campaign),
        "definitions": {
            "ccr_explicit": "Reviewed verdict about the initially correct chart; correct/(correct+incorrect), with unknowns and coverage; full completed-set rate withheld until all completed cases have determinate labels.",
            "ccr_behavior_proxy": "status=ok AND final chart.py byte-identical to input AND finish_origin=agent AND stop_reason=finished. Proxy only, not evidence of semantic recognition or absence of intermediate repairs.",
            "denominator": "Completed clean episodes with the required fields, separately from the fixed target clean population. Error, deferred and missing cases remain uncovered, not recognition failures.",
            "preservation": "Separate submitted-artifact score; neither preservation nor final script equality labels recognition.",
        },
        "recovery_overlay": {"receipt": str(recovery_receipt), "task_id": overlay.get("task_id"),
            "clean": overlay.get("clean"), "effect_on_clean_population": bool(overlay.get("clean"))} if overlay else None,
        "summaries": reports, "rows": all_rows, "source_sha256": sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--recovery-receipt", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("choose a new output directory; snapshots are immutable")
    annotations = json.loads(args.annotations.read_text()) if args.annotations else None
    report = analyze_campaign(args.campaign, annotations, args.recovery_receipt)
    report["analysis_source_sha256"] = digest(Path(__file__).read_bytes())
    if args.annotations:
        report["annotations_sha256"] = digest(args.annotations.read_bytes())
    args.out.mkdir(parents=True)
    (args.out / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, rows in (("cases.csv", report["rows"]), ("summary.csv", [{"model": m, **v} for m, v in report["summaries"].items()])):
        with (args.out / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(report["summaries"], indent=2))


if __name__ == "__main__":
    main()
