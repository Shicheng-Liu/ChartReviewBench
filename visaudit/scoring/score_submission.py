"""Score one repaired script against its instance.

    python score_submission.py --instance instances/<id> --submission repaired.py --category wrong-title
    python score_submission.py --instance instances/<id> --submission chart.py --clean

Prints the metric vector as JSON. Recovery counts only the figure-object paths
the injected flaws touched, preservation only the paths outside that footprint,
and data fidelity compares the plotted values against table.csv. A value that is
undefined for an instance stays undefined and is never counted as zero.
For a Track C clean twin, --clean uses ground_truth.py as the initial chart:
recovery is undefined and every original property belongs to preservation.

Run every instance with the pinned environment in env-lock.txt; a different
matplotlib version moves autoscale floats and changes preservation.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chartrepair_eval.capture import capture_path
from chartrepair_eval.compare import TableIndex
from chartrepair_eval.metrics import score


CATEGORIES = {
    "axis-truncation", "color-contradicts-summary", "dropped-series",
    "missing-legend", "swapped-legend-labels", "swapped-series",
    "value-perturbation", "wrong-axis-label", "wrong-chart-type", "wrong-title",
}


def category_argument(value: str) -> str:
    prefix, _, combination = value.partition("__")
    parts = (combination if combination else value).split("+")
    if (not 1 <= len(parts) <= 3 or len(set(parts)) != len(parts)
            or not set(parts) <= CATEGORIES
            or (combination and prefix != f"multi{len(parts)}")):
        raise argparse.ArgumentTypeError("use a category or combination from dataset.json")
    # The metric accepts the internal multiN__a+b form; the release stores a+b.
    return parts[0] if len(parts) == 1 else f"multi{len(parts)}__{'+'.join(parts)}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--instance", type=Path, required=True,
                    help="instances/<id> from this package")
    ap.add_argument("--submission", type=Path, required=True,
                    help="the script the agent handed back")
    ap.add_argument("--category", type=category_argument,
                    help="the flaw category; for a multi-flaw instance the "
                         "combination as it stands in dataset.json. Required "
                         "unless --clean; with --clean it only identifies the paired task")
    ap.add_argument("--clean", action="store_true",
                    help="score a Track C clean twin against ground_truth.py as "
                         "its initial chart, without flaw-specific preservation exceptions")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--isolate", action="store_true",
                    help="require the macOS OS isolation used by the evaluation runner; "
                         "fails if this backend is unavailable")
    args = ap.parse_args()
    if not args.clean and args.category is None:
        ap.error("--category is required unless --clean is used")

    inst = args.instance
    table_text = (inst / "table.csv").read_text(encoding="utf-8", errors="replace")
    summary_text = (inst / "summary.txt").read_text(encoding="utf-8", errors="replace")
    extra = {"table.csv": table_text, "summary.txt": summary_text}
    gt = capture_path(inst / "ground_truth.py", extra, args.timeout, isolate=args.isolate)
    flawed = gt if args.clean else capture_path(inst / "flawed.py", extra, args.timeout,
                                               isolate=args.isolate)
    submitted = capture_path(args.submission, extra, args.timeout, isolate=args.isolate)
    if gt["status"] != "ok" or flawed["status"] != "ok":
        print(json.dumps({"error": "reference capture failed",
                          "ground_truth": gt["status"], "flawed": flawed["status"]}))
        return 2
    result = score(gt, flawed, submitted, TableIndex(table_text),
                   "" if args.clean else args.category, summary_text=summary_text)
    print(json.dumps(result, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
