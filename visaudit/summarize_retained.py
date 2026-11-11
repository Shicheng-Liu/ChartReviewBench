"""Reproduce local five-model means from retained measured outcomes; no rescoring."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parent
METRICS = ("recovery_binary", "recovery", "preservation_binary", "preservation")


def defined_mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None, len(values)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--include-historical-single-response", action="store_true")
    args = ap.parse_args()
    conditions = json.loads((ROOT / "history/conditions.json").read_text())["conditions"]
    fields = ["model", "condition", "group", "tasks", "executability", "steps", "input_tokens", "output_tokens"]
    for metric in METRICS:
        fields.extend([metric + "_strict", metric + "_defined_n"])
    writer = csv.DictWriter(sys.stdout, fieldnames=fields)
    writer.writeheader()
    for c in conditions:
        if not c["current_paper_scope"] and not args.include_historical_single_response:
            continue
        p = ROOT / c["outcomes_path"]
        assert hashlib.sha256(p.read_bytes()).hexdigest() == c["outcomes_sha256"]
        episodes = [json.loads(line) for line in p.read_text().splitlines()]
        for clean in (False, True):
            rows = [r for r in episodes if r["clean"] == clean]
            if not rows:
                continue
            row = {"model": c["model"], "condition": c["condition"],
                   "group": "clean" if clean else "flawed", "tasks": len(rows)}
            for name in ("executability", "steps", "output_tokens"):
                row[name] = defined_mean(r.get(name) for r in rows)[0]
            row["input_tokens"] = defined_mean(r.get("total_input_tokens", r.get("input_tokens")) for r in rows)[0]
            for metric in METRICS:
                value, n = defined_mean(r["metrics"][metric]["strict"] for r in rows)
                row[metric + "_strict"], row[metric + "_defined_n"] = value, n
            writer.writerow(row)


if __name__ == "__main__":
    main()
