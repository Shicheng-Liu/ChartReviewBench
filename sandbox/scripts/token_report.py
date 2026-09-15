"""Aggregate the token cost of a run, and extrapolate it to a full track.

Every episode records what it spent (`token_usage` in `result.json`), split by
role. This rolls those up and projects them, which is the number you actually need
before committing to a suite: a pilot of ten episodes is only useful if it tells
you what nineteen hundred will cost.

The projection is deliberately reported as a range rather than a point. Episode
cost is driven by how many iterations an agent takes, and that varies several-fold
across tasks — quoting the mean alone hides the tail that decides whether a run
fits a budget.

    python scripts/token_report.py <run_dir>
    python scripts/token_report.py <run_dir> --project 1900 --price-in 0.28 --price-out 0.42
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROLES = ("agent", "judge")


def load(run_dir: Path) -> list[dict]:
    rows = []
    for d in sorted(run_dir.iterdir()):
        f = d / "result.json"
        if not f.is_file():
            continue
        r = json.loads(f.read_text())
        tu = r.get("token_usage")
        if not tu:
            continue
        rows.append({
            "id": d.name,
            "score": r.get("final_score"),
            "iterations": (r.get("agent_stats") or {}).get("turns_taken"),
            "steps": r.get("steps_taken"),
            **{f"{role}_{k}": (tu.get(role) or {}).get(k, 0)
               for role in ROLES
               for k in ("calls", "input_tokens", "output_tokens", "reasoning_tokens",
                         "cache_read_input_tokens")},
            "total_tokens": (tu.get("total") or {}).get("total_tokens", 0),
            "total_in": (tu.get("total") or {}).get("input_tokens", 0),
            "total_out": (tu.get("total") or {}).get("output_tokens", 0),
        })
    return rows


def money(tokens_in: int, tokens_out: int, price_in: float | None,
          price_out: float | None) -> float | None:
    if price_in is None or price_out is None:
        return None
    return (tokens_in * price_in + tokens_out * price_out) / 1_000_000


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--project", type=int, default=1900,
                    help="episodes in the full track to extrapolate to (default: 1900)")
    ap.add_argument("--price-in", type=float, help="USD per million input tokens")
    ap.add_argument("--price-out", type=float, help="USD per million output tokens")
    ap.add_argument("--json", action="store_true", help="emit the rows as JSON")
    args = ap.parse_args()

    rows = load(args.run_dir)
    if not rows:
        raise SystemExit(f"no episodes with token_usage under {args.run_dir}")
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    print(f"{'task':40} {'iters':>5} {'agent in':>9} {'agent out':>9} "
          f"{'judge in':>9} {'judge out':>9} {'total':>9}")
    print("-" * 96)
    for r in rows:
        print(f"{r['id'][:39]:40} {str(r['iterations'] or '-'):>5} "
              f"{r['agent_input_tokens']:>9,} {r['agent_output_tokens']:>9,} "
              f"{r['judge_input_tokens']:>9,} {r['judge_output_tokens']:>9,} "
              f"{r['total_tokens']:>9,}")

    n = len(rows)
    tot_in = sum(r["total_in"] for r in rows)
    tot_out = sum(r["total_out"] for r in rows)
    per = [r["total_tokens"] for r in rows]
    reasoning = sum(r["agent_reasoning_tokens"] + r["judge_reasoning_tokens"] for r in rows)
    cached = sum(r["agent_cache_read_input_tokens"] + r["judge_cache_read_input_tokens"]
                 for r in rows)

    print("-" * 96)
    print(f"{n} episodes: {tot_in + tot_out:,} tokens ({tot_in:,} in, {tot_out:,} out)")
    if reasoning:
        print(f"  of the output, {reasoning:,} are reasoning tokens "
              f"({reasoning / tot_out:.0%}) — billed as output")
    if cached:
        print(f"  of the input, {cached:,} were cache hits ({cached / tot_in:.0%}) "
              f"— usually billed well below the base rate")
    print(f"  per episode: mean {statistics.mean(per):,.0f}, median "
          f"{statistics.median(per):,.0f}, min {min(per):,}, max {max(per):,}")

    lo, hi = min(per), max(per)
    mean = statistics.mean(per)
    print(f"\nprojected to {args.project:,} episodes:")
    print(f"  tokens: {mean * args.project:,.0f} at the mean "
          f"(range {lo * args.project:,} – {hi * args.project:,})")
    cost = money(tot_in, tot_out, args.price_in, args.price_out)
    if cost is not None:
        scale = args.project / n
        print(f"  cost:   ${cost * scale:,.2f} at the mean "
              f"(${cost / n * scale:,.4f} per episode)")
    else:
        print("  cost:   pass --price-in and --price-out (USD per million tokens) "
              "to price it; no list price for these models is on file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
