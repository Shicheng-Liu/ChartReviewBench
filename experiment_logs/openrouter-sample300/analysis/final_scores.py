"""Assemble the six evaluation dimensions into one table, and optionally one number.

Each dimension comes from whichever mechanism can actually measure it, which is not
the same mechanism for all six:

    1 Code Validity          rule    execution verifier: the script ran and wrote a chart
    2 Data Fidelity          rule    release scorer: plotted values vs table.csv
    3 Perturbation Recovery  both    release scorer (recovery) + the repair judge
    4 Preservation           rule    release scorer: what was right is still right
    5 Visual Quality         judge   readability and presentation; no rule-based substitute
    6 Efficiency             rule    iterations, executions and tokens spent

The table also reports one derived joint-success diagnostic requested for analysis:

    Joint Success = executability == 1 AND recovery_binary == 1
                    AND preservation_binary == 1

This is not a seventh independent dimension and is never included in the weighted
score.  It is undefined when any of its three inputs is undefined (notably recovery
on clean Track C tasks), rather than treating an unmeasured input as a failure.

Dimensions 2 and 4 require `scripts/score_runs.py` to have merged `rule_based` into
the results; without it they are reported as missing rather than as zero, because a
dimension nobody measured is not a dimension the agent failed.

Weights are off by default. The release's own scorer reports its metrics "as a vector,
never aggregated", and that caution is worth keeping: a single number here is a claim
about how much a legible chart is worth relative to a correct one, and that claim
should be made deliberately, in the open, not inherited from a default.

    python3 scripts/final_scores.py <run_dir>
    python3 scripts/final_scores.py <run_dir> --weights 0.1,0.25,0.25,0.2,0.15,0.05
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

NAMES = ["code_validity", "data_fidelity", "perturbation_recovery",
         "preservation", "visual_quality", "efficiency"]
JOINT_NAME = "joint_success"


def efficiency(result: dict) -> float | None:
    """1.0 for a one-iteration repair, decaying towards 0 as the budget is used up.

    Only defined for an episode that actually succeeded: spending two iterations to
    produce a broken chart is not efficient, it is just short, and rewarding it would
    make the cheapest strategy "give up immediately".
    """
    stats = result.get("agent_stats") or {}
    used, cap = stats.get("turns_taken"), stats.get("max_turns")
    if not used or not cap:
        return None
    if not result.get("all_passed"):
        return 0.0
    return round(max(0.0, (cap - used + 1) / cap), 4)


def dimensions(result: dict) -> dict:
    sub = {s["id"]: s for s in result.get("subgoal_results", [])}
    rule = result.get("rule_based") or {}
    has_rule = "error" not in rule and rule
    return {
        "code_validity": sub.get("code_valid", {}).get("score"),
        "data_fidelity": rule.get("data_fidelity") if has_rule else None,
        "perturbation_recovery": (rule.get("recovery") if has_rule
                                  else sub.get("flaw_fixed", {}).get("score")),
        "preservation": rule.get("preservation") if has_rule else None,
        "visual_quality": sub.get("visual_quality", {}).get("score"),
        "efficiency": efficiency(result),
    }


def joint_success(result: dict) -> float | None:
    """Whether all three published binary rule checks pass on this episode.

    Keep this separate from ``dimensions``: it is a conjunction of existing rule
    metrics, not another independently measured quality dimension.  Requiring all
    three values to exist also preserves the benchmark convention that undefined
    recovery on a clean chart is not silently converted to zero.
    """
    rule = result.get("rule_based") or {}
    if not rule or "error" in rule:
        return None
    values = [rule.get("executability"), rule.get("recovery_binary"),
              rule.get("preservation_binary")]
    if not all(isinstance(v, (int, float)) for v in values):
        return None
    return float(all(v == 1 for v in values))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--weights", help="six comma-separated weights, in the order above")
    args = ap.parse_args()

    weights = None
    if args.weights:
        weights = [float(w) for w in args.weights.split(",")]
        if len(weights) != 6:
            raise SystemExit(f"--weights needs 6 values, got {len(weights)}")
        if abs(sum(weights) - 1.0) > 1e-6:
            raise SystemExit(f"--weights must sum to 1.0, got {sum(weights)}")

    rows = []
    for d in sorted(args.run_dir.iterdir()):
        f = d / "result.json"
        if not f.is_file():
            continue
        result = json.loads(f.read_text())
        rows.append({"id": d.name, **dimensions(result),
                     JOINT_NAME: joint_success(result)})

    if not rows:
        raise SystemExit(f"no results under {args.run_dir}")

    display_names = [*NAMES, JOINT_NAME]
    head = "  ".join(f"{n.split('_')[0][:6]:>6}" for n in display_names)
    print(f"{'task':38} {head}" + ("   final" if weights else ""))
    print("-" * (40 + len(head) + (8 if weights else 0)))
    for r in rows:
        cells = "  ".join(f"{r[n]:>6.2f}" if r[n] is not None else f"{'--':>6}"
                          for n in display_names)
        line = f"{r['id'][:37]:38} {cells}"
        if weights:
            got = [(w, r[n]) for w, n in zip(weights, NAMES) if r[n] is not None]
            # Renormalised over the dimensions that exist, so a missing one does not
            # silently drag the total towards zero.
            line += f"   {sum(w * v for w, v in got) / sum(w for w, _ in got):>5.3f}"
        print(line)

    print("-" * (40 + len(head) + (8 if weights else 0)))
    means = []
    for n in display_names:
        vals = [r[n] for r in rows if r[n] is not None]
        means.append(f"{statistics.mean(vals):>6.2f}" if vals else f"{'--':>6}")
    print(f"{'mean':38} " + "  ".join(means))
    joint = [r[JOINT_NAME] for r in rows if r[JOINT_NAME] is not None]
    if joint:
        passed = sum(v == 1 for v in joint)
        print(f"\njoint success: {passed}/{len(joint)} = {passed / len(joint):.4f}"
              f"  (undefined: {len(rows) - len(joint)})"
              "\n  executability == 1 AND recovery_binary == 1"
              " AND preservation_binary == 1")
    missing = [n for n in NAMES if all(r[n] is None for r in rows)]
    if missing:
        print(f"\nnot measured: {', '.join(missing)}"
              f"\n  run scripts/score_runs.py first to fill in the rule-based dimensions")
    if not weights:
        print("\nreported as a vector. Pass --weights to combine, deliberately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
