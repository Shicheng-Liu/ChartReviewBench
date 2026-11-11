"""Long-horizon metrics derived from an episode."""
from __future__ import annotations


def progress_curve(first_pass_step: dict[str, int | None], n_steps: int, n_subgoals: int) -> list[float]:
    """Fraction of subgoals completed after each step (0..n_steps-1)."""
    if n_subgoals == 0:
        return []
    curve = []
    for step in range(max(n_steps, 1)):
        done = sum(1 for s in first_pass_step.values() if s is not None and s <= step)
        curve.append(round(done / n_subgoals, 4))
    return curve


def summarize(result: dict) -> str:
    lines = [
        f"task:        {result['task_id']}  ({result['family']})",
        f"final_score: {result['final_score']:.3f}"
        + ("  ALL SUBGOALS PASSED" if result["all_passed"] else ""),
        f"subgoals:    {result['subgoals_passed']}/{result['subgoals_total']} passed",
        f"steps:       {result['steps_taken']}/{result['step_budget']}"
        + (f"  (horizon_hint={result['horizon_hint']})" if result.get("horizon_hint") else ""),
        f"wall_time:   {result['wall_time_s']:.1f}s",
        f"tools:       {result['tool_counts']}",
        "per-subgoal:",
    ]
    for sg in result["subgoal_results"]:
        mark = "PASS" if sg["passed"] else "FAIL"
        fp = sg["first_pass_step"]
        fp_s = f"first passed @step {fp}" if fp is not None else "never passed"
        lines.append(f"  [{mark}] {sg['id']:<24} score={sg['score']:.2f}  w={sg['weight']}  ({fp_s})")
        for v in sg["verifiers"]:
            vmark = "ok" if v["passed"] else "xx"
            lines.append(f"        - {vmark} {v['name']:<12} {v['score']:.2f}  {v['detail']}")
    lines.append(f"progress:    {result['progress_curve']}")
    return "\n".join(lines)
