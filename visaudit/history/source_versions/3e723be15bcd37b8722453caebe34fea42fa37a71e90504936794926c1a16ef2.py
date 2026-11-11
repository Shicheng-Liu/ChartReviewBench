"""Evaluation tracks and interaction budgets for the repair benchmark.

Three selectable protocols share a fixed instance set and support comparisons
across interaction budgets:

* Track A, *diagnosed repair*: the instruction names the flaw type, so the
  episode measures repair alone.  The gap to Track B is the discovery cost.
* Track B, *autonomous repair*: the agent has to find the flaw itself.
* Track C, *open-world verification*: the chart may or may not be faithful.
  An explicit population lists flawed and additional clean charts. The optional
  twins mode pairs each flawed task with its clean reference. Editing a clean
  chart is a false repair; leaving a flawed chart alone is a missed repair.

A track and a step budget are part of the task identity so two protocols of
one instance never overwrite each other on disk. The generic runner defaults to
Track B with 20 steps; the released suite explicitly selects uncapped budgets.
"""
from __future__ import annotations

import re
import textwrap

TRACKS = ("A", "B", "C")
DEFAULT_TRACK = "B"
DEFAULT_STEP_BUDGET = 20
DEFAULT_WALL_TIME_S = 900
STEP_LADDER = (1, 2, 3, 5, 10, 20)
_WRAP = 80
_SUFFIX = re.compile(r"__t-(A|B|C)(clean)?$|__s-(\d+|none)$|__w-(\d+|none)$")

# What Track A tells the agent.  These name the flaw, never the repaired state
# (that is the judge rubric in to_sandbox.RUBRICS, which the agent never sees).
FLAW_HINTS = {
    "wrong-title": "the chart title contradicts the provided data or summary",
    "wrong-axis-label": "an axis label names a quantity other than the one plotted",
    "missing-legend": "the legend that identifies the plotted series is missing",
    "swapped-legend-labels": "two legend entries are attached to each other's series",
    "value-perturbation": "one plotted value does not match its number in table.csv",
    "swapped-series": "two series plot each other's data",
    "dropped-series": "a series described in the data is missing from the chart",
    "wrong-chart-type": "the chart or a series uses a chart type that contradicts summary.txt",
    "axis-truncation": "a value axis is cut so that some plotted data is not visible",
    "color-contradicts-summary": "a plotted element's color contradicts the summary",
}


def flaw_components(category: str) -> list[str]:
    """The flaw categories behind a task category.

    SingleFlow tasks name their category directly; MultiFlow folders use
    ``multiN__a+b[+c]`` so the diagnosis can name every component.
    """
    if "__" in category and category.split("__", 1)[0].startswith("multi"):
        prefix, joined = category.split("__", 1)
        parts = joined.split("+")
        if (prefix not in {"multi2", "multi3"} or len(parts) != int(prefix[-1])
                or not all(parts) or len(set(parts)) != len(parts)):
            raise ValueError("invalid multi-flaw category %r" % category)
        return parts
    return [category]


def validate_track(track: str | None) -> str:
    value = (track or DEFAULT_TRACK).strip().upper()
    if value not in TRACKS:
        raise ValueError("unknown track %r (known: %s)" % (track, ", ".join(TRACKS)))
    return value


def validate_step_budget(steps: int | None) -> int | None:
    if steps is None:
        return None
    value = int(steps)
    if isinstance(steps, bool) or value < 1:
        raise ValueError("step budget must be a positive integer, got %r" % (steps,))
    return value


def validate_wall_time(seconds: int | None) -> int | None:
    if seconds is None:
        return None
    value = int(seconds)
    if isinstance(seconds, bool) or value < 1:
        raise ValueError("wall-time limit must be positive or None")
    return value


def task_suffix(track: str | None = None, step_budget: int | None = DEFAULT_STEP_BUDGET,
                clean: bool = False, *, wall_time_s: int | None = DEFAULT_WALL_TIME_S) -> str:
    """The task-id suffix of a protocol; empty for the generic runner defaults."""
    track = validate_track(track)
    steps = validate_step_budget(step_budget)
    wall_time_s = validate_wall_time(wall_time_s)
    if clean and track != "C":
        raise ValueError("a clean twin only exists under Track C")
    parts = []
    if track != DEFAULT_TRACK or clean:
        parts.append("__t-%s%s" % (track, "clean" if clean else ""))
    if steps != DEFAULT_STEP_BUDGET:
        parts.append("__s-%s" % ("none" if steps is None else steps))
    if wall_time_s != DEFAULT_WALL_TIME_S:
        parts.append("__w-%s" % ("none" if wall_time_s is None else wall_time_s))
    return "".join(parts)


def parse_suffix(task_id: str) -> tuple[str, str, int | None, bool]:
    """(base id, track, step budget, clean) from a task id."""
    track, steps, clean = DEFAULT_TRACK, DEFAULT_STEP_BUDGET, False
    base = task_id
    while True:
        match = _SUFFIX.search(base)
        if not match:
            break
        if match.group(3) is not None:
            steps = None if match.group(3) == "none" else int(match.group(3))
        elif match.group(4) is None:
            track = match.group(1)
            clean = bool(match.group(2))
        base = base[:match.start()]
    if clean and track != "C":
        raise ValueError("%s: a clean twin only exists under Track C" % task_id)
    return base, track, steps, clean


def instruction_for(base_instruction: str, track: str | None, category: str,
                    clean: bool = False) -> str:
    """The task text of a protocol, derived from the modality instruction.

    Track B is the modality instruction unchanged.  Track A appends the flaw
    diagnosis.  Track C replaces the "it is flawed" premise by an open question
    and asks for an explicit decision; the flawed task and its clean twin get
    the identical text, so the text carries no signal about which one it is.
    """
    track = validate_track(track)
    if track == "B":
        return base_instruction
    if track == "A":
        components = flaw_components(category)
        hints = [FLAW_HINTS.get(component) for component in components]
        if not components or any(hint is None for hint in hints):
            raise ValueError("no Track A diagnosis for category %r" % category)
        if len(hints) == 1:
            text = "Diagnosis: the flaw is that %s. Repair exactly that." % hints[0]
        else:
            listed = "; ".join("(%d) %s" % (i + 1, hint) for i, hint in enumerate(hints))
            text = ("Diagnosis: the chart carries %d flaws: %s. Repair "
                    "exactly those." % (len(hints), listed))
        diagnosis = textwrap.fill(text, width=_WRAP)
        return base_instruction.rstrip("\n") + "\n\n" + diagnosis + "\n"
    premise = textwrap.fill(
        "chart.py builds a chart from the data in table.csv and saves it as "
        "output.png; chart_current.png is how it looks right now. The chart may be "
        "faithful to table.csv and summary.txt, or one or more things about it may "
        "contradict them. Your first job is to decide which.", width=_WRAP)
    action = textwrap.fill(
        "If the chart is faithful, do not edit chart.py: run it once so that "
        "output.png exists and call finish. If something is wrong, fix chart.py, "
        "change only what is actually wrong, run it so that output.png shows a "
        "faithful chart, and call finish.", width=_WRAP)
    return "%s\n\n%s\n" % (premise, action)


def track_summary(episodes: list) -> dict:
    """Track C detection outcomes over completed episodes.

    `edited` is whether the submitted chart.py differs from the one the agent
    was handed; on a clean chart that is a false repair, on a flawed instance
    leaving it alone is a missed repair.  Recovery on a clean twin is undefined
    by construction (its footprint is empty) and stays undefined.
    """
    completed = [e for e in episodes if e.get("status") == "ok"]
    clean = [e for e in completed if e.get("clean")]
    flawed = [e for e in completed if not e.get("clean")]
    false_repairs = [e for e in clean if e.get("edited") is True]
    missed = [e for e in flawed if e.get("edited") is False]

    def rate(part, whole):
        return round(len(part) / len(whole), 4) if whole else None

    return {
        "clean_n": len(clean),
        "flawed_n": len(flawed),
        "false_repair_n": len(false_repairs),
        "false_repair_rate": rate(false_repairs, clean),
        "missed_repair_n": len(missed),
        "missed_repair_rate": rate(missed, flawed),
        "edit_on_flawed_rate": rate([e for e in flawed if e.get("edited") is True], flawed),
    }
