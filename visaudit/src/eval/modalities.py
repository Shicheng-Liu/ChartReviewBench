"""Which modalities a task puts in front of the agent.

The benchmark is multi-modal by construction, but the configuration is what makes
it a benchmark with difficulty levels: with all four modalities the repair is as
easy as it gets, and every modality taken away makes the same instance harder.
The default is therefore all four; any other set is an explicit experiment and
carries its own code in the task id so the two never collide on disk.

A modality set is a canonical tuple ordered like ALL. `validate` is the guard
that stops an experiment from silently producing an unsolvable difficulty level:
without a truth modality the flaw cannot be found, without an artefact modality
there is nothing to repair.
"""
import textwrap

ALL = ("chart", "code", "table", "summary")
DEFAULT = ALL

# The file each modality becomes in the agent's workspace.
WORKSPACE_FILE = {
    "chart": "chart_current.png",
    "code": "chart.py",
    "table": "table.csv",
    "summary": "summary.txt",
}

# The order task.yaml lists the workspace files in. It predates the modality
# configuration and is kept so a default task stays byte-identical to the task
# dirs that earlier runs already produced.
WORKSPACE_ORDER = ("code", "table", "summary", "chart")

# Short ids for the task id and for filenames. The image gets "i" because chart
# and code both start with a "c".
_CODE = {"chart": "i", "code": "c", "table": "t", "summary": "s"}
_FROM_CODE = {v: k for k, v in _CODE.items()}

TRUTH = ("table", "summary")      # what the agent can hold the chart against
ARTEFACT = ("chart", "code")      # what carries the flaw

# The width the existing instruction was hard-wrapped at; keeping it is what
# makes the generated default text identical to the one written by hand.
_WRAP = 80


def parse(spec=None) -> tuple:
    """Canonicalise a modality set given as a string, an iterable, or None.

    "chart, table" and ["table", "chart"] are the same set; the result is always
    ordered like ALL and deduplicated. None means the default, all four.
    """
    if spec is None:
        return DEFAULT
    if isinstance(spec, str):
        items = [part.strip() for part in spec.replace(";", ",").split(",")]
    else:
        items = [str(part).strip() for part in spec]
    items = [part.lower() for part in items if part]
    if not items:
        raise ValueError("empty modality set; pass None for the default %s"
                         % (",".join(DEFAULT),))
    unknown = [part for part in items if part not in ALL]
    if unknown:
        raise ValueError("unknown modality %s (known: %s)"
                         % (", ".join(repr(u) for u in unknown), ", ".join(ALL)))
    return tuple(m for m in ALL if m in items)


def code(mods=None) -> str:
    """The short id of a modality set, e.g. "icts" for the default."""
    return "".join(_CODE[m] for m in parse(mods))


def from_code(s: str) -> tuple:
    """Inverse of `code`."""
    letters = str(s).strip().lower()
    unknown = sorted(set(letters) - set(_FROM_CODE))
    if unknown:
        raise ValueError("unknown modality code %s in %r (known: %s)"
                         % (", ".join(repr(u) for u in unknown), s,
                            ", ".join("%s=%s" % (v, k) for k, v in _CODE.items())))
    if not letters:
        raise ValueError("empty modality code")
    return parse([_FROM_CODE[c] for c in letters])


def validate(mods=None) -> tuple:
    """The canonical set, or a ValueError naming why it cannot define a task."""
    mods = parse(mods)
    if not [m for m in TRUTH if m in mods]:
        raise ValueError(
            "modality set %s holds no truth: without table.csv or summary.txt the agent "
            "has nothing to compare the chart against, so the flaw is undiscoverable by "
            "construction. Keep at least one of %s."
            % (",".join(mods), ", ".join(TRUTH)))
    if not [m for m in ARTEFACT if m in mods]:
        raise ValueError(
            "modality set %s holds nothing to repair: without the flawed chart or the "
            "flawed code the agent never sees the flaw. Keep at least one of %s."
            % (",".join(mods), ", ".join(ARTEFACT)))
    return mods


def workspace_files(mods=None) -> list:
    """The workspace filenames of a modality set, in task.yaml order."""
    mods = parse(mods)
    return [WORKSPACE_FILE[m] for m in WORKSPACE_ORDER if m in mods]


def instruction(mods=None) -> str:
    """The task instruction for a modality set.

    It names exactly the files the agent actually has and never mentions one it
    does not, and it never names the category or the flaw. With the code present
    the task is a repair; without it the agent has to write chart.py itself.
    """
    mods = validate(mods)
    para1, para2 = (_repair_text(mods) if "code" in mods else _write_text(mods))
    return "%s\n\n%s\n" % (textwrap.fill(para1, width=_WRAP),
                           textwrap.fill(para2, width=_WRAP))


def _truth_phrase(mods: tuple) -> str:
    if "table" in mods and "summary" in mods:
        return "what table.csv contains or what summary.txt says the chart shows"
    if "table" in mods:
        return "what table.csv contains"
    return "what summary.txt says the chart shows"


def _repair_text(mods: tuple) -> tuple:
    if "table" in mods:
        opening = ("chart.py builds a chart from the data in table.csv and saves it as "
                   "output.png. The chart it currently produces is not faithful to that "
                   "data:")
    else:
        opening = ("chart.py builds a chart and saves it as output.png. The chart it "
                   "currently produces is not faithful to the data behind it:")
    para1 = "%s at least one thing about it contradicts %s." % (opening, _truth_phrase(mods))
    if "chart" in mods:
        para1 += " chart_current.png is how the chart looks right now."
    para2 = ("Find what is wrong, fix chart.py, and run it so that output.png shows a "
             "chart that is faithful to the data. Change only what is actually wrong and "
             "leave the rest of the chart as it is. Call finish when you are done.")
    return para1, para2


def _write_text(mods: tuple) -> tuple:
    """Without the script the task stops being an edit: the agent has to rebuild
    the chart from what it can see, and only the flaw may come out different."""
    if "table" in mods:
        opening = ("chart_current.png is a chart built from the data in table.csv, and it "
                   "is not faithful to that data:")
    else:
        opening = ("chart_current.png is the chart as it currently looks, and it is not "
                   "faithful to the data behind it:")
    para1 = ("%s at least one thing about it contradicts %s. There is no script for it in "
             "the workspace." % (opening, _truth_phrase(mods)))
    para2 = ("Write chart.py so that it builds the chart chart_current.png was meant to "
             "be, and run it so that output.png shows a chart that is faithful to the "
             "data. Reproduce the rest of the chart as it looks now and change only what "
             "is actually wrong. Call finish when you are done.")
    return para1, para2
