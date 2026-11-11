"""Comparison primitives: the two tolerance modes and the table rule.

strict  = exact (float epsilon only, so 7.6 vs 7.6000001 still passes)
lenient = within 5 % relative
wide    = within 10 %, computed as a diagnostic column, never reported as a mode

The hallucination rule: a value that is literally in table.csv and written
differently is a failure in BOTH modes. Tolerance exists for values the script
computes at runtime, not for misread table cells.
"""
import csv
import io
import math
import re

EPS = 1e-9
LENIENT = 0.05
WIDE = 0.10
MODES = ("strict", "lenient")


def norm_text(s) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().casefold()


def is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and not math.isnan(float(v))


def numbers_close(a, b, rel: float) -> bool:
    a, b = float(a), float(b)
    if math.isnan(a) or math.isnan(b):
        return False
    if a == b:
        return True
    scale = max(abs(a), abs(b))
    if scale == 0:
        return True
    if rel <= 0:  # strict: float noise only
        return abs(a - b) <= max(EPS, scale * 1e-9)
    return abs(a - b) <= scale * rel


def tolerance_of(mode: str) -> float:
    return {"strict": 0.0, "lenient": LENIENT, "wide": WIDE}[mode]


def equal(a, b, mode: str = "strict") -> bool:
    """Structural equality of two property values under a tolerance mode."""
    rel = tolerance_of(mode)
    if a is None or b is None:
        return a is None and b is None
    if is_number(a) and is_number(b):
        return numbers_close(a, b, rel)
    if isinstance(a, str) and isinstance(b, str):
        return norm_text(a) == norm_text(b)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(equal(x, y, mode) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return False
        return all(equal(a[k], b[k], mode) for k in a)
    return norm_text(a) == norm_text(b)


class TableIndex:
    """Which numbers are literally readable from table.csv."""

    def __init__(self, csv_text: str = ""):
        self.numbers = set()
        self.texts = set()
        if not csv_text:
            return
        try:
            for row in csv.reader(io.StringIO(csv_text)):
                for cell in row:
                    cell = (cell or "").strip()
                    if not cell:
                        continue
                    self.texts.add(norm_text(cell))
                    try:
                        self.numbers.add(round(float(cell.replace(",", "")), 10))
                    except ValueError:
                        pass
        except csv.Error:
            pass

    def has_number(self, v) -> bool:
        if not is_number(v) or not self.numbers:
            return False
        return any(numbers_close(v, t, 0.0) for t in self.numbers)

    @classmethod
    def from_path(cls, path):
        try:
            return cls(open(path, encoding="utf-8", errors="replace").read())
        except OSError:
            return cls("")


def value_verdict(gt_value, sub_value, table: TableIndex, *, grounded=None) -> dict:
    """Compare one plotted number. Returns pass/fail per mode plus the reason.

    A table-readable ground-truth value never gets tolerance: writing 7.58 for a
    table's 7.6 is a hallucination, not a rounding artefact.
    """
    exact = equal(gt_value, sub_value, "strict")
    if grounded is None:
        grounded = table.has_number(gt_value)
    out = {
        "strict": exact,
        "lenient": exact,
        "wide": exact,
        "grounded": grounded,
        "hallucination": False,
    }
    if exact:
        return out
    if grounded:
        out["hallucination"] = True  # stays False in every mode
        return out
    out["lenient"] = equal(gt_value, sub_value, "lenient")
    out["wide"] = equal(gt_value, sub_value, "wide")
    return out
