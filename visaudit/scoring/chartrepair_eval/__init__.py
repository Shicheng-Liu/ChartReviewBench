"""The ChartRepairBench scorer, extracted from the benchmark repository.

`capture` runs a plotting script in an isolated working directory and reads the
figure objects out of matplotlib or plotly. `metrics.score` compares the agent's
submission against the ground truth and the flawed input and returns recovery,
preservation and data fidelity, each as a continuous score and as an
all-or-nothing binary. No aggregate score is defined on purpose.
"""
import sys as _sys
from pathlib import Path as _Path

# capture.py imports `harness` as a top-level module, as it does inside the
# benchmark repository; make the package directory importable so the extracted
# copy works unchanged.
_sys.path.insert(0, str(_Path(__file__).resolve().parent))

from . import capture, compare, metrics  # noqa: F401,E402
