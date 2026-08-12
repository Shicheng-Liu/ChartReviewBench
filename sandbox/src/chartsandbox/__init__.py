"""chartsandbox — execution + verification runtime for a long-horizon chart-agent benchmark.

This package is *only* the sandbox: it takes a task instance (produced by the data
pipeline, following the contract in `contract.py`), runs an agent inside an isolated,
stateful Python workspace, and scores the result with pluggable verifiers.

It knows nothing about how tasks are built from ChartNet — that is a separate concern.
"""

__version__ = "0.1.0"
