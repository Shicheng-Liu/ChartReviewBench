"""Verifier plugin contract.

A verifier is a pure function of (workspace state, oracle, params) -> Result.
Register with @register("<type>"); tasks reference the type in task.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class VerifierResult:
    name: str
    passed: bool
    score: float  # in [0, 1]
    detail: str = ""


@dataclass
class VerifyContext:
    workspace: Path              # the agent's (post-run) workspace
    oracle: Path                 # read-only reference dir
    params: dict[str, Any] = field(default_factory=dict)
    last_exec: dict | None = None   # last execute_python result (for execution checks)


REGISTRY: dict[str, Callable[[VerifyContext], VerifierResult]] = {}

# verifiers cheap enough to run on every step for the progress curve
CHEAP: set[str] = set()


def register(type_name: str, cheap: bool = False):
    def deco(fn: Callable[[VerifyContext], VerifierResult]):
        REGISTRY[type_name] = fn
        if cheap:
            CHEAP.add(type_name)
        return fn

    return deco


def run_verifier(type_name: str, ctx: VerifyContext) -> VerifierResult:
    fn = REGISTRY.get(type_name)
    if fn is None:
        return VerifierResult(type_name, False, 0.0, f"unknown verifier type {type_name!r}")
    try:
        return fn(ctx)
    except Exception as e:  # a verifier must never crash the runner
        return VerifierResult(type_name, False, 0.0, f"verifier error: {e!r}")
