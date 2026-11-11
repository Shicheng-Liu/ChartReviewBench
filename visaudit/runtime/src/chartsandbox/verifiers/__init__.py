"""Verifier registry. Importing this package registers all built-in verifiers."""
from . import data, execution, vlm_judge  # noqa: F401  (side-effect: registration)
from .base import (
    CHEAP,
    REGISTRY,
    VerifierResult,
    VerifyContext,
    register,
    run_verifier,
)
from .vlm_judge import set_judge_backend

__all__ = [
    "REGISTRY",
    "CHEAP",
    "VerifierResult",
    "VerifyContext",
    "register",
    "run_verifier",
    "set_judge_backend",
]
