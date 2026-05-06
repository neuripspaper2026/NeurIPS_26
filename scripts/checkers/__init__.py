"""Correctness checker registry and shared interfaces."""

from .base import (
    CandidateCollectionResult,
    CheckResult,
    CheckerContext,
    CorrectnessChecker,
)
from .collector import collect_candidate_files
from .manual_runner import run_single_check
from .registry import resolve_checker

__all__ = [
    "CandidateCollectionResult",
    "CheckResult",
    "CheckerContext",
    "CorrectnessChecker",
    "collect_candidate_files",
    "resolve_checker",
    "run_single_check",
]

