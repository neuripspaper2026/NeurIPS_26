"""
Utility helpers for experiment (EX) version naming.
"""
from __future__ import annotations

from typing import Dict, List, Optional

_EXPERIMENT_ROOTS: Dict[str, str] = {
    "EX1": "EX1",
    "EX2": "EX2",
    "EX3": "EX3",
}

AUTO_RUN_EX_CHOICES: List[str] = ["EX1", "EX2", "EX3"]
TIME_MEASUREMENT_EX_CHOICES: List[str] = ["EX1", "EX2", "EX3"]


def canonical_ex_version(ex_version: Optional[str]) -> Optional[str]:
    """Return the canonical EX label. No-op pass-through, kept as a stable hook."""
    return ex_version


def experiment_root_dir(ex_version: str) -> str:
    """Physical directory name under the repository root for an EX label."""
    return _EXPERIMENT_ROOTS.get(ex_version, ex_version)


def optimized_subdir_name(ex_version: str) -> str:
    """Name of the optimized-code directory inside a benchmark for this EX."""
    canonical = canonical_ex_version(ex_version) or ex_version
    return f"{canonical}_optimized_codes"


def all_supported_ex_versions() -> List[str]:
    return list(_EXPERIMENT_ROOTS.keys())
