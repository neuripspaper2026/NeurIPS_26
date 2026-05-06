from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class CheckerContext:
    """
    Shared context passed to every correctness checker.
    """

    root: Path
    benchmark: str
    ex_version: str
    metadata: Dict[str, Any]
    benchmark_dir: Path
    results_dir: Path
    log_path: Path


@dataclass
class CheckResult:
    status: str  # pass / fail / skip / error
    message: str
    details: Optional[str] = None
    artifacts: List[Path] = field(default_factory=list)
    records: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class CandidateCollectionResult:
    """
    Outcome produced by the shared correctness artifact collector.
    """

    status: str  # ok / skip / error
    message: Optional[str] = None
    files: List[Path] = field(default_factory=list)

    def ok(self) -> bool:
        return self.status == "ok"


class CorrectnessChecker:
    """
    Base class for benchmark-specific correctness checkers.
    """

    name: str = "base"

    def run(self, context: CheckerContext, candidates: List[Path]) -> CheckResult:
        raise NotImplementedError

    def run_single(
        self,
        *,
        context: Optional[CheckerContext],
        baseline_path: Path,
        candidate_path: Path,
        dataset: Optional[str] = None,
        model: Optional[str] = None,
        version: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> CheckResult:
        raise NotImplementedError

