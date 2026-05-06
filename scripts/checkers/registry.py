from __future__ import annotations

from typing import Dict, Optional

from scripts.benchmark_catalog import (
    BENCHMARK_CODEE,
    BENCHMARK_MACHSUITE,
    BENCHMARK_MIBENCH,
    BENCHMARK_PARBOIL,
    BENCHMARK_PARSEC,
    BENCHMARK_POLYBENCH,
    BENCHMARK_RODINIA,
)

from .base import CheckResult, CheckerContext, CorrectnessChecker
from .codee import CodeeChecker
from .mibench import MibenchChecker
from .machsuite import MachSuiteChecker
from .polybench import PolybenchChecker
from .parsec import ParsecChecker
from .parboil import ParboilChecker
from .rodinia import RodiniaChecker


class PlaceholderChecker(CorrectnessChecker):
    name = "placeholder"

    def run(self, context: CheckerContext, candidates) -> CheckResult:
        return CheckResult(
            status="skip",
            message=(
                f"No correctness checker registered for {context.benchmark}; "
                "placeholder checker in use."
            ),
        )

    def run_single(
        self,
        *,
        context: Optional[CheckerContext],
        baseline_path,
        candidate_path,
        dataset: Optional[str] = None,
        model: Optional[str] = None,
        version: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> CheckResult:
        benchmark = context.benchmark if context else "unknown"
        return CheckResult(
            status="skip",
            message=f"No correctness checker registered for {benchmark}; placeholder checker in use.",
            records=[
                {
                    "stage": "check",
                    "file": str(candidate_path),
                    "file_name": getattr(candidate_path, "name", str(candidate_path)),
                    "dataset": dataset,
                    "model": model,
                    "version": version,
                    "status": "skip",
                    "message": "No checker available.",
                }
            ],
        )


POLYBENCH_CHECKER = PolybenchChecker()
RODINIA_CHECKER = RodiniaChecker()
PARBOIL_CHECKER = ParboilChecker()
CODEE_CHECKER = CodeeChecker()
MIBENCH_CHECKER = MibenchChecker()
MACHSUITE_CHECKER = MachSuiteChecker()
PARSEC_CHECKER = ParsecChecker()
PLACEHOLDER_CHECKER = PlaceholderChecker()

SUITE_REGISTRY: Dict[str, CorrectnessChecker] = {
    "polybench": POLYBENCH_CHECKER,
    "rodinia": RODINIA_CHECKER,
    "codee": CODEE_CHECKER,
    "parboil": PARBOIL_CHECKER,
    "mibench": MIBENCH_CHECKER,
    "parsec": PARSEC_CHECKER,
    "machsuite": MACHSUITE_CHECKER,
}


def resolve_checker(benchmark_name: str) -> CorrectnessChecker:
    if benchmark_name in BENCHMARK_POLYBENCH:
        return SUITE_REGISTRY["polybench"]
    if benchmark_name in BENCHMARK_RODINIA:
        return SUITE_REGISTRY["rodinia"]
    if benchmark_name in BENCHMARK_CODEE:
        return SUITE_REGISTRY["codee"]
    if benchmark_name in BENCHMARK_PARBOIL:
        return SUITE_REGISTRY["parboil"]
    if benchmark_name in BENCHMARK_MIBENCH:
        return SUITE_REGISTRY["mibench"]
    if benchmark_name in BENCHMARK_PARSEC:
        return SUITE_REGISTRY["parsec"]
    if benchmark_name in BENCHMARK_MACHSUITE:
        return SUITE_REGISTRY["machsuite"]
    return PLACEHOLDER_CHECKER

