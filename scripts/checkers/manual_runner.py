from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

from scripts.benchmark_catalog import (
    BENCHMARK_CODEE,
    BENCHMARK_MACHSUITE,
    BENCHMARK_MIBENCH,
    BENCHMARK_PARBOIL,
    BENCHMARK_PARSEC,
    BENCHMARK_POLYBENCH,
    BENCHMARK_RODINIA,
)

from .base import CheckResult, CheckerContext
from .registry import resolve_checker

DEFAULT_ROOT = Path(__file__).resolve().parents[2]

_METADATA_CACHE: Optional[Dict[str, Dict[str, object]]] = None


def _load_metadata() -> Dict[str, Dict[str, object]]:
    global _METADATA_CACHE  # pylint: disable=global-statement
    if _METADATA_CACHE is None:
        combined: Dict[str, Dict[str, object]] = {}
        for source in (
            BENCHMARK_POLYBENCH,
            BENCHMARK_RODINIA,
            BENCHMARK_CODEE,
            BENCHMARK_PARBOIL,
            BENCHMARK_MIBENCH,
            BENCHMARK_PARSEC,
            BENCHMARK_MACHSUITE,
        ):
            combined.update(source)
        _METADATA_CACHE = combined
    return _METADATA_CACHE


def run_single_check(
    *,
    benchmark: str,
    baseline: Union[Path, str],
    target: Union[Path, str],
    threshold: Optional[float] = None,
    root: Optional[Union[Path, str]] = None,
) -> CheckResult:
    """
    Compare a single baseline/target pair using the checker registered for the benchmark.
    """

    baseline_path = Path(baseline).expanduser().resolve()
    target_path = Path(target).expanduser().resolve()
    root_path = Path(root).expanduser().resolve() if root else DEFAULT_ROOT
    metadata = _load_metadata().get(benchmark, {})
    checker = resolve_checker(benchmark)
    context = CheckerContext(
        root=root_path,
        benchmark=benchmark,
        ex_version="MANUAL",
        metadata=metadata,
        benchmark_dir=baseline_path.parent,
        results_dir=baseline_path.parent,
        log_path=baseline_path.parent / "manual_check.log",
    )
    return checker.run_single(
        context=context,
        baseline_path=baseline_path,
        candidate_path=target_path,
        threshold=threshold,
    )

