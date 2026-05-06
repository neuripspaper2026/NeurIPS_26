from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .base import CheckResult, CheckerContext, CorrectnessChecker
from .output_specs import OutputFormat, get_output_format

DEFAULT_DATASET = "default"
IMPROVEMENT_PATTERN = re.compile(
    r"Improvement:\s*([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)"
)
ABS_TOL = 1e-9
REL_TOL = 1e-6


class ParsecChecker(CorrectnessChecker):
    """
    Suite-specific checker for PARSEC benchmarks such as canneal.

    These programs emit textual summaries that include an "Improvement:"
    value describing how much the routing/annealing process improved the
    cost. The correctness rule is simple: a candidate run passes if its
    Improvement is not lower than the baseline's Improvement (within a
    tiny numerical tolerance to absorb floating-point formatting noise).
    """

    name = "parsec"

    def run(self, context: CheckerContext, candidates: List[Path]) -> CheckResult:
        fmt = get_output_format(context.benchmark)
        if fmt is None:
            return CheckResult(
                status="skip",
                message=f"No output spec registered for {context.benchmark}.",
            )

        candidate_files = sorted(self._filter_candidates(candidates, fmt))
        if not candidate_files:
            return CheckResult(
                status="skip",
                message="No PARSEC correctness artifacts found to inspect.",
            )

        baseline_dir = context.benchmark_dir / "input_data"
        records: List[Dict[str, object]] = []
        failures: List[str] = []
        total = len(candidate_files)

        for index, candidate in enumerate(candidate_files, start=1):
            dataset = self._infer_dataset_from_name(candidate)
            baseline_path = self._baseline_path(baseline_dir, dataset, fmt)
            if baseline_path is None or not baseline_path.exists():
                message = f"Baseline output not found for dataset '{dataset}'."
                failures.append(message)
                records.append(
                    self._record(
                        candidate,
                        dataset,
                        status="fail",
                        message=message,
                        index=index,
                        total=total,
                        baseline_improvement=None,
                        candidate_improvement=None,
                    )
                )
                continue

            result = self._compare_improvement(baseline_path, candidate)
            records.append(
                self._record(
                    candidate,
                    dataset,
                    status="pass" if result.ok else "fail",
                    message=result.message,
                    index=index,
                    total=total,
                    baseline_improvement=result.baseline_value,
                    candidate_improvement=result.candidate_value,
                )
            )
            if not result.ok:
                failures.append(f"{candidate.name}: {result.message}")

        if failures:
            return CheckResult(
                status="fail",
                message="; ".join(failures),
                records=records,
            )
        return CheckResult(
            status="pass",
            message="All PARSEC outputs met the Improvement baseline.",
            records=records,
        )

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
        benchmark = context.benchmark if context else "unknown"
        fmt = get_output_format(benchmark)
        if fmt is None:
            return CheckResult(
                status="skip",
                message=f"No output spec registered for {benchmark}.",
            )

        result = self._compare_improvement(baseline_path, candidate_path)
        record = {
            "stage": "check",
            "file": str(candidate_path),
            "file_name": candidate_path.name,
            "dataset": dataset or DEFAULT_DATASET,
            "model": model,
            "version": version,
            "status": "pass" if result.ok else "fail",
            "message": result.message,
            "baseline_improvement": result.baseline_value,
            "candidate_improvement": result.candidate_value,
        }
        return CheckResult(
            status="pass" if result.ok else "fail",
            message=result.message,
            records=[record],
        )

    def _compare_improvement(
        self,
        baseline_path: Path,
        candidate_path: Path,
    ) -> "EvaluationResult":
        try:
            baseline_value = self._extract_improvement(baseline_path)
        except ValueError as exc:
            return EvaluationResult(
                ok=False,
                message=f"Failed to parse baseline improvement: {exc}",
                baseline_value=None,
                candidate_value=None,
            )
        try:
            candidate_value = self._extract_improvement(candidate_path)
        except ValueError as exc:
            return EvaluationResult(
                ok=False,
                message=f"Failed to parse candidate improvement: {exc}",
                baseline_value=baseline_value,
                candidate_value=None,
            )

        tol = max(ABS_TOL, REL_TOL * abs(baseline_value))
        if candidate_value + tol >= baseline_value:
            return EvaluationResult(
                ok=True,
                message=(
                    f"Improvement {candidate_value:.6g} "
                    f">= baseline {baseline_value:.6g}"
                ),
                baseline_value=baseline_value,
                candidate_value=candidate_value,
            )
        return EvaluationResult(
            ok=False,
            message=(
                f"Improvement {candidate_value:.6g} "
                f"< baseline {baseline_value:.6g}"
            ),
            baseline_value=baseline_value,
            candidate_value=candidate_value,
        )

    def _extract_improvement(self, path: Path) -> float:
        with path.open("r", encoding="utf-8", errors="ignore") as fp:
            for line in fp:
                match = IMPROVEMENT_PATTERN.search(line)
                if match:
                    return float(match.group(1))
        raise ValueError("No 'Improvement:' line found.")

    def _baseline_path(
        self,
        base_dir: Path,
        dataset: str,
        fmt: OutputFormat,
    ) -> Optional[Path]:
        candidate_dir = base_dir / dataset / "output"
        if not candidate_dir.exists():
            return None
        suffix = fmt.suffix or ""
        files = (
            sorted(candidate_dir.glob(f"*{suffix}"))
            if suffix
            else sorted(candidate_dir.glob("*"))
        )
        if not files:
            return None
        preferred = [path for path in files if dataset in path.stem]
        return preferred[0] if preferred else files[0]

    def _filter_candidates(
        self,
        files: Sequence[Path],
        fmt: OutputFormat,
    ) -> List[Path]:
        suffix = fmt.suffix or ".txt"
        return [path for path in files if path.suffix == suffix]

    def _infer_dataset_from_name(self, candidate: Path) -> str:
        stem = candidate.stem
        if "_output" not in stem:
            return DEFAULT_DATASET
        core = stem[len("output_") :]
        if not core.endswith("_output"):
            return DEFAULT_DATASET
        tokens = core[: -len("_output")].split("_")
        if len(tokens) < 2:
            return DEFAULT_DATASET
        dataset_token = tokens[-1]
        return dataset_token or DEFAULT_DATASET

    def _record(
        self,
        candidate: Path,
        dataset: str,
        *,
        status: str,
        message: str,
        index: int,
        total: int,
        baseline_improvement: Optional[float],
        candidate_improvement: Optional[float],
    ) -> Dict[str, object]:
        record: Dict[str, object] = {
            "stage": "check",
            "file": str(candidate),
            "file_name": candidate.name,
            "dataset": dataset,
            "model": None,
            "version": None,
            "status": status,
            "message": message,
            "index": index,
            "total": total,
        }
        if baseline_improvement is not None:
            record["baseline_improvement"] = baseline_improvement
        if candidate_improvement is not None:
            record["candidate_improvement"] = candidate_improvement
        return record


class EvaluationResult:
    def __init__(
        self,
        *,
        ok: bool,
        message: str,
        baseline_value: Optional[float],
        candidate_value: Optional[float],
    ) -> None:
        self.ok = ok
        self.message = message
        self.baseline_value = baseline_value
        self.candidate_value = candidate_value


