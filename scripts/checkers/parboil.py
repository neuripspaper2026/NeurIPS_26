from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .base import CheckResult, CheckerContext, CorrectnessChecker
from .output_specs import get_output_format, OutputFormat

DEFAULT_DATASET = "default"


@dataclass
class ComparisonResult:
    ok: bool
    message: str


class ParboilChecker(CorrectnessChecker):
    name = "parboil"

    def run(self, context: CheckerContext, candidates: List[Path]) -> CheckResult:
        fmt = get_output_format(context.benchmark)
        if not fmt:
            return CheckResult(
                status="skip",
                message=f"No output spec registered for {context.benchmark}.",
            )
        candidate_files = sorted(self._filter_candidates(candidates, fmt))
        if not candidate_files:
            return CheckResult(status="skip", message="No output files to compare.")

        baseline_dir = self._baseline_dir(context)
        records: List[Dict[str, object]] = []
        failures: List[str] = []
        total = len(candidate_files)

        for idx, candidate in enumerate(candidate_files, start=1):
            dataset = self._infer_dataset_from_name(candidate)
            baseline_path = self._baseline_path(baseline_dir, dataset)
            if not baseline_path or not baseline_path.exists():
                message = f"Baseline output not found for dataset '{dataset}'."
                failures.append(message)
                records.append(
                    self._record(
                        candidate,
                        dataset,
                        status="fail",
                        message=message,
                        index=idx,
                        total=total,
                    )
                )
                continue

            result = self._compare_files(baseline_path, candidate, fmt)
            records.append(
                self._record(
                    candidate,
                    dataset,
                    status="pass" if result.ok else "fail",
                    message=result.message,
                    index=idx,
                    total=total,
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
            message="All binary outputs matched baseline.",
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
        if context is None:
            raise ValueError("CheckerContext is required for parboil check.")
        fmt = get_output_format(context.benchmark)
        if not fmt:
            return CheckResult(
                status="skip",
                message=f"No output spec registered for {context.benchmark}.",
            )
        if threshold is not None:
            fmt = OutputFormat(
                suite=fmt.suite,
                type=fmt.type,
                dtype=fmt.dtype,
                endian=fmt.endian,
                atol=threshold,
                rtol=fmt.rtol,
                suffix=fmt.suffix,
            )
        result = self._compare_files(baseline_path, candidate_path, fmt)
        record = {
            "stage": "check",
            "file": str(candidate_path),
            "file_name": candidate_path.name,
            "dataset": dataset or "manual",
            "model": model,
            "version": version,
            "status": "pass" if result.ok else "fail",
            "message": result.message,
        }
        return CheckResult(
            status="pass" if result.ok else "fail",
            message=result.message,
            records=[record],
        )

    def _compare_files(
        self,
        baseline_path: Path,
        candidate_path: Path,
        fmt: OutputFormat,
    ) -> ComparisonResult:
        if fmt.is_binary:
            return self._compare_binary_files(baseline_path, candidate_path, fmt)
        return self._compare_text_files(baseline_path, candidate_path, fmt)

    def _compare_binary_files(
        self,
        baseline_path: Path,
        candidate_path: Path,
        fmt: OutputFormat,
    ) -> ComparisonResult:
        dtype = np.dtype(fmt.dtype or "float32")
        if fmt.endian == "big":
            dtype = dtype.newbyteorder(">")
        else:
            dtype = dtype.newbyteorder("<")
        baseline_data = np.fromfile(baseline_path, dtype=dtype)
        candidate_data = np.fromfile(candidate_path, dtype=dtype)

        if baseline_data.shape != candidate_data.shape:
            return ComparisonResult(
                ok=False,
                message=f"Shape mismatch: {baseline_data.shape} vs {candidate_data.shape}",
            )

        if np.issubdtype(dtype, np.floating):
            ok = np.allclose(
                baseline_data,
                candidate_data,
                atol=fmt.atol,
                rtol=fmt.rtol,
            )
        else:
            ok = np.array_equal(baseline_data, candidate_data)
        if ok:
            return ComparisonResult(True, "Outputs matched.")
        diff_index = int(np.where(baseline_data != candidate_data)[0][0])
        return ComparisonResult(
            ok=False,
            message=f"Mismatch at element {diff_index}",
        )

    def _compare_text_files(
        self,
        baseline_path: Path,
        candidate_path: Path,
        fmt: OutputFormat,
    ) -> ComparisonResult:
        dtype = np.dtype(fmt.dtype or "float64")
        try:
            baseline_data = np.loadtxt(baseline_path, dtype=dtype)
            candidate_data = np.loadtxt(candidate_path, dtype=dtype)
        except ValueError as exc:  # pylint: disable=broad-except
            return ComparisonResult(False, f"Failed to parse text output: {exc}")
        if baseline_data.shape != candidate_data.shape:
            return ComparisonResult(
                ok=False,
                message=f"Shape mismatch: {baseline_data.shape} vs {candidate_data.shape}",
            )
        if np.issubdtype(dtype, np.floating):
            ok = np.allclose(
                baseline_data,
                candidate_data,
                atol=fmt.atol,
                rtol=fmt.rtol,
            )
        else:
            ok = np.array_equal(baseline_data, candidate_data)
        if ok:
            return ComparisonResult(True, "Outputs matched.")
        diff_index = int(np.where(baseline_data != candidate_data)[0][0])
        return ComparisonResult(
            ok=False,
            message=f"Mismatch at element {diff_index}",
        )

    def _baseline_dir(self, context: CheckerContext) -> Path:
        return context.benchmark_dir / "input_data"

    def _baseline_path(self, base_dir: Path, dataset: str) -> Optional[Path]:
        candidate_dir = base_dir / dataset / "output"
        if not candidate_dir.exists():
            return None
        files = sorted(candidate_dir.glob("*"))
        return files[0] if files else None

    def _filter_candidates(self, files: Sequence[Path], fmt: OutputFormat) -> List[Path]:
        suffix = fmt.suffix or ".txt"
        return [path for path in files if path.suffix == suffix]

    def _infer_dataset_from_name(self, candidate: Path) -> str:
        stem = candidate.stem
        if not stem.startswith("output_") or not stem.endswith("_output"):
            return DEFAULT_DATASET
        core = stem[len("output_"):-len("_output")]
        parts = core.split("_")
        if len(parts) < 3:
            return DEFAULT_DATASET
        dataset_token = parts[-1]
        return dataset_token or DEFAULT_DATASET

    def _record(
        self,
        candidate: Path,
        dataset: str,
        *,
        status: str,
        message: str,
        index: Optional[int],
        total: Optional[int],
    ) -> Dict[str, object]:
        record = {
            "stage": "check",
            "file": str(candidate),
            "file_name": candidate.name,
            "dataset": dataset,
            "model": None,
            "version": None,
            "status": status,
            "message": message,
        }
        if index is not None:
            record["index"] = index
        if total is not None:
            record["total"] = total
        return record

