from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .base import CheckResult, CheckerContext, CorrectnessChecker
from .output_specs import OutputFormat, get_output_format

DEFAULT_DATASET = "default"


@dataclass
class ComparisonResult:
    ok: bool
    message: str


class CodeeChecker(CorrectnessChecker):
    """
    Suite-specific checker for Codee benchmarks.

    Codee programs follow the same naming convention as other suites:
    `output_<model>_<version>_<dataset>_output.<suffix>`. Each benchmark
    registers an `OutputFormat` entry that defines the file suffix as well
    as the value type (text/binary) and dtype. This checker simply looks up
    the spec and dispatches to the appropriate comparison routine.
    """

    name = "codee"

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

        baseline_dir = context.benchmark_dir / "input_data"
        records: List[Dict[str, object]] = []
        failures: List[str] = []
        total = len(candidate_files)

        for idx, candidate in enumerate(candidate_files, start=1):
            dataset = self._infer_dataset_from_name(candidate)
            baseline_path = self._baseline_path(baseline_dir, dataset, fmt)
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
            message="All outputs matched baseline.",
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
            raise ValueError("CheckerContext is required for codee check.")
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
        # Special handling for haccmk: self-validating output
        if "haccmk" in str(candidate_path).lower():
            return self._check_haccmk_self_validation(candidate_path, fmt)
        if fmt.type == "binary":
            return self._compare_binary_files(baseline_path, candidate_path, fmt)
        return self._compare_text_files(baseline_path, candidate_path, fmt)

    def _compare_binary_files(
        self,
        baseline_path: Path,
        candidate_path: Path,
        fmt: OutputFormat,
    ) -> ComparisonResult:
        dtype = np.dtype(fmt.dtype or "float32")
        dtype = dtype.newbyteorder(">" if fmt.endian == "big" else "<")
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

    def _check_haccmk_self_validation(
        self,
        candidate_path: Path,
        fmt: OutputFormat,
    ) -> ComparisonResult:
        """
        Check haccmk output by parsing internal validation results.
        Expected format:
          Result validation: <value>
          Result expected  : <value>
        Returns pass if |validation - expected| <= atol
        """
        try:
            with candidate_path.open("r", encoding="utf-8") as fp:
                content = fp.read()
            
            # Extract "Result validation: <number>"
            validation_match = re.search(r"Result\s+validation:\s*([-+]?[\d.]+(?:[eE][-+]?\d+)?)", content)
            expected_match = re.search(r"Result\s+expected\s*:\s*([-+]?[\d.]+(?:[eE][-+]?\d+)?)", content)
            
            if not validation_match or not expected_match:
                return ComparisonResult(
                    ok=False,
                    message="Could not parse 'Result validation' or 'Result expected' from output",
                )
            
            validation = float(validation_match.group(1))
            expected = float(expected_match.group(1))
            diff = abs(validation - expected)
            
            if diff <= fmt.atol:
                return ComparisonResult(
                    ok=True,
                    message=f"Self-validation passed: diff={diff:.6e} <= atol={fmt.atol}",
                )
            else:
                return ComparisonResult(
                    ok=False,
                    message=f"Self-validation failed: diff={diff:.6e} > atol={fmt.atol} (validation={validation}, expected={expected})",
                )
        except Exception as e:
            return ComparisonResult(
                ok=False,
                message=f"Error parsing haccmk output: {e}",
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
        except ValueError:
            baseline_data = self._load_text_tokens(baseline_path, dtype)
            candidate_data = self._load_text_tokens(candidate_path, dtype)
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

    def _load_text_tokens(self, path: Path, dtype: np.dtype) -> np.ndarray:
        values: List[float] = []
        converter = dtype.type if hasattr(dtype, "type") else float
        with path.open("r", encoding="utf-8") as fp:
            for raw_line in fp:
                matches = re.findall(
                    r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?",
                    raw_line,
                )
                for token in matches:
                    values.append(converter(token))
        if not values:
            raise ValueError(f"No numeric tokens extracted from {path}")
        return np.asarray(values, dtype=dtype)

    def _baseline_path(self, base_dir: Path, dataset: str, fmt: OutputFormat) -> Optional[Path]:
        candidate_dir = base_dir / dataset / "output"
        if not candidate_dir.exists():
            return None
        suffix = fmt.suffix or ""
        files = (
            sorted(candidate_dir.glob(f"*{suffix}"))
            if suffix
            else sorted(candidate_dir.glob("*"))
        )
        return files[0] if files else None

    def _filter_candidates(self, files: Sequence[Path], fmt: OutputFormat) -> List[Path]:
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
        if len(tokens) < 3:
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
        index: Optional[int],
        total: Optional[int],
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
        }
        if index is not None:
            record["index"] = index
        if total is not None:
            record["total"] = total
        return record

