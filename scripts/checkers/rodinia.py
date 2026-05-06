from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .base import CheckResult, CheckerContext, CorrectnessChecker
from .output_specs import OutputFormat, get_output_format

DEFAULT_DATASET = "default"


@dataclass
class ComparisonResult:
    ok: bool
    message: str


class RodiniaChecker(CorrectnessChecker):
    name = "rodinia"

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
            raise ValueError("CheckerContext is required for rodinia check.")
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
        if fmt.suffix == ".pgm":
            return self._compare_pgm_files(baseline_path, candidate_path)
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

    def _compare_pgm_files(
        self,
        baseline_path: Path,
        candidate_path: Path,
    ) -> ComparisonResult:
        baseline_header, baseline_pixels = self._load_pgm(baseline_path)
        candidate_header, candidate_pixels = self._load_pgm(candidate_path)
        if baseline_header != candidate_header:
            return ComparisonResult(
                ok=False,
                message=f"PGM header mismatch: {baseline_header} vs {candidate_header}",
            )
        if baseline_pixels.shape != candidate_pixels.shape:
            return ComparisonResult(
                ok=False,
                message=f"Shape mismatch: {baseline_pixels.shape} vs {candidate_pixels.shape}",
            )
        if np.array_equal(baseline_pixels, candidate_pixels):
            return ComparisonResult(True, "Outputs matched.")
        diff_index = int(np.where(baseline_pixels != candidate_pixels)[0][0])
        return ComparisonResult(
            ok=False,
            message=f"Mismatch at element {diff_index}",
        )

    def _load_pgm(self, path: Path) -> Tuple[Tuple[str, int, int, int], np.ndarray]:
        with path.open("r", encoding="utf-8") as fp:
            tokens: List[str] = []
            for line in fp:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                tokens.extend(stripped.split())
        if len(tokens) < 4:
            raise ValueError(f"PGM header missing fields in {path}")
        magic = tokens[0]
        if magic not in {"P2", "P5"}:
            raise ValueError(f"Unsupported PGM type '{magic}' in {path}")
        try:
            width = int(tokens[1])
            height = int(tokens[2])
            maxval = int(tokens[3])
        except ValueError as exc:
            raise ValueError(f"Invalid PGM header in {path}") from exc
        pixel_tokens = tokens[4:]
        expected = width * height
        if len(pixel_tokens) != expected:
            raise ValueError(
                f"PGM pixel count mismatch in {path}: expected {expected}, got {len(pixel_tokens)}"
            )
        pixels = np.asarray([int(tok) for tok in pixel_tokens], dtype=np.int32)
        return (magic, width, height, maxval), pixels

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
                    r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", raw_line
                )
                for match in matches:
                    values.append(converter(match))
        if not values:
            raise ValueError(f"No numeric tokens extracted from {path}")
        return np.asarray(values, dtype=dtype)

    def _baseline_path(self, base_dir: Path, dataset: str, fmt: OutputFormat) -> Optional[Path]:
        candidate_dir = base_dir / dataset / "output"
        if not candidate_dir.exists():
            return None
        suffix = fmt.suffix or ""
        files = sorted(candidate_dir.glob(f"*{suffix}")) if suffix else sorted(candidate_dir.glob("*"))
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

