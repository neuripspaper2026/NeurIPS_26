from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

from .base import CheckResult, CheckerContext, CorrectnessChecker

DATASET_ORDER = {
    "extra-large": 0,
    "large": 1,
    "medium": 2,
    "mini": 3,
    "small": 4,
}


class PolybenchChecker(CorrectnessChecker):
    name = "polybench"

    def run(self, context: CheckerContext, candidates: List[Path]) -> CheckResult:
        correctness_dir = context.benchmark_dir / "EX5_correctness"
        candidate_files = [path for path in candidates if path.suffix.lower() == ".txt"]
        if not candidate_files:
            return CheckResult(
                status="skip",
                message=f"No *.txt outputs in {correctness_dir}",
            )

        parsed_candidates = self._parse_candidates(candidate_files)
        if not parsed_candidates:
            return CheckResult(
                status="skip",
                message="No candidate outputs matched expected naming pattern (<benchmark>_<model>[_vN]_<dataset>_output.txt).",
            )

        parsed_candidates = self._sort_candidates(parsed_candidates)

        datasets = {item["dataset"] for item in parsed_candidates}
        baseline_paths: Dict[str, Path] = {}
        baseline_arrays: Dict[str, Optional["np.ndarray"]] = {}
        for dataset in datasets:
            baseline_path = self._baseline_output_path(context, dataset)
            if not baseline_path:
                return CheckResult(
                    status="fail",
                    message=f"Baseline output not found for dataset '{dataset}'. Expected under input_data/{dataset}/output/.",
                )
            baseline_paths[dataset] = baseline_path

        threshold = context.metadata.get("threshold", 0.0)
        failures: List[str] = []
        records: List[Dict[str, Any]] = []
        total = len(parsed_candidates)
        for idx, candidate in enumerate(parsed_candidates, start=1):
            dataset = candidate["dataset"]
            if dataset not in baseline_arrays:
                baseline_path = baseline_paths[dataset]
                baseline_arrays[dataset] = self._load_numeric_array(baseline_path, dataset)
            baseline_arr = baseline_arrays[dataset]
            print(
                f"[INFO] [polybench-checker] [{idx}/{total}] Comparing {candidate['path'].name} "
                f"(dataset={dataset}, model={candidate['model']}, version={candidate['version']})"
            )
            candidate_arr = self._load_numeric_array(candidate["path"], dataset, cache=False)
            status, msg, record = self._evaluate_pair(
                baseline_arr=baseline_arr,
                candidate_arr=candidate_arr,
                dataset=dataset,
                model=candidate["model"],
                version=candidate["version"],
                threshold=threshold or 0.0,
                candidate_path=candidate["path"],
                index=idx,
                total=total,
            )
            records.append(record)
            if status != "pass":
                failures.append(f"{candidate['path'].name}: {msg}")

        if failures:
            return CheckResult(
                status="fail",
                message="; ".join(failures),
                records=records,
            )
        return CheckResult(
            status="pass",
            message="All outputs matched baseline within threshold.",
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
        dataset_label = dataset or "manual"
        threshold_value = (
            threshold
            if threshold is not None
            else (context.metadata.get("threshold", 0.0) if context else 0.0)
        )
        baseline_arr = self._load_numeric_array(baseline_path, dataset_label, quiet=True)
        candidate_arr = self._load_numeric_array(
            candidate_path,
            dataset_label,
            cache=False,
            quiet=True,
        )
        status, message, record = self._evaluate_pair(
            baseline_arr=baseline_arr,
            candidate_arr=candidate_arr,
            dataset=dataset_label,
            model=model,
            version=version,
            threshold=threshold_value,
            candidate_path=candidate_path,
        )
        return CheckResult(
            status=status,
            message=message,
            records=[record],
        )

    def _parse_candidates(self, files: List[Path]) -> List[Dict[str, Any]]:
        parsed: List[Dict[str, Any]] = []
        for path in files:
            dataset, model, version = self._extract_metadata_from_filename(path.name)
            if not dataset or not model:
                continue
            parsed.append(
                {
                    "path": path,
                    "dataset": dataset,
                    "model": model,
                    "version": version,
                }
            )
        return parsed

    def _sort_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        def _version_key(value: Optional[str]) -> int:
            if value is None:
                return 0
            try:
                return int(value)
            except ValueError:
                return 0

        def _dataset_order(dataset: str) -> int:
            return DATASET_ORDER.get(dataset, len(DATASET_ORDER))

        return sorted(
            candidates,
            key=lambda item: (
                item["model"],
                _version_key(item.get("version")),
                _dataset_order(item["dataset"]),
                item["path"].name,
            ),
        )

    def _extract_metadata_from_filename(self, name: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Expected pattern: output_<model>_<version>_<dataset>_output.<ext>
        """
        stem = Path(name).stem
        if not stem.startswith("output_") or not stem.endswith("_output"):
            return None, None, None
        core = stem[len("output_"):-len("_output")]
        parts = core.split("_")
        if len(parts) < 3:
            return None, None, None
        dataset = parts[-1]
        version_token = parts[-2]
        model = "_".join(parts[:-2])
        if not model or not dataset:
            return None, None, None
        version = None
        if version_token.lower() != "baseline":
            if version_token.startswith("v") and version_token[1:].isdigit():
                version = version_token[1:]
        else:
                version = version_token
        return dataset, model, version

    def _baseline_output_path(self, context: CheckerContext, dataset: str) -> Optional[Path]:
        base_dir = context.benchmark_dir / "input_data" / dataset / "output"
        if not base_dir.exists():
            return None
        txt_candidates = sorted(base_dir.glob("*.txt"))
        if not txt_candidates:
            return None
        preferred = [p for p in txt_candidates if dataset in p.name]
        return preferred[0] if preferred else txt_candidates[0]

    def _load_numeric_array(self, path: Path, dataset: str, cache: bool = True, quiet: bool = False):
        from scripts.tol import parse_numeric_arrays

        if not quiet:
            action = "baseline" if cache else "candidate"
            print(f"[INFO] [polybench-checker] Loading {action} data for dataset={dataset} from {path}")
        return parse_numeric_arrays(str(path))

    def _compare_arrays(self,
                        baseline_arr: "np.ndarray",
                        candidate_arr: "np.ndarray",
                        tolerance: float) -> Tuple[bool, str]:
        rows = min(baseline_arr.shape[0], candidate_arr.shape[0])
        cols = min(baseline_arr.shape[1], candidate_arr.shape[1])
        base_sub = baseline_arr[:rows, :cols]
        cand_sub = candidate_arr[:rows, :cols]
        for i in range(rows):
            for j in range(cols):
                val_b = base_sub[i, j]
                val_c = cand_sub[i, j]
                if np.isnan(val_b) and np.isnan(val_c):
                    continue
                is_b_int = (not np.isnan(val_b)) and float(val_b).is_integer()
                is_c_int = (not np.isnan(val_c)) and float(val_c).is_integer()
                if is_b_int and is_c_int:
                    if val_b != val_c:
                        return False, f"Mismatch at row={i+1}, col={j+1}: int {val_b} != {val_c}"
                else:
                    if not np.isclose(val_b, val_c, atol=tolerance, rtol=0.0):
                        diff = abs(val_b - val_c)
                        return False, (
                            f"Float mismatch at row={i+1}, col={j+1}: "
                            f"{val_b} vs {val_c}, diff={diff} > tol={tolerance}"
                        )
        return True, "All numeric tokens match within tolerance."

    def _evaluate_pair(
        self,
        *,
        baseline_arr: "np.ndarray",
        candidate_arr: "np.ndarray",
        dataset: str,
        model: Optional[str],
        version: Optional[str],
        threshold: float,
        candidate_path: Path,
        index: Optional[int] = None,
        total: Optional[int] = None,
    ) -> Tuple[str, str, Dict[str, Any]]:
        try:
            ok, msg = self._compare_arrays(
                baseline_arr,
                candidate_arr,
                threshold,
            )
        except Exception as exc:  # pylint: disable=broad-except
            ok = False
            msg = f"Exception during compare: {exc}"
        status = "pass" if ok else "fail"
        record = {
            "stage": "check",
            "file": str(candidate_path),
            "file_name": candidate_path.name,
            "dataset": dataset,
            "model": model,
            "version": version,
            "status": status,
            "message": msg,
        }
        if index is not None:
            record["index"] = index
        if total is not None:
            record["total"] = total
        return status, msg, record

