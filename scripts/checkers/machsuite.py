from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .base import CheckResult, CheckerContext, CorrectnessChecker
from .output_specs import OutputFormat, get_output_format

DEFAULT_DATASET = "default"
SUCCESS_PATTERN = re.compile(r"\bSUCCESS\b", re.IGNORECASE)
FAIL_PATTERN = re.compile(r"\bFAIL(?:URE)?\b", re.IGNORECASE)


class MachSuiteChecker(CorrectnessChecker):
    """
    Checker for MachSuite programs.

    These executables already compare their computed output against a provided
    reference file and print either "Success." or "Fail." to stdout. During
    `correctness-run`, that stdout stream is redirected into files inside
    `EX5_correctness`, so correctness can be determined by scanning the text.
    """

    name = "machsuite"

    def run(self, context: CheckerContext, candidates: List[Path]) -> CheckResult:
        fmt = get_output_format(context.benchmark)
        if not fmt:
            return CheckResult(
                status="skip",
                message=f"No output spec registered for {context.benchmark}.",
            )
        candidate_files = sorted(self._filter_candidates(candidates, fmt))
        if not candidate_files:
            return CheckResult(
                status="skip",
                message="No MachSuite correctness artifacts found to inspect.",
            )

        records: List[Dict[str, object]] = []
        failures: List[str] = []
        total = len(candidate_files)

        for index, candidate in enumerate(candidate_files, start=1):
            dataset, model, version = self._parse_metadata(candidate)
            ok, message = self._evaluate_output(candidate)
            record = {
                "stage": "check",
                "file": str(candidate),
                "file_name": candidate.name,
                "dataset": dataset,
                "model": model,
                "version": version,
                "status": "pass" if ok else "fail",
                "message": message,
                "index": index,
                "total": total,
            }
            records.append(record)
            if not ok:
                failures.append(f"{candidate.name}: {message}")

        if failures:
            return CheckResult(
                status="fail",
                message="; ".join(failures),
                records=records,
            )
        return CheckResult(
            status="pass",
            message="All MachSuite executions reported Success.",
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
        _ = baseline_path, threshold  # Unused for MachSuite.
        ok, message = self._evaluate_output(candidate_path)
        record = {
            "stage": "check",
            "file": str(candidate_path),
            "file_name": candidate_path.name,
            "dataset": dataset or "manual",
            "model": model,
            "version": version,
            "status": "pass" if ok else "fail",
            "message": message,
        }
        return CheckResult(
            status="pass" if ok else "fail",
            message=message,
            records=[record],
        )

    def _filter_candidates(
        self,
        files: Sequence[Path],
        fmt: OutputFormat,
    ) -> List[Path]:
        """
        MachSuite correctness-run outputs follow the standard capture naming:
        `output_<model>_<version>_<dataset>_output.<suffix>`. We ignore any
        legacy artifacts (e.g., `aes_output_claude_v1.data`) to avoid misreading
        golden reference files as stdout captures.
        """

        suffix = fmt.suffix or ".txt"
        selected: List[Path] = []
        for path in files:
            if path.suffix != suffix:
                continue
            name = path.name
            if not (name.startswith("output_") and name.endswith(f"_output{suffix}")):
                continue
            selected.append(path)
        return selected

    def _parse_metadata(self, candidate: Path) -> Tuple[str, Optional[str], Optional[str]]:
        dataset = DEFAULT_DATASET
        model: Optional[str] = None
        version: Optional[str] = None
        stem = candidate.stem
        if stem.startswith("output_") and stem.endswith("_output"):
            core = stem[len("output_") : -len("_output")]
            tokens = [token for token in core.split("_") if token]
            if tokens:
                dataset = tokens[-1] or DEFAULT_DATASET
                candidate_version = tokens[-2] if len(tokens) >= 2 else None
                if candidate_version and candidate_version.startswith("v") and candidate_version[1:].isdigit():
                    version = candidate_version
                    model_tokens = tokens[:-2]
                else:
                    model_tokens = tokens[:-1]
                if model_tokens:
                    model = "_".join(model_tokens)
        return dataset, model, version

    def _evaluate_output(self, path: Path) -> Tuple[bool, str]:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            return False, f"Failed to read output ({exc})."
        stripped = text.strip()
        if not stripped:
            return False, "Output file is empty."
        success = SUCCESS_PATTERN.search(stripped)
        failure = FAIL_PATTERN.search(stripped)
        if success and not failure:
            return True, "Detected 'Success' marker."
        if failure and not success:
            return False, "Detected explicit 'Fail' marker."
        if success and failure:
            return False, "Both success and failure markers present; treating as failure."
        return False, "No success marker found in stdout."


