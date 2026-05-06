from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.benchmark_catalog import (
    BENCHMARK_CODEE,
    BENCHMARK_MACHSUITE,
    BENCHMARK_MIBENCH,
    BENCHMARK_PARBOIL,
    BENCHMARK_PARSEC,
    BENCHMARK_POLYBENCH,
    BENCHMARK_RODINIA,
)
from scripts.checkers import (
    CheckerContext,
    CheckResult,
    collect_candidate_files,
    resolve_checker,
)
from scripts.checkers.output_specs import get_output_format


def _merge_benchmark_metadata() -> Dict[str, Dict[str, Any]]:
    combined: Dict[str, Dict[str, Any]] = {}
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
    return combined


class CorrectnessRunner:
    def __init__(
        self,
        *,
        absolute_path: str,
        log_file: Optional[str] = None,
        activity_log: Optional[str] = None,
    ):
        self.absolute_path = Path(absolute_path)
        self.metadata = _merge_benchmark_metadata()
        self.log_file_path = self._resolve_log_path(
            log_file,
            default_relative="logs/check_correctness_errors.log",
        )
        self.activity_log_path = self._resolve_log_path(
            activity_log,
            default_relative="logs/activity_correctness.log",
        )

    def run_checks(
        self,
        benchmarks: List[str],
        ex_versions: List[str],
        *,
        datasets: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        results = []
        file_counter: Counter[str] = Counter()
        for benchmark in benchmarks:
            for ex_version in ex_versions:
                result = self._run_single(benchmark, ex_version, datasets=datasets)
                results.append(result)
                for record in result.get("records", []):
                    status = record.get("status", "unknown")
                    file_counter[status] += 1
        file_total = sum(file_counter.values())
        summary = {
            "total": len(results),
            "pass": sum(1 for item in results if item["status"] == "pass"),
            "fail": sum(1 for item in results if item["status"] == "fail"),
            "skip": sum(1 for item in results if item["status"] == "skip"),
            "error": sum(1 for item in results if item["status"] == "error"),
            "results": results,
            "file_summary": {
                "total": file_total,
                "pass": file_counter.get("pass", 0),
                "fail": file_counter.get("fail", 0),
                "skip": file_counter.get("skip", 0),
                "error": file_counter.get("error", 0),
                "unknown": file_counter.get("unknown", 0),
            },
        }
        return summary

    def _run_single(
        self,
        benchmark: str,
        ex_version: str,
        *,
        datasets: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        checker = resolve_checker(benchmark)
        metadata = self.metadata.get(benchmark, {})
        benchmark_dir = self.absolute_path / ex_version / benchmark
        results_dir = self.absolute_path / "results" / "correctness" / benchmark / ex_version
        detail_log = results_dir / "correctness.jsonl"
        context = CheckerContext(
            root=self.absolute_path,
            benchmark=benchmark,
            ex_version=ex_version,
            metadata=metadata,
            benchmark_dir=benchmark_dir,
            results_dir=results_dir,
            log_path=results_dir / "check.log",
        )
        self._log_activity(
            "check_start",
            benchmark=benchmark,
            ex_version=ex_version,
            checker=checker.name,
        )
        pattern = "*.txt"
        fmt = get_output_format(benchmark)
        if fmt and fmt.suffix:
            pattern = f"*{fmt.suffix}"
        collection = collect_candidate_files(
            context,
            pattern=pattern,
            datasets=datasets,
        )
        if not collection.ok():
            message = collection.message or "No correctness artifacts found."
            self._log_error(
                benchmark=benchmark,
                ex_version=ex_version,
                message=message,
                level="WARN",
            )
            self._log_activity(
                "check_complete",
                benchmark=benchmark,
                ex_version=ex_version,
                checker=checker.name,
                status="skip",
                message=message,
            )
            print(
                f"[WARN] Correctness skip for {benchmark} ({ex_version}). "
                f"Reason: {message}."
            )
            return {
                "benchmark": benchmark,
                "ex_version": ex_version,
                "status": "skip",
                "message": message,
                "artifacts": [],
                "records": [],
            }
        try:
            result = checker.run(context, collection.files)
        except Exception as exc:  # pylint: disable=broad-except
            self._log_error(
                benchmark=benchmark,
                ex_version=ex_version,
                message=str(exc),
                level="ERROR",
            )
            self._log_activity(
                "check_complete",
                benchmark=benchmark,
                ex_version=ex_version,
                checker=checker.name,
                status="error",
                message=str(exc),
            )
            print(
                f"[ERROR] Correctness checker error for {benchmark} ({ex_version}): {exc}."
            )
            return {
                "benchmark": benchmark,
                "ex_version": ex_version,
                "status": "error",
                "message": str(exc),
                "artifacts": [],
                "records": [],
            }

        if result.status in {"fail", "error"}:
            self._log_error(
                benchmark=benchmark,
                ex_version=ex_version,
                message=result.message,
                level="ERROR",
            )
        elif result.status == "skip":
            self._log_error(
                benchmark=benchmark,
                ex_version=ex_version,
                message=result.message,
                level="WARN",
            )

        self._log_activity(
            "check_complete",
            benchmark=benchmark,
            ex_version=ex_version,
            checker=checker.name,
            status=result.status,
            message=result.message,
        )
        print(
            f"[INFO] Correctness result for {benchmark} ({ex_version}) -> {result.status.upper()} "
            f"(details JSONL: {detail_log})"
        )
        self._append_detail_entries(detail_log, benchmark, ex_version, checker.name, result)
        return {
            "benchmark": benchmark,
            "ex_version": ex_version,
            "status": result.status,
            "message": result.message,
            "artifacts": [str(path) for path in result.artifacts],
            "records": result.records,
        }

    def _resolve_log_path(self, candidate: Optional[str], *, default_relative: str) -> Path:
        if candidate:
            path = Path(candidate)
            if not path.is_absolute():
                path = (self.absolute_path / path).resolve()
            else:
                path = path.resolve()
        else:
            path = (self.absolute_path / default_relative).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _log_error(self, *, benchmark: str, ex_version: str, message: str, level: str) -> None:
        record = (
            f"[TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"[{level}] [{ex_version}] Benchmark: {benchmark}\n"
            f"Message: {message}\n"
            + "-" * 80
        )
        with self.log_file_path.open("a", encoding="utf-8") as fp:
            fp.write(record + "\n")

    def _log_activity(self, action: str, **details: Any) -> None:
        record = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "action": action,
        }
        record.update(details)
        with self.activity_log_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _append_detail_entries(
        self,
        detail_log: Path,
        benchmark: str,
        ex_version: str,
        checker: str,
        result: CheckResult,
    ) -> None:
        detail_log.parent.mkdir(parents=True, exist_ok=True)
        records = result.records or [
            {
                "stage": "check",
                "status": result.status,
                "message": result.message,
            }
        ]
        with detail_log.open("a", encoding="utf-8") as fp:
            for record in records:
                payload = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "benchmark": benchmark,
                    "ex_version": ex_version,
                    "checker": checker,
                }
                payload.update(record)
                if "stage" not in payload:
                    payload["stage"] = "check"
                fp.write(json.dumps(payload, ensure_ascii=False) + "\n")

