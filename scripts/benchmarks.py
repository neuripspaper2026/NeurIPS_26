"""Benchmark auto-run utilities split from arg.py."""
from __future__ import annotations

import os
import re
import time
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import re
import subprocess
import math

from scripts.benchmark_catalog import (
    BENCHMARK_CODEE,
    BENCHMARK_MACHSUITE,
    BENCHMARK_MIBENCH,
    BENCHMARK_PARBOIL,
    BENCHMARK_PARSEC,
    BENCHMARK_POLYBENCH,
    BENCHMARK_RODINIA,
)
from scripts.benchmark_catalog_ex2 import (
    BENCHMARK_CODEE_EX2,
    BENCHMARK_MACHSUITE_EX2,
    BENCHMARK_MIBENCH_EX2,
    BENCHMARK_PARBOIL_EX2,
    BENCHMARK_PARSEC_EX2,
    BENCHMARK_POLYBENCH_EX2,
    BENCHMARK_RODINIA_EX2,
)
from configs.EX3_configs.benchmark_catalog_ex3 import (
    BENCHMARK_RODINIA_EX3,
)
from scripts.ex_versions import canonical_ex_version, optimized_subdir_name, experiment_root_dir
from .head_code import resolve_head_code_paths
from configs.EX3_configs.head_code_ex3 import resolve_head_code_paths_ex3
from scripts.llm import create_llm_client
from scripts.benchmark_args import BENCHMARK_ARG_MAP, BENCHMARK_ARG_MAP_EX3

# LLM API keys are read from the environment.
# Export them before invoking benchmark auto-run, e.g.:
#   export OPENAI_API_KEY="..."
#   export ANTHROPIC_API_KEY="..."
#   export TOGETHER_API_KEY="..."
# Never hardcode keys in source.

DEFAULT_DATASET = "default"
KNOWN_COMPILERS = {
    "gcc",
    "g++",
    "clang",
    "clang++",
    "icc",
    "icpc",
    "hipcc",
    "nvcc",
}

class Benchmarks():
    # Store benchmark metadata (correctness pattern + line numbers)
    benchmark_polybench = BENCHMARK_POLYBENCH
    benchmark_rodinia = BENCHMARK_RODINIA
    benchmark_codee = BENCHMARK_CODEE
    benchmark_parboil = BENCHMARK_PARBOIL
    benchmark_mibench = BENCHMARK_MIBENCH
    benchmark_parsec = BENCHMARK_PARSEC
    benchmark_machsuite = BENCHMARK_MACHSUITE

    def __len__(self):
        return len(self.benchmark_dict)

    def __init__(self,
                 write_state: bool = None,
                 absolute_path: str = None,
                 llm_model: str = None,
                 llm_abbrev: str = None,
                 model_family: str = None,
                 customer_meta: dict = None,
                 log_file: Optional[str] = None,
                 activity_log: Optional[str] = None,
                 llm_guard_enabled: bool = False,
                 llm_guard_limit: int = 12000):
        self.write_state = write_state

        self.absolute_path = Path(absolute_path)

        self.model_family = model_family if model_family is not None else "together.ai"
        default_model = llm_model if llm_model is not None else "meta-llama/Llama-3-8b-chat-hf"
        default_abbrev = llm_abbrev if llm_abbrev is not None else "llama"
        if self.model_family == "hpc-coder":
            if llm_model is None:
                default_model = "hpcgroup/hpc-coder-v2-16b"
            if llm_abbrev is None:
                default_abbrev = "hpc"
        self.llm_model = default_model
        self.llm_abbrev = default_abbrev
        self.log_file_path = self._resolve_log_path(
            log_file,
            default_relative="logs/benchmark_errors.log",
        )
        self.activity_log_path = self._resolve_log_path(
            activity_log,
            default_relative="logs/activity_makefile.log",
        )
        self.activity_log_is_default = activity_log is None
        self.correctness_capture_activity_log_path = self._resolve_log_path(
            None,
            default_relative="logs/activity_correctness_capture.log",
        )
        self.correctness_capture_error_log_path = self._resolve_log_path(
            None,
            default_relative="logs/correctness_capture_errors.log",
        )
        self.llm_guard_enabled = llm_guard_enabled
        self.llm_guard_limit = llm_guard_limit
        self.optimized_codes = {}  # 存储优化后的代码
        self._suffix_cache: Dict[tuple[str, str], str] = {}
        self.polybench_list = list(self.benchmark_polybench.keys())
        self.rodinia_list = list(self.benchmark_rodinia.keys())
        self.codee_list = list(self.benchmark_codee.keys())
        self.parboil_list = list(self.benchmark_parboil.keys())
        self.machsuite_list = list(self.benchmark_machsuite.keys())
        self.mibench_list = list(self.benchmark_mibench.keys())
        self.parsec_list = list(self.benchmark_parsec.keys())
        self.rodinia_list_ex3 = list(BENCHMARK_RODINIA_EX3.keys())
        

        if customer_meta is not None:
            self.benchmark_dict = customer_meta
            self.benchmark_dict_ex2 = customer_meta
        else:
            self.benchmark_dict = (
                self.benchmark_polybench.copy() | 
                self.benchmark_rodinia.copy() | 
                self.benchmark_codee.copy() | 
                self.benchmark_parboil.copy() |
                self.benchmark_mibench.copy() |
                self.benchmark_parsec.copy() |
                self.benchmark_machsuite.copy()
            )
            self.benchmark_dict_ex2 = (
                BENCHMARK_POLYBENCH_EX2.copy() |
                BENCHMARK_RODINIA_EX2.copy() |
                BENCHMARK_CODEE_EX2.copy() |
                BENCHMARK_PARBOIL_EX2.copy() |
                BENCHMARK_MIBENCH_EX2.copy() |
                BENCHMARK_PARSEC_EX2.copy() |
                BENCHMARK_MACHSUITE_EX2.copy()
            )
            self.benchmark_dict_ex3 = (
                BENCHMARK_RODINIA_EX3.copy()
            )

        self.llm_client = create_llm_client(
            self.model_family,
            self.llm_model,
            absolute_path=self.absolute_path,
        )

    def log_error(self, ex_version: str, benchmark: str, model: Optional[str], path: Optional[str], error: Exception, level: str = "ERROR"):
        """
        Append a formatted error entry directly to the configured log file and emit an activity event.
        """
        error_type = type(error).__name__
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        model_name = model or self.llm_model
        entry = {
            "timestamp": timestamp,
            "level": level,
            "experiment": ex_version,
            "benchmark": benchmark,
            "model": model_name,
            "path": path or "-",
            "reason": str(error),
            "error_type": error_type,
        }
        self._append_error_entry(entry)
        self.log_activity(
            "error",
            level=level,
            ex_version=ex_version,
            benchmark=benchmark,
            model=model_name,
            path=path,
            error_type=error_type,
            message=str(error),
        )

    def _append_error_entry(self, entry: Dict[str, Any]) -> None:
        """
        Write a human-readable error entry into the configured log file.
        """
        target = self.log_file_path
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"[TIME: {entry['timestamp']}] [{entry['level']}] [{entry['experiment']}] Benchmark: {entry['benchmark']} | Model: {entry['model']}",
            f"Reason: {entry['reason']}",
            f"Path: {entry['path']}",
            f"Error Type: {entry['error_type']}",
            "-" * 80,
        ]
        with target.open("a", encoding="utf-8") as fp:
            fp.write("\n".join(lines) + "\n")

    def log_activity(self, action: str, **details: Any):
        """
        Append a JSON line capturing informational events for successful runs.
        """
        if not self.activity_log_path:
            return
        ex_version = details.get("ex_version")
        target = self.activity_log_path
        if self.activity_log_is_default and ex_version:
            suffix = ex_version.lower()
            target = target.parent / f"{target.stem}_{suffix}{target.suffix}"
        record = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "action": action,
            "model": self.llm_model,
        }
        for key, value in details.items():
            if value is None:
                continue
            if isinstance(value, (Path, )):
                record[key] = str(value)
            else:
                record[key] = value
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record) + "\n")

    def configure_guard(self, enabled: bool, limit: Optional[int] = None):
        self.llm_guard_enabled = enabled
        if limit is not None:
            self.llm_guard_limit = limit

    def configure_correctness_logging(self, activity_log: Optional[str], error_log: Optional[str]) -> None:
        if activity_log:
            self.correctness_capture_activity_log_path = self._resolve_log_path(
                activity_log,
                default_relative="logs/activity_correctness_capture.log",
            )
        if error_log:
            self.correctness_capture_error_log_path = self._resolve_log_path(
                error_log,
                default_relative="logs/correctness_capture_errors.log",
            )

    def available_benchmarks(self, ex_version: Optional[str] = None, ex_versions: Optional[List[str]] = None) -> List[str]:
        """
        Return available benchmarks for specific EX version(s) or all versions.
        
        Args:
            ex_version: Single EX version (takes precedence over ex_versions)
            ex_versions: List of EX versions to get union of benchmarks
        """
        if ex_version:
            metadata_dict = self._metadata_dict_for(ex_version)
            return sorted(list(metadata_dict.keys()))
        
        if ex_versions:
            # Return union of benchmarks across specified versions
            all_benchmarks = set()
            for ver in ex_versions:
                metadata_dict = self._metadata_dict_for(ver)
                all_benchmarks.update(metadata_dict.keys())
            return sorted(list(all_benchmarks))
        
        # If no version specified, return all benchmarks across all versions
        all_benchmarks = set(self.benchmark_dict.keys())
        if hasattr(self, "benchmark_dict_ex2"):
            all_benchmarks.update(self.benchmark_dict_ex2.keys())
        if hasattr(self, "benchmark_dict_ex3"):
            all_benchmarks.update(self.benchmark_dict_ex3.keys())
        return sorted(list(all_benchmarks))

    def _benchmark_dir(self, ex_version: str, benchmark_name: str) -> Path:
        root_name = experiment_root_dir(ex_version)
        return self.absolute_path / root_name / benchmark_name

    def _metadata_dict_for(self, ex_version: Optional[str]) -> Dict[str, Dict[str, Any]]:
        canonical = canonical_ex_version(ex_version) if ex_version else ex_version
        if canonical and canonical.upper() == "EX2":
            return getattr(self, "benchmark_dict_ex2", self.benchmark_dict)
        if canonical and canonical.upper() == "EX3":
            return getattr(self, "benchmark_dict_ex3", self.benchmark_dict)
        return self.benchmark_dict

    def _get_benchmark_metadata(self, benchmark_name: str, ex_version: Optional[str] = None) -> Dict[str, Any]:
        return self._metadata_dict_for(ex_version).get(benchmark_name, {})

    def _optimized_dir(self, ex_version: str, benchmark_name: str) -> Path:
        base_dir = self._benchmark_dir(ex_version, benchmark_name)
        canonical_name = optimized_subdir_name(ex_version)
        requested_name = f"{ex_version}_optimized_codes"
        canonical_path = base_dir / canonical_name
        if requested_name == canonical_name:
            return canonical_path
        requested_path = base_dir / requested_name
        if requested_path.exists():
            return requested_path
        if canonical_path.exists():
            self._ensure_dir_alias(requested_path, canonical_path)
            return canonical_path
        return canonical_path

    def _ensure_dir_alias(self, alias_path: Path, target_path: Path) -> None:
        if alias_path.exists() or not target_path.exists():
            return
        try:
            alias_path.parent.mkdir(parents=True, exist_ok=True)
            alias_path.symlink_to(target_path, target_is_directory=True)
        except OSError:
            # Creating a symlink can fail on some filesystems; fall back silently.
            pass

    def _target_spec(self, benchmark_name: str, *, ex_version: Optional[str] = None):
        metadata = self._get_benchmark_metadata(benchmark_name, ex_version)
        return metadata.get("target_file")

    def _target_basename_and_suffix(
        self,
        benchmark_name: str,
        target_index: int = 0,
        *,
        ex_version: Optional[str] = None,
    ) -> tuple[str, str]:
        spec = self._target_spec(benchmark_name, ex_version=ex_version)
        if isinstance(spec, (list, tuple)):
            try:
                primary = spec[target_index]
            except IndexError as exc:
                raise IndexError(
                    f"[auto_run] target_index {target_index} is out of range for benchmark {benchmark_name}"
                ) from exc
        else:
            primary = spec
        path = Path(primary)
        stem = path.stem or benchmark_name
        suffix = path.suffix or ".c"
        return stem, suffix

    def _build_output_path(self, ex_version: str, benchmark_name: str, attempt: Optional[int], target_index: int = 0) -> Path:
        base_dir = self._optimized_dir(ex_version, benchmark_name)
        base_dir.mkdir(parents=True, exist_ok=True)
        stem, suffix = self._target_basename_and_suffix(
            benchmark_name,
            target_index=target_index,
            ex_version=ex_version,
        )
        version_tag = f"_v{attempt}" if attempt is not None else ""
        filename = f"{stem}_{self.llm_abbrev}{version_tag}{suffix}"
        return base_dir / filename

    def _resolve_log_path(self, candidate: Optional[str], *, default_relative: str) -> Path:
        """
        Resolve log paths so that relative paths are anchored at the repository root.
        """
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

    def _log_capture_activity(self, action: str, **details: Any) -> None:
        record = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "action": action,
        }
        for key, value in details.items():
            if isinstance(value, Path):
                record[key] = str(value)
            else:
                record[key] = value
        target = self.correctness_capture_activity_log_path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _log_capture_error(self, *, benchmark: str, ex_version: str, message: str, level: str = "ERROR") -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            f"[TIME: {timestamp}] [{level}] [{ex_version}] Benchmark: {benchmark}",
            f"Message: {message}",
            "-" * 80,
        ]
        target = self.correctness_capture_error_log_path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fp:
            fp.write("\n".join(lines) + "\n")

    def _resolve_log_root_base(self, log_root: Optional[str]) -> Path:
        """
        Resolve the directory used to store make/build logs.
        Relative paths are anchored under `<absolute_path>/results/`.
        """
        base_candidate = Path(log_root) if log_root else Path("make_results")
        if base_candidate.is_absolute():
            resolved = base_candidate
        else:
            if base_candidate.parts and base_candidate.parts[0] == "results":
                resolved = (self.absolute_path / base_candidate).resolve()
            else:
                resolved = (self.absolute_path / "results" / base_candidate).resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    def _benchmark_suite(self, benchmark_name: str) -> str:
        if benchmark_name in self.polybench_list:
            return "polybench"
        if benchmark_name in self.rodinia_list or benchmark_name in self.rodinia_list_ex3:
            return "rodinia"
        if benchmark_name in self.codee_list:
            return "codee"
        if benchmark_name in self.parboil_list:
            return "parboil"
        if benchmark_name in self.mibench_list:
            return "mibench"
        if benchmark_name in self.parsec_list:
            return "parsec"
        if benchmark_name in self.machsuite_list:
            return "machsuite"
        return "unknown"

    def _correctness_results_dir(self, benchmark_name: str, ex_version: str) -> Path:
        path = self.absolute_path / "results" / "correctness" / benchmark_name / ex_version
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _correctness_output_dir(self, benchmark_name: str, ex_version: str) -> Path:
        path = self.absolute_path / ex_version / benchmark_name / "EX5_correctness"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _detect_output_suffix(self, benchmark_name: str, ex_version: str) -> str:
        key = (ex_version, benchmark_name)
        cached = self._suffix_cache.get(key)
        if cached:
            return cached
        from scripts.checkers.output_specs import get_output_format

        fmt = get_output_format(benchmark_name)
        if fmt and fmt.suffix:
            suffix = fmt.suffix
        else:
            suffix = ".txt"
            input_dir = self.absolute_path / ex_version / benchmark_name / "input_data"
            candidates: List[Path] = []
            if input_dir.exists():
                files = [p for p in input_dir.iterdir() if p.is_file()]
                preferred = [p for p in files if "output" in p.name.lower()]
                candidates = sorted(preferred or files)
            if candidates:
                candidate_suffix = candidates[0].suffix or ""
                if candidate_suffix:
                    suffix = candidate_suffix
        self._suffix_cache[key] = suffix
        return suffix

    def _correctness_output_path(self,
                                 benchmark_name: str,
                                 ex_version: str,
                                 suite: str,
                                 entry: Dict[str, Any],
                                 suffix: str) -> Path:
        target_dir = self._correctness_output_dir(benchmark_name, ex_version)
        model = entry["model"] or "baseline"
        dataset = entry.get("dataset") or DEFAULT_DATASET
        version = entry.get("version")
        if entry.get("is_baseline"):
            return Path(os.devnull)
        version_label = f"v{version}" if version is not None else "baseline"
        filename = f"output_{model}_{version_label}_{dataset}_output{suffix}"
        return target_dir / filename

    STDOUT_REDIRECT_BENCHMARKS = {
        "basicmath-mibench",
        "bitcount-mibench",
        "qsort-mibench",
    }

    def _benchmark_args(
        self,
        benchmark_name: str,
        dataset: str,
        *,
        output_path: Optional[Path] = None,
        ex_version: Optional[str] = None,
    ) -> Tuple[List[str], bool]:
        # Choose the appropriate config based on ex_version
        if ex_version and ex_version.upper() == "EX3":
            arg_map = BENCHMARK_ARG_MAP_EX3
        else:
            arg_map = BENCHMARK_ARG_MAP
        
        config = arg_map.get(benchmark_name)
        if not config:
            return [], False
        dataset_key = (dataset or DEFAULT_DATASET).lower()
        base_args = config.get(dataset_key) or config.get("default", [])
        args = list(base_args) if base_args else []
        replaced = False
        if benchmark_name in self.STDOUT_REDIRECT_BENCHMARKS and args:
            args = [token for token in args if token != "/dev/null"]
            replaced = False
        elif output_path and str(output_path) != os.devnull and args:
            args, replaced = self._override_output_placeholders(args, output_path)
        return args, replaced

    def _override_output_placeholders(
        self,
        args: List[str],
        output_path: Path,
    ) -> Tuple[List[str], bool]:
        """
        Replace any '/dev/null' placeholder in benchmark args with the real capture output path.
        This keeps configs declarative while ensuring correctness-run produces tangible artifacts.
        """
        if not args:
            return [], False
        target = str(output_path)
        replaced = False
        new_args: List[str] = []
        for token in args:
            if token == "/dev/null":
                new_args.append(target)
                replaced = True
            else:
                new_args.append(token)
        return new_args, replaced

    def _parse_executable_entry(self,
                                exe_path: Path,
                                benchmark_name: str,
                                default_dataset: str,
                                expect_dataset: bool) -> Optional[Dict[str, Any]]:
        original_name = exe_path.name
        version = None
        remainder = original_name
        if remainder.startswith(benchmark_name):
            remainder = remainder[len(benchmark_name):]
            if remainder.startswith("_"):
                remainder = remainder[1:]
        raw_tokens = remainder.split("_") if remainder else []
        tokens: List[str] = []
        for token in raw_tokens:
            if token.startswith("v") and token[1:].isdigit():
                version = int(token[1:])
                continue
            tokens.append(token)

        dataset = default_dataset
        compiler = "unknown"
        if expect_dataset:
            if tokens:
                dataset = tokens.pop()
            if tokens and tokens[-1] in KNOWN_COMPILERS:
                compiler = tokens.pop()
            else:
                compiler = "gcc"
        else:
            if tokens and tokens[-1] in KNOWN_COMPILERS:
                compiler = tokens.pop()
            elif tokens and tokens[0] in KNOWN_COMPILERS:
                compiler = tokens.pop(0)
        model = "_".join(tokens) or "baseline"
        is_baseline = (model == "baseline")
        model = self._normalize_model_name(model)
        return {
            "exe_path": exe_path,
            "benchmark": benchmark_name,
            "dataset": dataset,
            "compiler": compiler,
            "model": model or "baseline",
            "version": version,
            "is_baseline": is_baseline,
            "workdir": exe_path.parent,
        }

    def _normalize_model_name(self, model: Optional[str]) -> Optional[str]:
        if not model:
            return model
        tokens = model.split("_")
        if tokens and tokens[0] in KNOWN_COMPILERS:
            trimmed = "_".join(tokens[1:]) if len(tokens) > 1 else ""
            return trimmed or tokens[0]
        return model

    def _collect_capture_executables(self,
                                     ex_version: str,
                                     benchmark_name: str,
                                     models: Optional[List[str]] = None,
                                     datasets: Optional[List[str]] = None,
                                     versions: Optional[List[int]] = None,
                                     versions_all: bool = False) -> List[Dict[str, Any]]:
        opt_dir = self._optimized_dir(ex_version, benchmark_name)
        if not opt_dir.exists():
            raise FileNotFoundError(f"{opt_dir} does not exist. Please build the benchmark first.")
        expect_dataset = benchmark_name in self.polybench_list
        default_dataset = DEFAULT_DATASET
        model_filter = [m.lower() for m in models] if models else None
        dataset_filter = [d.lower() for d in datasets] if datasets else None
        dataset_values = datasets if datasets else None
        version_filter = set(versions) if (versions and not versions_all) else None

        entries: List[Dict[str, Any]] = []
        for item in sorted(opt_dir.iterdir()):
            if not item.is_file():
                continue
            if not os.access(item, os.X_OK):
                # Skip sources or intermediate artifacts (e.g., *.c) that are not executable.
                continue
            suffix = item.suffix
            allow_suffix = "gpt5.1" in item.name
            if suffix and not allow_suffix:
                continue
            entry = self._parse_executable_entry(item, benchmark_name, default_dataset, expect_dataset)
            if not entry:
                continue
            model_name = entry["model"].lower()
            if model_filter and model_name not in model_filter:
                continue
            datasets_to_use = (
                [entry["dataset"]] if expect_dataset else (dataset_values or [default_dataset])
            )
            for ds in datasets_to_use:
                ds_lower = ds.lower()
                if dataset_filter and ds_lower not in dataset_filter:
                    continue
                clone = entry.copy()
                clone["dataset"] = ds
                clone["benchmark_dir"] = self.absolute_path / ex_version / benchmark_name
                if version_filter is not None and not clone.get("is_baseline"):
                    if clone.get("version") not in version_filter:
                        continue
                entries.append(clone)
        return entries

    def _execute_and_capture(self,
                             entry: Dict[str, Any],
                             output_path: Path,
                             args: Optional[List[str]] = None,
                             writes_to_output_file: bool = False) -> Dict[str, Any]:
        exe_path = entry["exe_path"]
        benchmark_dir = entry.get("benchmark_dir")
        workdir = benchmark_dir or entry.get("workdir") or exe_path.parent
        env = os.environ.copy()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if writes_to_output_file and output_path.exists():
            try:
                output_path.unlink()
            except OSError:
                pass
        try:
            if writes_to_output_file:
                completed = subprocess.run(
                    [str(exe_path)] + (args or []),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=str(workdir),
                    check=False,
                    env=env,
                )
            else:
                with output_path.open("w", encoding="utf-8") as stdout_fp:
                    completed = subprocess.run(
                        [str(exe_path)] + (args or []),
                        stdout=stdout_fp,
                        stderr=subprocess.PIPE,
                        text=True,
                        cwd=str(workdir),
                        check=False,
                        env=env,
                    )
            return {"returncode": completed.returncode, "stderr": completed.stderr}
        except Exception as exc:  # pylint: disable=broad-except
            return {"returncode": None, "stderr": str(exc)}

    def _append_capture_log(self,
                            log_path: Path,
                            entry: Dict[str, Any],
                            output_path: Path,
                            result: Dict[str, Any]) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "stage": "capture",
            "executable": entry["exe_path"].name,
            "output": str(output_path),
            "dataset": entry["dataset"],
            "model": entry["model"],
            "version": entry.get("version"),
            "status": "ok" if result["returncode"] == 0 else "error",
            "returncode": result["returncode"],
            "stderr": (result["stderr"] or "").strip(),
        }
        with log_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    def capture_correctness_outputs(self,
                                    benchmark_name: str,
                                    ex_versions: List[str],
                                    *,
                                    models: Optional[List[str]] = None,
                                    datasets: Optional[List[str]] = None,
                                    versions: Optional[List[int]] = None,
                                    versions_all: bool = False) -> None:
        suite = self._benchmark_suite(benchmark_name)
        model_filter = [m.lower() for m in models] if models else None
        dataset_filter = [d.lower() for d in datasets] if datasets else None
        for ex_version in ex_versions:
            try:
                executables = self._collect_capture_executables(
                    ex_version,
                    benchmark_name,
                    models=model_filter,
                    datasets=dataset_filter,
                    versions=versions,
                    versions_all=versions_all,
                )
            except FileNotFoundError as missing:
                self._log_capture_error(
                    benchmark=benchmark_name,
                    ex_version=ex_version,
                    message=str(missing),
                    level="WARN",
                )
                continue

            if not executables:
                self._log_capture_error(
                    benchmark=benchmark_name,
                    ex_version=ex_version,
                    message="No executables matched the provided filters.",
                    level="WARN",
                )
                continue

            suffix = self._detect_output_suffix(benchmark_name, ex_version)
            capture_log_path = self._correctness_results_dir(benchmark_name, ex_version) / "correctness.jsonl"
            success = 0
            failure = 0
            for entry in executables:
                output_path = self._correctness_output_path(
                    benchmark_name,
                    ex_version,
                    suite,
                    entry,
                    suffix,
                )
                args, writes_to_output_file = self._benchmark_args(
                    benchmark_name,
                    entry["dataset"],
                    output_path=output_path,
                    ex_version=ex_version,
                )
                print(
                    f"[INFO] <correctness-run> Running {entry['exe_path'].name} "
                    f"(dataset={entry['dataset']}, model={entry['model']}, version={entry.get('version')}) "
                    f"-> {output_path}"
                )
                self._log_capture_activity(
                    "capture_start",
                    benchmark=benchmark_name,
                    ex_version=ex_version,
                    executable=str(entry["exe_path"]),
                    output=str(output_path),
                )
                result = self._execute_and_capture(
                    entry,
                    output_path,
                    args,
                    writes_to_output_file=writes_to_output_file,
                )
                self._append_capture_log(capture_log_path, entry, output_path, result)
                if result["returncode"] == 0:
                    success += 1
                    self._log_capture_activity(
                        "capture_complete",
                        benchmark=benchmark_name,
                        ex_version=ex_version,
                        executable=str(entry["exe_path"]),
                        output=str(output_path),
                        status="ok",
                    )
                    print(
                        f"[INFO] <correctness-run> Completed {entry['exe_path'].name} "
                        f"(dataset={entry['dataset']}, model={entry['model']}) ✔"
                    )
                else:
                    failure += 1
                    self._log_capture_error(
                        benchmark=benchmark_name,
                        ex_version=ex_version,
                        message=f"Execution failed for {entry['exe_path'].name}: {result['stderr']}",
                        level="ERROR",
                    )
                    self._log_capture_activity(
                        "capture_complete",
                        benchmark=benchmark_name,
                        ex_version=ex_version,
                        executable=str(entry["exe_path"]),
                        output=str(output_path),
                        status="error",
                        stderr=result["stderr"],
                    )
                    print(
                        f"[ERROR] <correctness-run> {entry['exe_path'].name} failed "
                        f"(rc={result['returncode']}): {result['stderr']}"
                    )
            print(
                f"[INFO] Correctness capture for {benchmark_name} ({ex_version}) finished: "
                f"{success} success, {failure} failed. Outputs in "
                f"{self._correctness_output_dir(benchmark_name, ex_version)}"
            )

    def _write_fallback_artifact(self,
                                 ex_version: str,
                                 benchmark_name: str,
                                 attempt: Optional[int],
                                 response_text: str,
                                 target_index: int = 0):
        base_dir = self._optimized_dir(ex_version, benchmark_name)
        base_dir.mkdir(parents=True, exist_ok=True)
        stem, _ = self._target_basename_and_suffix(
            benchmark_name,
            target_index=target_index,
            ex_version=ex_version,
        )
        version_tag = f"_v{attempt}" if attempt is not None else ""
        fallback_path = base_dir / f"{stem}_{self.llm_abbrev}{version_tag}.txt"
        fallback_path.write_text(response_text, encoding="utf-8")
        self.log_activity(
            "fallback_saved",
            ex_version=ex_version,
            benchmark=benchmark_name,
            attempt=attempt,
            path=str(fallback_path),
        )
        self.log_error(
            ex_version=ex_version,
            benchmark=benchmark_name,
            model=self.llm_model,
            path=str(fallback_path),
            error=RuntimeError("LLM response did not match expected code format; saved raw text."),
            level="WARN",
        )
        return fallback_path

    def _benchmark_has_multiple_targets(self, benchmark_name: str, *, ex_version: Optional[str] = None) -> bool:
        spec = self._target_spec(benchmark_name, ex_version=ex_version)
        return isinstance(spec, (list, tuple))

    def _collect_target_contexts(self, ex_version: str, benchmark_name: str) -> List[Dict[str, Any]]:
        metadata = self._get_benchmark_metadata(benchmark_name, ex_version)
        target_files = metadata.get("target_file")
        start_lines = metadata.get("start_line")
        end_lines = metadata.get("end_line")
        if target_files is None or start_lines is None or end_lines is None:
            raise ValueError(f"[auto_run_{ex_version}] Missing metadata for {benchmark_name}.")

        if isinstance(target_files, (list, tuple)):
            files = list(target_files)
        else:
            files = [target_files]

        if isinstance(start_lines, (list, tuple)):
            starts = list(start_lines)
        else:
            starts = [start_lines]

        if isinstance(end_lines, (list, tuple)):
            ends = list(end_lines)
        else:
            ends = [end_lines]

        if not (len(files) == len(starts) == len(ends)):
            raise ValueError(
                f"[auto_run_{ex_version}] Inconsistent metadata lengths for {benchmark_name}: "
                f"files={len(files)}, starts={len(starts)}, ends={len(ends)}."
            )

        contexts: List[Dict[str, Any]] = []
        optimized_dir = self._optimized_dir(ex_version, benchmark_name)
        for idx, file_name in enumerate(files):
            path = optimized_dir / file_name
            try:
                with path.open("r", encoding="utf-8") as handle:
                    lines = handle.readlines()
            except FileNotFoundError as missing:
                raise FileNotFoundError(
                    f"[auto_run_{ex_version}] Unable to read target file {path}"
                ) from missing
            contexts.append(
                {
                    "index": idx,
                    "path": path,
                    "file_name": file_name,
                    "lines": lines,
                    "start_line": int(starts[idx]),
                    "end_line": int(ends[idx]),
                }
            )
        return contexts

    def _compose_prompt_user(self,
                             *,
                             head_content: str,
                             instruction_block: str,
                             file_name: str,
                             target_code: str,
                             previous_outputs: Sequence[Dict[str, str]]) -> str:
        prompt = (
            f"Header file content (macros and definitions) or related information: \n{head_content}\n\n"
            f"Target file: {file_name}\n"
            f"Input code:\n{target_code}\n\n"
            f"{instruction_block}"
        )
        if previous_outputs:
            prompt += (
                "\n\nFor consistency with other files optimized in this attempt, review the following generated code snippets:\n"
            )
            for artifact in previous_outputs:
                prompt += (
                    f"\nFile {artifact['file_name']} optimized earlier in this attempt:\n"
                    "<<<REFERENCE-CODE>>>\n"
                    f"{artifact['code']}\n"
                    "<<<END-REFERENCE-CODE>>>\n"
                )
            prompt += "\nEnsure the new code stays compatible with the referenced files."
        return prompt

    def _run_multi_target_attempts(self,
                                   *,
                                   ex_version: str,
                                   benchmark_name: str,
                                   head_content: str,
                                   prompt_system: str,
                                   instruction_block: str,
                                   benchmark_dir: Path,
                                   explanation_output_path: Path,
                                   max_attempts: int) -> int:
        canonical = canonical_ex_version(ex_version) or ex_version
        contexts = self._collect_target_contexts(ex_version, benchmark_name)
        success_count = 0
        for attempt in range(1, max_attempts + 1):
            primary_output_hint = self._build_output_path(ex_version, benchmark_name, attempt, target_index=0)
            self.log_activity(
                "attempt_start",
                ex_version=ex_version,
                benchmark=benchmark_name,
                attempt=attempt,
                target_path=str(primary_output_hint),
            )
            previous_outputs: List[Dict[str, str]] = []
            attempt_failed = False

            for ctx in contexts:
                target_code = "".join(ctx["lines"][ctx["start_line"] - 1:ctx["end_line"]])
                if canonical == "EX1":
                    self._guard_payload("Source snippet", target_code, benchmark_name, "EX2")
                    self._guard_payload("Source snippet", target_code, benchmark_name, "EX1")
                prompt_user = self._compose_prompt_user(
                    head_content=head_content,
                    instruction_block=instruction_block,
                    file_name=ctx["file_name"],
                    target_code=target_code,
                    previous_outputs=previous_outputs,
                )
                if self.model_family == 'hpc-coder':
                    print(
                        f"[INFO] <auto_run_{ex_version}> Using HPC-Coder model: {self.llm_model} "
                        f"to try [attempt={attempt}, file={ctx['file_name']}]"
                    )
                response_text = self._generate_llm_response(
                    ex_version=ex_version,
                    prompt_system=prompt_system,
                    prompt_user=prompt_user,
                    attempt=attempt,
                )
                if self.model_family == 'hpc-coder':
                    self._persist_hpc_response(
                        benchmark_dir,
                        benchmark_name,
                        response_text,
                        ex_version,
                        variant_label=ctx["file_name"].replace(".", "_"),
                    )
                    print(response_text, flush=True)

                optimized_code, explanation_content, has_explanation = self.process_response_text(
                    response_text,
                    benchmark_name,
                    ex_version,
                )
                if optimized_code is None:
                    self._write_fallback_artifact(
                        ex_version,
                        benchmark_name,
                        attempt,
                        response_text,
                        target_index=ctx["index"],
                    )
                    attempt_failed = True
                    break

                new_c_content = (
                    ctx["lines"][:ctx["start_line"] - 1] + [optimized_code + "\n"] + ctx["lines"][ctx["end_line"]:]
                )
                output_path = self._build_output_path(
                    ex_version,
                    benchmark_name,
                    attempt,
                    target_index=ctx["index"],
                )
                with output_path.open("w", encoding="utf-8") as file:
                    file.writelines(new_c_content)

                if self.write_state and has_explanation and explanation_content:
                    explanation_lines = explanation_content.split("\n")
                    explanation_comment = "/**\n" + "\n".join(f" * {line.strip()}" for line in explanation_lines) + "\n */\n"
                    write_mode = "w" if not explanation_output_path.exists() else "a"
                    with explanation_output_path.open(write_mode, encoding="utf-8") as file:
                        file.write("\n" + "=" * 80 + "\n")
                        file.write(
                            f"<<< {ex_version} >>> <{self.llm_abbrev}[v_{attempt}]> "
                            f"Optimization Explanation for {ctx['file_name']}:\n"
                        )
                        file.write(explanation_comment + "\n")

                previous_outputs.append(
                    {
                        "file_name": ctx["file_name"],
                        "code": optimized_code,
                    }
                )
                self.log_activity(
                    "artifact_saved",
                    ex_version=ex_version,
                    benchmark=benchmark_name,
                    attempt=attempt,
                    path=str(output_path),
                    target_file=ctx["file_name"],
                )

            if attempt_failed:
                continue

            success_count += 1
            if previous_outputs:
                self.optimized_codes[benchmark_name] = previous_outputs[-1]["code"]
            print(f"[INFO] <auto_run_{ex_version}> [{benchmark_name}] Optimization completed for v{attempt}.")
            print("=" * 80, flush=True)

        return success_count

    def _estimate_tokens(self, text: str) -> int:
        return max(1, math.ceil(len(text) / 4))

    def _guard_payload(self, label: str, text: str, benchmark_name: str, ex_version: str):
        if not self.llm_guard_enabled:
            return
        approx = self._estimate_tokens(text)
        if approx > self.llm_guard_limit:
            self.log_activity(
                "guard_violation",
                label=label,
                ex_version=ex_version,
                benchmark=benchmark_name,
                approx_tokens=approx,
                limit=self.llm_guard_limit,
            )
            raise ValueError(
                f"{label} for {benchmark_name} ({ex_version}) is too long "
                f"(~{approx} tokens > guard {self.llm_guard_limit})."
            )

    def _respect_rate_limits(self, ex_version: str):
        return

    def _llm_params(self, ex_version: str) -> Dict[str, Any]:
        params = {"temperature": 1.0, "max_tokens": None, "stream": False}
        if self.model_family == "together.ai":
            params["temperature"] = 1.0
            params["max_tokens"] = None
        elif self.model_family == "openai":
            params["temperature"] = 1.0
        elif self.model_family == "claude":
            params["temperature"] = 0.5
            params["max_tokens"] = None
            params["stream"] = True
        elif self.model_family == "hpc-coder":
            params["temperature"] = 0.0
            params["max_tokens"] = None
        return params

    def _generate_llm_response(
        self,
        ex_version: str,
        prompt_system: str,
        prompt_user: str,
        attempt: Optional[int] = None,
    ) -> str:
        self._respect_rate_limits(ex_version)
        params = self._llm_params(ex_version)
        response_text = self.llm_client.generate(
            prompt_system,
            prompt_user,
            temperature=params["temperature"],
            max_output_tokens=params["max_tokens"],
            stream=params["stream"],
            context={"attempt": attempt, "ex_version": ex_version} if attempt is not None else {"ex_version": ex_version},
        )
        return response_text

    def _persist_hpc_response(self, benchmark_dir: Path, benchmark_name: str, response_text: str, ex_version: str, variant_label: Optional[str] = None):
        suffix = f"_{variant_label}" if variant_label else ""
        hpc_txt = f"{benchmark_name}_{self.llm_abbrev}{suffix}.txt"
        hpc_txt_path = benchmark_dir / hpc_txt
        print(f"[INFO] <auto_run_{ex_version}> [HPC-Coder] Start to save")
        hpc_txt_path.write_text(response_text, encoding="utf-8")
        print(f"[INFO] <auto_run_{ex_version}> [HPC-Coder] Response text has been saved to {hpc_txt_path}")

    def auto_run_EX1(self, benchmark_name: str, max_attempts: int = 10, ex_version: str = "EX1"):
        """Automatically runs optimization for the EX1 benchmark for type 1 experiment."""
        ex_version = ex_version or "EX1"
        canonical_ex = canonical_ex_version(ex_version) or "EX1"
        try:
            success_count = 0
            metadata = self._get_benchmark_metadata(benchmark_name, canonical_ex)
            if not metadata:
                raise ValueError(f"[ERROR] <auto_run_EX1> Unknown benchmark: {benchmark_name}. Please define metadata.")
            
            head_code_paths = resolve_head_code_paths(self, benchmark_name)

            # Read all header files (missing files yield empty content)
            placeholder_path = self.absolute_path / "utilities" / ".empty_header_placeholder"
            if not head_code_paths:
                placeholder_path.parent.mkdir(parents=True, exist_ok=True)
                if not placeholder_path.exists():
                    placeholder_path.write_text("", encoding="utf-8")
                head_code_paths = [placeholder_path]

            head_contents = []
            for path in head_code_paths:
                head_contents.append(f"The following is the content of {path.name}:")
                try:
                    with path.open("r", encoding="utf-8") as file:
                        head_contents.append(file.read())
                except FileNotFoundError as missing:
                    self.log_error(
                        ex_version=ex_version,
                        benchmark=benchmark_name,
                        model=self.llm_model,
                        path=str(path),
                        error=missing,
                        level="WARN"
                    )
                    head_contents.append("")
            head_content = "\n".join(head_contents)
            self._guard_payload("Header payload", head_content, benchmark_name, "EX2")
            self._guard_payload("Header payload", head_content, benchmark_name, canonical_ex)

            print(f"[INFO] <auto_run_EX1> head_code_paths = {head_code_paths}", flush=True)
            
            benchmark_dir = self.absolute_path / ex_version / benchmark_name
            optimized_dir = self._optimized_dir(ex_version, benchmark_name)
            target_spec = metadata["target_file"]
            is_multi_target = self._benchmark_has_multiple_targets(benchmark_name, ex_version=canonical_ex)
            if is_multi_target:
                run_target_display = [optimized_dir / Path(path) for path in target_spec]
            else:
                single_target = target_spec[0] if isinstance(target_spec, (list, tuple)) else target_spec
                run_target_display = optimized_dir / single_target
            sample_output_path = self._build_output_path(ex_version, benchmark_name, 1)
            explanation_dir = self.absolute_path / "generation_explainations"
            explanation_dir.mkdir(parents=True, exist_ok=True)
            explanation_output_path = explanation_dir / f"{ex_version}_optimization_explanations_{self.llm_abbrev}.txt"

            print(f"[INFO] <auto_run_EX1> Running {self.model_family} on {run_target_display}")
            print(f"[INFO] <auto_run_EX1> Saving optimized code to {sample_output_path}")
            print(f"[INFO] <auto_run_EX1> Saving explanation to {explanation_output_path}")
            print(f"[INFO] <auto_run_EX1> Using LLM model: {self.llm_model}")
            prompt_system_EX1 = (
                "You are a code generation/optimization assistant. Your task is to take a prompt and input code and "
                "generate an optimized version of the code. "
                "Your output must only be compilable source code without explanations. "
                "The computation environment is a Linux system (Enterprise Linux 8, kernel version 4.18.0-348.7.1) and "
                "a single AMD EPYC 7543 32-Core CPU. "
                "The available C/C++ compilers are: GCC/G++ v14.2.0 and CLANG/CLANG++ v19.1.3."
            )
            instruction_block_EX1 = (
                "Provide the optimized C code with serial optimizations for above input code, ensuring that none of the existing functions "
                "or header files are removed, and no new functions or print statements are added. "
                "Wrap the generated optimized C code between <<<CODE>>> and <<<END-CODE>>>. Then provide an explanation of why this optimization was made, "
                "and wrap the explanation between <<<NOTE>>> and <<<END-NOTE>>>. "
                "Ensure that the optimized code contains only compilable source code, and the explanation contains only plain text or code snippets as needed. "
                "Do not emit any extra commentary before or after these markers. "
                "Example format:\n"
                "<<<CODE>>>\n"
                "static void foo() {\n"
                "  // optimized code body\n"
                "}\n"
                "<<<END-CODE>>>\n"
                "<<<NOTE>>>\n"
                "Explain the optimization rationale here.\n"
                "<<<END-NOTE>>>"
            )

            if is_multi_target:
                success_count = self._run_multi_target_attempts(
                    ex_version=ex_version,
                    benchmark_name=benchmark_name,
                    head_content=head_content,
                    prompt_system=prompt_system_EX1,
                    instruction_block=instruction_block_EX1,
                    benchmark_dir=benchmark_dir,
                    explanation_output_path=explanation_output_path,
                    max_attempts=max_attempts,
                )
            else:
                start_meta = metadata["start_line"]
                end_meta = metadata["end_line"]
                start_line = int(start_meta[0] if isinstance(start_meta, (list, tuple)) else start_meta)
                end_line = int(end_meta[0] if isinstance(end_meta, (list, tuple)) else end_meta)
                print(f"[INFO] <auto_run_EX1> Benchmark start line: {start_line}, end line: {end_line}")

                c_file_path = optimized_dir / single_target
                with c_file_path.open("r", encoding="utf-8") as file:
                    c_content = file.readlines()
                target_code = "".join(c_content[start_line - 1:end_line])
                self._guard_payload("Source snippet", target_code, benchmark_name, "EX2")
                self._guard_payload("Source snippet", target_code, benchmark_name, canonical_ex)

                prompt_user_EX1 = (
                    f"Header file content (macros and definitions) or related information: \n{head_content}\n\n"
                    f"Input code:\n{target_code}\n\n"
                    f"{instruction_block_EX1}"
                )
                for i in range(1, max_attempts + 1):
                    output_c_file_path = self._build_output_path(ex_version, benchmark_name, i)
                    self.log_activity(
                        "attempt_start",
                        ex_version=ex_version,
                        benchmark=benchmark_name,
                        attempt=i,
                        target_path=str(output_c_file_path),
                    )
                    if self.model_family == 'hpc-coder':
                        print(f"[INFO] <auto_run_EX1> Using HPC-Coder model: {self.llm_model} to try [{i} times]")
                    response_text = self._generate_llm_response(
                        ex_version=ex_version,
                        prompt_system=prompt_system_EX1,
                        prompt_user=prompt_user_EX1,
                        attempt=i,
                    )
                    if self.model_family == 'hpc-coder':
                        self._persist_hpc_response(benchmark_dir, benchmark_name, response_text, ex_version)
                        print(response_text, flush=True)

                    optimized_code, explanation_content, has_explanation = self.process_response_text(
                        response_text,
                        benchmark_name,
                        canonical_ex,
                    )
                    if optimized_code is None:
                        self._write_fallback_artifact(ex_version, benchmark_name, i, response_text)
                        continue

                    self.optimized_codes[benchmark_name] = optimized_code

                    if has_explanation and explanation_content:
                        explanation_lines = explanation_content.split("\n")
                        explanation_comment = "/**\n" + "\n".join(f" * {line.strip()}" for line in explanation_lines) + "\n */\n"
                    else:
                        explanation_comment = ""

                    new_c_content = (
                        c_content[:start_line - 1] + [optimized_code + "\n"] + c_content[end_line:]
                    )

                    with output_c_file_path.open("w", encoding="utf-8") as file:
                        file.writelines(new_c_content)

                    if self.write_state and has_explanation and explanation_content:
                        write_mode = "w" if not explanation_output_path.exists() else "a" 
                        with explanation_output_path.open(write_mode, encoding="utf-8") as file:
                            file.write("\n" + "=" * 80 + "\n")
                            file.write(f"<<< {canonical_ex} >>> <{self.llm_abbrev}[v_{i}]> Optimization Explanation for {c_file_path.name}:\n")
                            file.write(explanation_comment + "\n")

                    success_count += 1
                    print(f"[INFO] <auto_run_EX1> [{benchmark_name}] Optimization completed for v{i}.")
                    print("=" * 80, flush=True)
                    self.log_activity(
                        "artifact_saved",
                        ex_version=ex_version,
                        benchmark=benchmark_name,
                        attempt=i,
                        path=str(output_c_file_path),
                    )

            if success_count == 0:
                warn_msg = RuntimeError(f"All attempts failed for {benchmark_name} in {ex_version}.")
                self.log_error(
                    ex_version=ex_version,
                    benchmark=benchmark_name,
                    model=self.llm_model,
                    path=str(self._optimized_dir(ex_version, benchmark_name)),
                    error=warn_msg,
                    level="WARN"
                )
                print(
                    f"[WARN] <auto_run_EX1> Unable to produce compilable output for {benchmark_name} "
                    f"in {ex_version} after {max_attempts} attempts."
                )
        except Exception as e:
            # Use log_error to record the error
            self.log_error(
                ex_version=ex_version,
                benchmark=benchmark_name,
                model=self.llm_model,
                path=str(self.log_file_path),
                error=e,
                level="ERROR"
            )
            
            # Write the error to the log file
            # Optional: Print the error message to the console
            print(
                f"[ERROR] <auto_run_EX1> has Exception - Benchmark: {benchmark_name}, "
                f"EX version: {ex_version}, Error: {str(e)}"
            )

    def auto_run_EX2(self, benchmark_name: str, max_attempts: int = 10, ex_version: str = "EX2"):
        """Automatically runs optimization for the EX2 benchmark for type 1 experiment."""
        ex_version = ex_version or "EX2"
        canonical_ex = canonical_ex_version(ex_version) or "EX2"
        try:
            success_count = 0
            metadata = self._get_benchmark_metadata(benchmark_name, canonical_ex)
            if not metadata:
                raise ValueError(f"[ERROR] <auto_run_EX2> Unknown benchmark: {benchmark_name}. Please define metadata.")
            
            head_code_paths = resolve_head_code_paths(self, benchmark_name)

            # Read all header files (missing files yield empty content)
            placeholder_path = self.absolute_path / "utilities" / ".empty_header_placeholder"
            if not head_code_paths:
                placeholder_path.parent.mkdir(parents=True, exist_ok=True)
                if not placeholder_path.exists():
                    placeholder_path.write_text("", encoding="utf-8")
                head_code_paths = [placeholder_path]

            head_contents = []
            for path in head_code_paths:
                head_contents.append(f"The following is the content of {path.name}:")
                try:
                    with path.open("r", encoding="utf-8") as file:
                        head_contents.append(file.read())
                except FileNotFoundError as missing:
                    self.log_error(
                        ex_version=ex_version,
                        benchmark=benchmark_name,
                        model=self.llm_model,
                        path=str(path),
                        error=missing,
                        level="WARN"
                    )
                    head_contents.append("")
            head_content = "\n".join(head_contents)

            benchmark_dir = self.absolute_path / ex_version / benchmark_name
            optimized_dir = self._optimized_dir(ex_version, benchmark_name)
            target_spec = metadata["target_file"]
            is_multi_target = self._benchmark_has_multiple_targets(benchmark_name, ex_version=canonical_ex)
            if is_multi_target:
                run_target_display = [optimized_dir / Path(path) for path in target_spec]
            else:
                single_target = target_spec[0] if isinstance(target_spec, (list, tuple)) else target_spec
                run_target_display = optimized_dir / single_target

            print(f"[INFO] <auto_run_EX2> head_code_paths = {head_code_paths}", flush=True)
            print(f"[INFO] <auto_run_EX2> Running {self.model_family} on {run_target_display}")

            sample_output_path = self._build_output_path(ex_version, benchmark_name, 1)
            explanation_dir = self.absolute_path / "generation_explainations"
            explanation_dir.mkdir(parents=True, exist_ok=True)
            explanation_output_path = explanation_dir / f"{ex_version}_optimization_explanations_{self.llm_abbrev}.txt"

            print(f"[INFO] <auto_run_EX2> Saving optimized code to {sample_output_path}")
            print(f"[INFO] <auto_run_EX2> Saving explanation to {explanation_output_path}")
            print(f"[INFO] <auto_run_EX2> Using LLM model: {self.llm_model}")
            prompt_system_EX2 = (
                "You are a code generation/optimization assistant. Your task is to take a prompt and input code and "
                "generate an optimized version of the code. "
                "Your output must only be compilable source code without explanations. "
                "The computation environment is a Linux system (Enterprise Linux 8, kernel version 4.18.0-348.7.1) and "
                "a single AMD EPYC 7543 32-Core CPU. "
                "The available C/C++ compilers are: GCC/G++ v14.2.0 and CLANG/CLANG++ v19.1.3."
            )
            instruction_block_EX2 = (
                "Provide the optimized C code with both serial and parallel optimizations, ensuring that none of the existing functions "
                "or header files are removed, and no new functions or print statements are added. "
                "The file already contains an `_OPENMP`-guarded `#include <omp.h>`; keep it intact and feel free to use OpenMP pragmas/API directly, "
                "but never insert `omp_set_num_threads` (thread counts come from the runtime via `OMP_NUM_THREADS`). "
                "Do not modify, relocate, or duplicate any time-measurement instrumentation (`clock_gettime`, `omp_get_wtime`, kernel-time accumulators, or related globals); "
                "if you restructure control flow, ensure the function still reaches the existing end-of-function timing accumulation (avoid early returns by storing results first). "
                "Avoid adding printf/logging inside parallel regions so multi-threaded output stays deterministic. "
                "Wrap the generated optimized C code between <<<CODE>>> and <<<END-CODE>>>. Then provide an explanation of why this optimization was made, "
                "and wrap the explanation between <<<NOTE>>> and <<<END-NOTE>>>. "
                "Ensure that the optimized code contains only compilable source code, and the explanation contains only plain text or code snippets as needed. "
                "Do not emit anything outside those markers. "
                "Example:\n"
                "<<<CODE>>>\n"
                "/* optimized code */\n"
                "<<<END-CODE>>>\n"
                "<<<NOTE>>>\n"
                "Explain the optimization rationale here.\n"
                "<<<END-NOTE>>>"
            )

            if is_multi_target:
                success_count = self._run_multi_target_attempts(
                    ex_version=ex_version,
                    benchmark_name=benchmark_name,
                    head_content=head_content,
                    prompt_system=prompt_system_EX2,
                    instruction_block=instruction_block_EX2,
                    benchmark_dir=benchmark_dir,
                    explanation_output_path=explanation_output_path,
                    max_attempts=max_attempts,
                )
            else:
                start_meta = metadata["start_line"]
                end_meta = metadata["end_line"]
                start_line = int(start_meta[0] if isinstance(start_meta, (list, tuple)) else start_meta)
                end_line = int(end_meta[0] if isinstance(end_meta, (list, tuple)) else end_meta)
                print(f"[INFO] <auto_run_EX2> Benchmark start line: {start_line}, end line: {end_line}")

                c_file_path = optimized_dir / single_target
                with c_file_path.open("r", encoding="utf-8") as file:
                    c_content = file.readlines()
                target_code = "".join(c_content[start_line - 1:end_line])

                prompt_user_EX2 = (
                    f"Header file content (macros and definitions) or related information: \n{head_content}\n\n"
                    f"Input code:\n{target_code}\n\n"
                    f"{instruction_block_EX2}"
                )

                for i in range(1, max_attempts + 1):
                    output_c_file_path = self._build_output_path(ex_version, benchmark_name, i)
                    self.log_activity(
                        "attempt_start",
                        ex_version=ex_version,
                        benchmark=benchmark_name,
                        attempt=i,
                        target_path=str(output_c_file_path),
                    )
                    if self.model_family == 'hpc-coder':
                        print(f"[INFO] <auto_run_EX2> Using HPC-Coder model: {self.llm_model} to try [{i} times]")
                    response_text = self._generate_llm_response(
                        ex_version=ex_version,
                        prompt_system=prompt_system_EX2,
                        prompt_user=prompt_user_EX2,
                        attempt=i,
                    )
                    if self.model_family == 'hpc-coder':
                        self._persist_hpc_response(benchmark_dir, benchmark_name, response_text, ex_version)
                        print(response_text, flush=True)

                    optimized_code, explanation_content, has_explanation = self.process_response_text(
                        response_text,
                        benchmark_name,
                        canonical_ex,
                    )
                    if optimized_code is None:
                        self._write_fallback_artifact(ex_version, benchmark_name, i, response_text)
                        continue

                    self.optimized_codes[benchmark_name] = optimized_code

                    if has_explanation and explanation_content:
                        explanation_lines = explanation_content.split("\n")
                        explanation_comment = "/**\n" + "\n".join(f" * {line.strip()}" for line in explanation_lines) + "\n */\n"
                    else:
                        explanation_comment = ""

                    new_c_content = (
                        c_content[:start_line - 1] + [optimized_code + "\n"] + c_content[end_line:]
                    )

                    with output_c_file_path.open("w", encoding="utf-8") as file:
                        file.writelines(new_c_content)

                    if self.write_state and has_explanation and explanation_content:
                        write_mode = "w" if not explanation_output_path.exists() else "a" 
                        with explanation_output_path.open(write_mode, encoding="utf-8") as file:
                            file.write("\n" + "=" * 80 + "\n")
                            file.write(f"<<< {canonical_ex} >>> <{self.llm_abbrev}[v_{i}]> Optimization Explanation for {c_file_path.name}:\n")
                            file.write(explanation_comment + "\n")

                    success_count += 1
                    print(f"[INFO] <auto_run_EX2> [{benchmark_name}] Optimization completed for v{i}.")
                    print("=" * 80, flush=True)
                    self.log_activity(
                        "artifact_saved",
                        ex_version=ex_version,
                        benchmark=benchmark_name,
                        attempt=i,
                        path=str(output_c_file_path),
                    )

            if success_count == 0:
                warn_msg = RuntimeError(f"All attempts failed for {benchmark_name} in {ex_version}.")
                self.log_error(
                    ex_version=ex_version,
                    benchmark=benchmark_name,
                    model=self.llm_model,
                    path=str(self._optimized_dir(ex_version, benchmark_name)),
                    error=warn_msg,
                    level="WARN"
                )
                print(
                    f"[WARN] <auto_run_EX2> Unable to produce compilable output for {benchmark_name} "
                    f"in {ex_version} after {max_attempts} attempts."
                )
        except Exception as e:
            # Use log_error to record the error
            self.log_error(
                ex_version=ex_version,
                benchmark=benchmark_name,
                model=self.llm_model,
                path=str(self.log_file_path),
                error=e,
                level="ERROR"
            )
            
            # Write the error to the log file
            # Optional: Print the error message to the console
            print(
                f"[ERROR] <auto_run_EX2> has Exception - Benchmark: {benchmark_name}, "
                f"EX version: {ex_version}, Error: {str(e)}"
            )

    def auto_run_EX3(self, benchmark_name: str, max_attempts: int = 10, ex_version: str = "EX3"):
        """Automatically runs CUDA optimization for the EX3 benchmark."""
        ex_version = ex_version or "EX3"
        canonical_ex = canonical_ex_version(ex_version) or "EX3"
        try:
            success_count = 0
            metadata = self._get_benchmark_metadata(benchmark_name, canonical_ex)
            if not metadata:
                raise ValueError(f"[ERROR] <auto_run_EX3> Unknown benchmark: {benchmark_name}. Please define metadata.")
            
            head_code_paths = resolve_head_code_paths_ex3(self, benchmark_name)

            # Read all header files (missing files yield empty content)
            placeholder_path = self.absolute_path / "utilities" / ".empty_header_placeholder"
            if not head_code_paths:
                placeholder_path.parent.mkdir(parents=True, exist_ok=True)
                if not placeholder_path.exists():
                    placeholder_path.write_text("", encoding="utf-8")
                head_code_paths = [placeholder_path]

            head_contents = []
            for path in head_code_paths:
                head_contents.append(f"The following is the content of {path.name}:")
                try:
                    with path.open("r", encoding="utf-8") as file:
                        head_contents.append(file.read())
                except FileNotFoundError as missing:
                    self.log_error(
                        ex_version=ex_version,
                        benchmark=benchmark_name,
                        model=self.llm_model,
                        path=str(path),
                        error=missing,
                        level="WARN"
                    )
                    head_contents.append("")
            head_content = "\n".join(head_contents)

            benchmark_dir = self.absolute_path / ex_version / benchmark_name
            optimized_dir = self._optimized_dir(ex_version, benchmark_name)
            target_spec = metadata["target_file"]
            is_multi_target = self._benchmark_has_multiple_targets(benchmark_name, ex_version=canonical_ex)
            if is_multi_target:
                run_target_display = [optimized_dir / Path(path) for path in target_spec]
            else:
                single_target = target_spec[0] if isinstance(target_spec, (list, tuple)) else target_spec
                run_target_display = optimized_dir / single_target

            print(f"[INFO] <auto_run_EX3> head_code_paths = {head_code_paths}", flush=True)
            print(f"[INFO] <auto_run_EX3> Running {self.model_family} on {run_target_display}")

            sample_output_path = self._build_output_path(ex_version, benchmark_name, 1)
            explanation_dir = self.absolute_path / "generation_explainations"
            explanation_dir.mkdir(parents=True, exist_ok=True)
            explanation_output_path = explanation_dir / f"{ex_version}_optimization_explanations_{self.llm_abbrev}.txt"

            print(f"[INFO] <auto_run_EX3> Saving optimized code to {sample_output_path}")
            print(f"[INFO] <auto_run_EX3> Saving explanation to {explanation_output_path}")
            print(f"[INFO] <auto_run_EX3> Using LLM model: {self.llm_model}")
            
            prompt_system_EX3 = (
                "You are a CUDA code optimization assistant. Your task is to take input CUDA code and "
                "generate an optimized version. "
                "Your output must only be compilable CUDA source code without explanations. "
                "The computation environment is a Linux system (Enterprise Linux 8, kernel version 4.18.0-348.7.1) with "
                "NVIDIA A100-SXM4-80GB GPUs. "
                "The available CUDA Toolkit is CUDA 12.6 (Driver Version: 560.35.05) and "
                "the CUDA compiler is nvcc version 12.6.85 (release 12.6, V12.6.85)."
            )
            instruction_block_EX3 = (
                "Provide the optimized CUDA code with GPU optimizations specifically tailored for NVIDIA A100 GPUs "
                "(e.g., memory coalescing, shared memory usage, warp-level optimizations, tensor cores usage if applicable, "
                "kernel fusion, etc.), ensuring that none of the existing kernel functions "
                "or header files are removed, and no new functions or print statements are added. "
                "The code must be compatible with CUDA 12.6 and compilable with nvcc 12.6.85. "
                "Do not modify, relocate, or duplicate any time-measurement instrumentation (`clock_gettime`, `cudaDeviceSynchronize`, "
                "kernel-time accumulators, or related globals); if you restructure control flow, ensure the function still reaches "
                "the existing end-of-function timing accumulation (avoid early returns by storing results first). "
                "Avoid adding printf/logging inside kernels so output stays deterministic. "
                "Wrap the generated optimized CUDA code between <<<CODE>>> and <<<END-CODE>>>. Then provide an explanation of why this optimization was made, "
                "and wrap the explanation between <<<NOTE>>> and <<<END-NOTE>>>. "
                "Ensure that the optimized code contains only compilable source code, and the explanation contains only plain text or code snippets as needed. "
                "Do not emit anything outside those markers. "
                "Example:\n"
                "<<<CODE>>>\n"
                "__global__ void optimized_kernel(...) {\n"
                "  // optimized CUDA code for A100\n"
                "}\n"
                "<<<END-CODE>>>\n"
                "<<<NOTE>>>\n"
                "Explain the CUDA optimization rationale here (e.g., why this optimization is beneficial for A100 architecture).\n"
                "<<<END-NOTE>>>"
            )

            if is_multi_target:
                success_count = self._run_multi_target_attempts(
                    ex_version=ex_version,
                    benchmark_name=benchmark_name,
                    head_content=head_content,
                    prompt_system=prompt_system_EX3,
                    instruction_block=instruction_block_EX3,
                    benchmark_dir=benchmark_dir,
                    explanation_output_path=explanation_output_path,
                    max_attempts=max_attempts,
                )
            else:
                start_meta = metadata["start_line"]
                end_meta = metadata["end_line"]
                start_line = int(start_meta[0] if isinstance(start_meta, (list, tuple)) else start_meta)
                end_line = int(end_meta[0] if isinstance(end_meta, (list, tuple)) else end_meta)
                print(f"[INFO] <auto_run_EX3> Benchmark start line: {start_line}, end line: {end_line}")

                cuda_file_path = optimized_dir / single_target
                with cuda_file_path.open("r", encoding="utf-8") as file:
                    cuda_content = file.readlines()
                target_code = "".join(cuda_content[start_line - 1:end_line])
                self._guard_payload("Source snippet", target_code, benchmark_name, canonical_ex)

                prompt_user_EX3 = (
                    f"Header file content (macros and definitions) or related information: \n{head_content}\n\n"
                    f"Input CUDA code:\n{target_code}\n\n"
                    f"{instruction_block_EX3}"
                )
                for i in range(1, max_attempts + 1):
                    output_cuda_file_path = self._build_output_path(ex_version, benchmark_name, i)
                    self.log_activity(
                        "attempt_start",
                        ex_version=ex_version,
                        benchmark=benchmark_name,
                        attempt=i,
                        target_path=str(output_cuda_file_path),
                    )
                    if self.model_family == 'hpc-coder':
                        print(f"[INFO] <auto_run_EX3> Using HPC-Coder model: {self.llm_model} to try [{i} times]")
                    response_text = self._generate_llm_response(
                        ex_version=ex_version,
                        prompt_system=prompt_system_EX3,
                        prompt_user=prompt_user_EX3,
                        attempt=i,
                    )
                    if self.model_family == 'hpc-coder':
                        self._persist_hpc_response(benchmark_dir, benchmark_name, response_text, ex_version)
                        print(response_text, flush=True)

                    optimized_code, explanation_content, has_explanation = self.process_response_text(
                        response_text,
                        benchmark_name,
                        canonical_ex,
                    )
                    if not optimized_code.strip():
                        self.log_error(
                            ex_version=ex_version,
                            benchmark=benchmark_name,
                            model=self.llm_model,
                            path=str(output_cuda_file_path),
                            error=ValueError("LLM returned empty code block"),
                            level="WARN"
                        )
                        print(f"[WARN] <auto_run_EX3> Empty response for {benchmark_name} attempt {i}. Skipping.")
                        continue

                    # Format explanation as C-style comment if present
                    if has_explanation and explanation_content:
                        explanation_lines = explanation_content.split("\n")
                        explanation_comment = "/**\n" + "\n".join(f" * {line.strip()}" for line in explanation_lines) + "\n */\n"
                    else:
                        explanation_comment = ""

                    new_cuda_content = cuda_content[:start_line - 1] + [optimized_code + "\n"] + cuda_content[end_line:]
                    output_cuda_file_path.parent.mkdir(parents=True, exist_ok=True)
                    with output_cuda_file_path.open("w", encoding="utf-8") as outfile:
                        outfile.writelines(new_cuda_content)
                    print(f"[INFO] <auto_run_EX3> Saved optimized CUDA code to {output_cuda_file_path}")
                    self.log_activity(
                        "code_saved",
                        ex_version=ex_version,
                        benchmark=benchmark_name,
                        attempt=i,
                        output_path=str(output_cuda_file_path),
                    )

                    if self.write_state and has_explanation and explanation_content:
                        write_mode = "w" if not explanation_output_path.exists() else "a"
                        with explanation_output_path.open(write_mode, encoding="utf-8") as file:
                            file.write("\n" + "=" * 80 + "\n")
                            file.write(f"<<< {canonical_ex} >>> <{self.llm_abbrev}[v_{i}]> Optimization Explanation for {cuda_file_path.name}:\n")
                            file.write(explanation_comment + "\n")
                        print(f"[INFO] <auto_run_EX3> Saved explanation to {explanation_output_path}")
                        self.log_activity(
                            "explanation_saved",
                            ex_version=ex_version,
                            benchmark=benchmark_name,
                            attempt=i,
                            explanation_path=str(explanation_output_path),
                        )

                    success_count += 1

            print(f"[INFO] <auto_run_EX3> Completed {success_count}/{max_attempts} successful optimizations for {benchmark_name}")
            
        except Exception as e:
            # Log error both in log file and activity log
            self.log_error(
                ex_version=ex_version,
                benchmark=benchmark_name,
                model=self.llm_model,
                path=str(benchmark_dir) if 'benchmark_dir' in locals() else "-",
                error=e,
                level="ERROR"
            )
            
            # Print the error message to the console
            print(
                f"[ERROR] <auto_run_EX3> has Exception - Benchmark: {benchmark_name}, "
                f"EX version: {ex_version}, Error: {str(e)}"
            )

    def auto_run_cuda(self,benchmark_name: str):
        pass


    def extract_hpc_coder_response(self, response_text: str):
        # Define regex pattern to extract optimized code and explanation
        code_pattern = r"Optimized code:\s*```c(.*?)```"
        explanation_pattern = r"Optimization explanation:(.*)"

        # Extract optimized code block
        code_match = re.search(code_pattern, response_text, re.DOTALL)
        optimized_code = code_match.group(1).strip() if code_match else None

        # Extract explanation block
        explanation_match = re.search(explanation_pattern, response_text, re.DOTALL)
        explanation_content = explanation_match.group(1).strip() if explanation_match else None

        if not optimized_code or not explanation_content:
            raise ValueError("[ERROR] Unable to extract code or explanation. Check HPC-Coder response format.")

        # Convert explanation to C-style comment
        explanation_lines = explanation_content.split("\n")
        explanation_comment = "/**\n" + "\n".join(f" * {line.strip()}" for line in explanation_lines) + "\n */\n"

        return optimized_code, explanation_comment

    def run_all_benchmarks(self):
        """Run all benchmarks"""
        for benchmark_name in self.benchmark_dict.keys():
            self.auto_run_EX1(benchmark_name)
            self.auto_run_EX2(benchmark_name)
            self.auto_run_EX3(benchmark_name)
    
    def run_all_benchmarks_EX1(self):
        """Run all EX1 benchmarks"""
        for benchmark_name in self.benchmark_dict.keys():
            self.auto_run_EX1(benchmark_name)
    
    def run_all_benchmarks_EX2(self):
        """Run all EX2 benchmarks"""
        for benchmark_name in self.benchmark_dict.keys():
            self.auto_run_EX2(benchmark_name)
    
    def run_all_benchmarks_EX3(self):
        """Run all EX3 benchmarks"""
        for benchmark_name in self.benchmark_dict.keys():
            self.auto_run_EX3(benchmark_name)

    def run_all_benchmarks_cuda(self):
        pass

    def cleanup_generated_files(self, benchmark_name: str, ex_versions: list, delete_all_llms: bool = False):
        """Deletes all generated optimization files for the specified EX versions.
         - input: ex_versions (list) - List of EX versions to clean up (e.g., ['EX1', 'EX2', 'EX3'])
        """
        if delete_all_llms:
            llm_list = ['claude', 'o4', 'llama4'] # Ignore user-defined list, delete all LLM versions
        else:
            llm_list = [self.llm_abbrev] # Default to current instance's LLM

        metadata = self.benchmark_dict.get(benchmark_name, {})
        target_spec = metadata.get("target_file", [])
        if isinstance(target_spec, str):
            target_files = [target_spec]
        else:
            target_files = list(target_spec)

        explanation_dir = self.absolute_path / "generation_explainations"
        for ex_version in ex_versions:
            ex_path = self.absolute_path / ex_version
            benchmark_dir = ex_path / benchmark_name
            optimized_dir = self._optimized_dir(ex_version, benchmark_name)
            for abbrev in llm_list:
                for version in ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10']:
                    removed_any = False
                    for tgt in target_files or [""]:
                        tgt_path = Path(tgt)
                        base = tgt_path.stem if tgt else benchmark_name
                        suffix = tgt_path.suffix if tgt else ".c"
                        output_c_file_path = optimized_dir / f"{base}_{abbrev}_{version}{suffix}"
                        if output_c_file_path.exists():
                            os.remove(output_c_file_path)
                            removed_any = True
                            print(f"[INFO] Deleted: {output_c_file_path}")
                        fallback_txt = optimized_dir / f"{base}_{abbrev}_{version}.txt"
                        if fallback_txt.exists():
                            os.remove(fallback_txt)
                            removed_any = True
                            print(f"[INFO] Deleted: {fallback_txt}")
                    if not removed_any:
                        print(f"[WARN] No generated files found for {benchmark_name} {abbrev} {version} in {ex_version}.")
                explanation_path = explanation_dir / f"{ex_version}_optimization_explanations_{abbrev}.txt"
                if explanation_path.exists():
                    os.remove(explanation_path)
                    print(f"[INFO] Deleted: {explanation_path}")
                else:
                    print(f"[ERROR] Not found: {explanation_path}")

        print("Cleanup completed.")

    def run_make(
        self,
        benchmark_name: str,
        ex_versions: List[str],
        log_root: Optional[str] = None,
        *,
        clean: bool = False,
        clean_only: bool = False,
        enable_logs: bool = True,
    ):
        """
        Execute `make` for the given benchmark across specified EX directories.
        When logging is enabled (default), stdout/stderr is stored under
        `<log_root>/<benchmark>/<EX>/build.log`.
        """
        if clean_only:
            clean = True
        base_log: Optional[Path] = None
        if enable_logs:
            base_log = self._resolve_log_root_base(log_root)

        def _log_activity(action: str, **kwargs: Any) -> None:
            if enable_logs:
                self.log_activity(action, **kwargs)

        def _log_error(**kwargs: Any) -> None:
            if enable_logs:
                self.log_error(**kwargs)

        targets_to_run: List[Optional[str]] = []
        if clean:
            targets_to_run.append("clean")
        if not clean_only:
            targets_to_run.append(None)  # default target
        if not targets_to_run:
            targets_to_run.append(None)

        for ex_version in ex_versions:
            benchmark_dir = self.absolute_path / ex_version / benchmark_name
            log_file: Optional[Path] = None
            if enable_logs and base_log is not None:
                log_dir = base_log / benchmark_name / ex_version
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file = log_dir / "build.log"
            start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            header = [
                "=" * 80,
                f"[BUILD] benchmark={benchmark_name} ex={ex_version}",
                f"Start: {start_time}",
                f"CWD: {benchmark_dir}",
                f"Targets: {', '.join('default' if t is None else t for t in targets_to_run)}",
            ]

            messages: List[str] = header.copy()
            makefile_exists = any((benchmark_dir / name).exists() for name in ("Makefile", "makefile"))

            def _flush_log():
                messages.append(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                messages.append("=" * 80)
                output = "\n".join(messages) + "\n"
                if log_file is not None:
                    with log_file.open("a", encoding="utf-8") as fp:
                        fp.write(output)
                else:
                    print(output, end="")

            if not benchmark_dir.exists():
                msg = f"[WARN] Directory not found: {benchmark_dir}"
                messages.append(msg)
                _log_error(
                    ex_version=ex_version,
                    benchmark=benchmark_name,
                    model=self.llm_model,
                    path=str(benchmark_dir),
                    error=FileNotFoundError(msg),
                    level="WARN",
                )
                _log_activity(
                    "build_missing_directory",
                    ex_version=ex_version,
                    benchmark=benchmark_name,
                    path=str(benchmark_dir),
                    log=str(log_file),
                )
                _flush_log()
                continue

            if not makefile_exists:
                msg = f"[WARN] Makefile not found under {benchmark_dir}"
                messages.append(msg)
                _log_error(
                    ex_version=ex_version,
                    benchmark=benchmark_name,
                    model=self.llm_model,
                    path=str(benchmark_dir),
                    error=FileNotFoundError(msg),
                    level="WARN",
                )
                _log_activity(
                    "build_missing_makefile",
                    ex_version=ex_version,
                    benchmark=benchmark_name,
                    path=str(benchmark_dir),
                    log=str(log_file),
                )
                _flush_log()
                continue

            _log_activity(
                "build_start",
                ex_version=ex_version,
                benchmark=benchmark_name,
                path=str(benchmark_dir),
                log=str(log_file),
            )

            for target in targets_to_run:
                label = target or "default"
                target_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                messages.append(f"--- TARGET: {label} ---")
                messages.append(f"Start: {target_start}")
                _log_activity(
                    "build_target_start",
                    ex_version=ex_version,
                    benchmark=benchmark_name,
                    path=str(benchmark_dir),
                    log=str(log_file),
                    target=label,
                )
                cmd = ["make"]
                if target:
                    cmd.append(target)
                try:
                    result = subprocess.run(
                        cmd,
                        cwd=str(benchmark_dir),
                        capture_output=True,
                        text=True,
                    )
                    messages.append("--- STDOUT ---")
                    messages.append(result.stdout or "")
                    messages.append("--- STDERR ---")
                    messages.append(result.stderr or "")
                    messages.append(f"Return code: {result.returncode}")
                    target_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    messages.append(f"Finish: {target_end}")
                    status = "success" if result.returncode == 0 else "failed"
                    _log_activity(
                        "build_complete",
                        ex_version=ex_version,
                        benchmark=benchmark_name,
                        path=str(benchmark_dir),
                        log=str(log_file),
                        returncode=result.returncode,
                        status=status,
                        target=label,
                    )
                    if result.returncode != 0:
                        _log_error(
                            ex_version=ex_version,
                            benchmark=benchmark_name,
                            model=self.llm_model,
                            path=str(log_file),
                            error=RuntimeError(f"`make {label}` exited with code {result.returncode}"),
                            level="ERROR",
                        )
                except Exception as exc:
                    target_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    messages.append(f"[ERROR] Exception during make {label}: {exc}")
                    messages.append(f"Finish: {target_end}")
                    _log_error(
                        ex_version=ex_version,
                        benchmark=benchmark_name,
                        model=self.llm_model,
                        path=str(log_file),
                        error=exc,
                        level="ERROR",
                    )
                    _log_activity(
                        "build_exception",
                        ex_version=ex_version,
                        benchmark=benchmark_name,
                        path=str(benchmark_dir),
                        log=str(log_file),
                        target=label,
                    )
            _flush_log()

    def log_build_summary(
        self,
        benchmarks: List[str],
        ex_versions: List[str],
        log_root: Optional[str],
        *,
        clean: bool,
        clean_only: bool,
    ) -> Path:
        """
        Record a high-level summary for a build invocation under the log root.
        Returns the resolved log root path for convenience.
        """
        base_log = self._resolve_log_root_base(log_root)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        mode = "clean-only" if clean_only else ("clean+build" if clean else "build")
        bench_line = ", ".join(benchmarks)
        for ex_version in ex_versions:
            summary_file = base_log / f"{ex_version}_build_summary.log"
            with summary_file.open("a", encoding="utf-8") as fp:
                fp.write("=" * 80 + "\n")
                fp.write(f"[{timestamp}] mode={mode}\n")
                fp.write(f"Benchmarks: {bench_line}\n")
                fp.write(f"Total: {len(benchmarks)} benchmark(s)\n")
                fp.write("=" * 80 + "\n")
        return base_log

    def process_response_text(self, response_text: str, benchmark_name: str, ex_version: str, version: int = None):
        """处理模型响应文本，返回优化后的代码和解释"""
        structured = self._extract_structured_output(response_text)
        if structured:
            return structured

        # 1. 检查是否包含两个【】格式
        matches = re.findall(r"【(.*?)】", response_text, re.DOTALL)
        if len(matches) >= 2:
            optimized_code = matches[0].strip()
            explanation_content = matches[1].strip()
            return optimized_code, explanation_content, True
        
        # 2. 检查是否只有一个【】格式
        if len(matches) == 1:
            optimized_code = matches[0].strip()
            return optimized_code, None, False
        
        # 3. 检查是否包含 ```c 格式
        code_blocks = re.findall(r"```c\n(.*?)```", response_text, re.DOTALL)
        if code_blocks:
            optimized_code = code_blocks[0].strip()
            # 尝试提取解释部分
            explanation_match = re.search(r"```.*?```\s*(.*?)(?:```|$)", response_text, re.DOTALL)
            explanation_content = explanation_match.group(1).strip() if explanation_match else None
            return optimized_code, explanation_content, bool(explanation_content)
        
        # 3.5 如果整体被 [] 或 【】 包裹，则去掉包裹
        trimmed = response_text.strip()
        for opener, closer in (("[", "]"), ("【", "】")):
            if trimmed.startswith(opener) and trimmed.endswith(closer):
                inner = trimmed[len(opener):-len(closer)].strip()
                if inner:
                    return inner, None, False
        
        # 4. 检查是否只有纯代码
        code_pattern = r"(?:static\s+|inline\s+|__kernel\s+)?(?:void|int|float|double)\s+\w+\s*\([^)]*\)\s*\{[\s\S]*\}"
        if re.search(code_pattern, response_text, re.DOTALL):
            return response_text.strip(), None, False
        
        # 5. 检查是否包含 "Optimized code:" 格式
        code_match = re.search(r"Optimized code:\s*(.*?)(?:\n\n|\nOptimization explanation:|$)", response_text, re.DOTALL)
        if code_match:
            optimized_code = code_match.group(1).strip()
            explanation_match = re.search(r"Optimization explanation:\s*(.*)", response_text, re.DOTALL)
            explanation_content = explanation_match.group(1).strip() if explanation_match else None
            return optimized_code, explanation_content, bool(explanation_content)
        
        return None, None, False

    def _extract_structured_output(self, response_text: str):
        tag_sets = [
            ("<<<CODE>>>", "<<<END-CODE>>>", "<<<NOTE>>>", "<<<END-NOTE>>>"),
            ("<<<CODE>>>", "<<<END-CODE>>>", None, None),
        ]
        for code_start, code_end, note_start, note_end in tag_sets:
            code_block = self._slice_between_tags(response_text, code_start, code_end)
            if not code_block:
                continue
            explanation_block = None
            has_explanation = False
            if note_start and note_end:
                explanation_block = self._slice_between_tags(response_text, note_start, note_end)
                has_explanation = bool(explanation_block and explanation_block.strip())
            return code_block.strip(), (explanation_block.strip() if explanation_block else None), has_explanation
        return None

    @staticmethod
    def _slice_between_tags(text: str, start_tag: str, end_tag: str) -> Optional[str]:
        start_idx = text.find(start_tag)
        if start_idx == -1:
            return None
        start_idx += len(start_tag)
        end_idx = text.find(end_tag, start_idx)
        if end_idx == -1:
            return None
        return text[start_idx:end_idx]
