from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import os
import re
import json
import subprocess
import shutil
from datetime import datetime
import tempfile
import math
from typing import List, Optional, Dict, Any, Tuple, Set

from scripts.tol import compare_files_with_numpy, parse_numeric_arrays

from scripts.benchmark_catalog import (
    BENCHMARK_CODEE,
    BENCHMARK_MACHSUITE,
    BENCHMARK_MIBENCH,
    BENCHMARK_PARBOIL,
    BENCHMARK_PARSEC,
    BENCHMARK_POLYBENCH,
    BENCHMARK_RODINIA,
)
from scripts.correctness_runner import CorrectnessRunner
from scripts.benchmark_args import BENCHMARK_ARG_MAP, BENCHMARK_ARG_MAP_EX3
from scripts.ex_versions import (
    AUTO_RUN_EX_CHOICES,
    TIME_MEASUREMENT_EX_CHOICES,
    canonical_ex_version,
    optimized_subdir_name,
)
def parse_csv_list(raw: Optional[str]) -> Optional[List[str]]:
    """Split comma/space separated strings into a clean list."""
    if not raw:
        return None
    parts = []
    for chunk in raw.replace(";", ",").split(","):
        item = chunk.strip()
        if item:
            parts.append(item)
    return parts or None


def calculate_warmup_strategy(runtime_seconds: Optional[float],
                              default_warmup: int = 2,
                              default_runs: int = 10):
    """
    Decide warmup/measurement counts based on a baseline runtime.

    Returns (warmup_runs, measure_runs, reason).
    """
    if runtime_seconds is None:
        return (default_warmup, default_runs, "No baseline timing available; using defaults.")

    if runtime_seconds < 0.01:
        return (5, 25, "Ultra-short kernel; dominated by startup overhead.")
    if runtime_seconds < 0.05:
        return (4, 20, "Very short kernel; startup overhead is significant.")
    if runtime_seconds < 0.1:
        return (3, 15, "Short kernel; appreciable startup overhead.")
    if runtime_seconds < 0.5:
        return (3, 10, "Mid-short kernel; warm-up recommended.")
    if runtime_seconds < 1.0:
        return (2, 10, "Medium kernel; light warm-up advised.")
    if runtime_seconds < 3.0:
        return (2, 8, "Long-running kernel; minimal warm-up.")
    if runtime_seconds < 10.0:
        return (1, 5, "Very long kernel; small warm-up is sufficient.")
    return (0, 3, "Extremely long kernel; warm-up not required.")


def compute_stats(values: List[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    count = len(values)
    mean_val = sum(values) / count
    variance = sum((val - mean_val) ** 2 for val in values) / count
    return {
        "count": count,
        "mean": mean_val,
        "min": min(values),
        "max": max(values),
        "std": math.sqrt(variance)
    }


DEFAULT_ROOT = PROJECT_ROOT
DEFAULT_DATASET = "default"
DEFAULT_AUTO_VERSION_COUNT = 10
DEFAULT_WARMUP = 2
DEFAULT_RUNS = 10
NO_HINT_REASON = "No runtime hint found; using default strategy."
DEFAULT_RESULTS_DIR = DEFAULT_ROOT / "results"
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

class ScriptManager():
    # Store benchmark metadata (correctness pattern + line numbers)
    benchmark_polybench = BENCHMARK_POLYBENCH
    benchmark_rodinia = BENCHMARK_RODINIA
    benchmark_codee = BENCHMARK_CODEE
    benchmark_parboil = BENCHMARK_PARBOIL
    benchmark_mibench = BENCHMARK_MIBENCH
    benchmark_parsec = BENCHMARK_PARSEC
    benchmark_machsuite = BENCHMARK_MACHSUITE

    def __init__(self, absolute_path: str = "D:\\CS_PhD\\research\\NeurIPS\\target_code\\NeurIPS_25", customer_list: list = None):
        self.absolute_path = Path(absolute_path)
        self.log_file_path = (self.absolute_path / "logs" / "script_manager_errors.log").resolve()
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)

        self.polybench = list(self.benchmark_polybench.keys())
        self.rodinia = list(self.benchmark_rodinia.keys())
        self.codee = list(self.benchmark_codee.keys())
        self.mibench = list(self.benchmark_mibench.keys())
        self.parsec = list(self.benchmark_parsec.keys())
        self.machsuite = list(self.benchmark_machsuite.keys())
        self.parboil = list(self.benchmark_parboil.keys())

        if customer_list is not None:
            # 如果 customer_metadata 存在，就使用外部数据
            self.benchmark_list = list(customer_list.keys())
        else:
            # 否则使用默认的 benchmark 列表
            self.benchmark_list = (
                self.polybench
                + self.rodinia
                + self.codee
                + self.mibench
                + self.parsec
                + self.machsuite
                + self.parboil
            )
            self.benchmark_dict = (
                self.benchmark_polybench.copy()
                | self.benchmark_rodinia.copy()
                | self.benchmark_codee.copy()
                | self.benchmark_mibench.copy()
                | self.benchmark_parsec.copy()
                | self.benchmark_machsuite.copy()
                | self.benchmark_parboil.copy()
            )

        self.dataset_profiles: Dict[str, List[str]] = {}
        self.all_datasets = ["mini", "small", "medium", "large", "extra-large"]
        for name in (self.polybench + self.rodinia + self.codee + self.mibench + self.parsec + self.machsuite + self.parboil):
            self.dataset_profiles[name] = self.all_datasets.copy()

    def _resolve_log_path(self, candidate: Optional[str], default_relative: str) -> Path:
        """
        Resolve log paths relative to repository root when not absolute.
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

    def log_error(self, ex_version: str, benchmark: str, model: Optional[str], path: Optional[str], error: Exception, level: str = "ERROR"):
        """
        Append a formatted error entry directly to the ScriptManager error log.
        """
        error_type = type(error).__name__
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = {
            "timestamp": timestamp,
            "level": level,
            "experiment": ex_version,
            "benchmark": benchmark,
            "model": model or "Unknown",
            "path": path or "-",
            "reason": str(error),
            "error_type": error_type,
        }
        self._append_error_entry(entry)

    def _append_error_entry(self, entry: Dict[str, Any]) -> None:
        lines = [
            f"[TIME: {entry['timestamp']}] [{entry['level']}] [{entry['experiment']}] Benchmark: {entry['benchmark']} | Model: {entry['model']}",
            f"Reason: {entry['reason']}",
            f"Path: {entry['path']}",
            f"Error Type: {entry['error_type']}",
            "-" * 80,
        ]
        with self.log_file_path.open("a", encoding="utf-8") as fp:
            fp.write("\n".join(lines) + "\n")

    def _get_dataset_profile(self, benchmark_name: str) -> List[str]:
        profile = self.dataset_profiles.get(benchmark_name)
        if profile is None:
            profile = self.all_datasets.copy()
            self.dataset_profiles[benchmark_name] = profile
        return profile

    def _default_dataset_tag(self, benchmark_name: str) -> str:
        profile = self._get_dataset_profile(benchmark_name)
        return profile[0] if profile else DEFAULT_DATASET

    def _strategy_file_path(self, benchmark_name: str) -> Path:
        return DEFAULT_ROOT / "configs" / f"runtime_strategy_{benchmark_name}.json"

    def _legacy_strategy_candidates(self, benchmark_name: str) -> List[Path]:
        configs_dir = DEFAULT_ROOT / "configs"
        variants = []
        for ex_version in ("EX1", "EX2", "EX3", "ex1", "ex2", "ex3"):
            variants.append(configs_dir / f"{benchmark_name}_{ex_version}_runtime_strategy.json")
        return variants

    def _load_cached_runtime_strategy(self, benchmark_name: str) -> Tuple[Dict[str, Dict[str, Any]],
                                                                         Dict[str, Dict[str, Any]],
                                                                         Optional[Path]]:
        candidates = [self._strategy_file_path(benchmark_name)] + self._legacy_strategy_candidates(benchmark_name)
        for path in candidates:
            if not path.exists():
                continue
            try:
                with path.open("r", encoding="utf-8") as fp:
                    data = json.load(fp)
            except (OSError, json.JSONDecodeError):
                continue
            strategies = data.get("strategies") or {}
            runtime_map: Dict[str, Dict[str, Any]] = {}
            strategy_map: Dict[str, Dict[str, Any]] = {}
            for raw_key, info in strategies.items():
                if not isinstance(info, dict):
                    continue
                normalized = str(raw_key).lower()
                runtime_val = info.get("runtime")
                runtime_map[normalized] = {"value": runtime_val, "raw_key": raw_key}
                strategy_map[normalized] = {
                    "warmup": info.get("warmup", DEFAULT_WARMUP),
                    "runs": info.get("runs", DEFAULT_RUNS),
                    "reason": info.get("reason", NO_HINT_REASON),
                    "runtime": runtime_val,
                    "raw_key": raw_key
                }
            return runtime_map, strategy_map, path
        return {}, {}, None

    def _dump_runtime_strategy(self, benchmark_name: str, strategy_map: Dict[str, Dict[str, Any]], hints_file: Path) -> Path:
        strategy_output = {}
        for normalized, info in strategy_map.items():
            raw_key = info.get("raw_key") or normalized
            strategy_output[raw_key] = {
                "warmup": info["warmup"],
                "runs": info["runs"],
                "reason": info["reason"],
                "runtime": info["runtime"]
            }
        output_file = self._strategy_file_path(benchmark_name)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "benchmark": benchmark_name,
            "source": str(hints_file) if hints_file else None,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "strategies": strategy_output
        }
        with output_file.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
        return output_file

    def _emit_missing_binaries(self,
                               missing_entries: List[Dict[str, Any]],
                               ex_version: str,
                               benchmark_name: str,
                               dataset_profile: List[str],
                               output_path: Optional[str]) -> Path:
        output_dir = DEFAULT_RESULTS_DIR / "missing_bins" / benchmark_name
        output_dir.mkdir(parents=True, exist_ok=True)
        report_file = output_dir / f"{ex_version}_missing_bins.json"
        payload = {
            "benchmark": benchmark_name,
            "ex_version": ex_version,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "missing": missing_entries
        }
        with report_file.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
        return report_file

    def _benchmark_dir(self, ex_version: str, benchmark_name: str) -> Path:
        return self.absolute_path / ex_version / benchmark_name

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
            pass

    def _load_runtime_hints(self, hints_path: Optional[str], benchmark_name: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Optional[Path]]:
        if not hints_path:
            return {}, {}, None
        path = Path(hints_path)
        if not path.is_absolute():
            candidate = (DEFAULT_ROOT / path).resolve()
            if candidate.exists():
                path = candidate
            else:
                path = (Path.cwd() / path).resolve()
        if not path.exists():
            print(f"[WARN] Runtime hints file {path} was not found. Hints will be ignored.")
            return {}, {}, None
        try:
            with path.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
        except Exception as exc:
            print(f"[WARN] Failed to read runtime hints: {exc}")
            return {}, {}, None
        target = None
        if isinstance(data, dict):
            bench_block = data.get(benchmark_name)
            if bench_block is None:
                metadata = self.benchmark_dict.get(benchmark_name)
                if metadata:
                    bench_block = data.get(metadata.get("target_file"))
            if isinstance(bench_block, dict):
                target = bench_block
            else:
                target = data
        else:
            print(f"[WARN] Runtime hints file must be a dict but got {type(data)}. Hints will be ignored.")
            return {}, {}, None
        runtime_map: Dict[str, Dict[str, Any]] = {}
        strategy_map: Dict[str, Dict[str, Any]] = {}
        for key, value in target.items():
            try:
                runtime_val = float(value)
            except (TypeError, ValueError):
                continue
            normalized_key = str(key).lower()
            runtime_map[normalized_key] = {"value": runtime_val, "raw_key": str(key)}
            warmup, runs, reason = calculate_warmup_strategy(runtime_val, DEFAULT_WARMUP, DEFAULT_RUNS)
            strategy_map[normalized_key] = {
                "warmup": warmup,
                "runs": runs,
                "reason": reason,
                "runtime": runtime_val,
                "raw_key": str(key)
            }
        return runtime_map, strategy_map, path

    def _lookup_runtime_hint(self,
                             runtime_hints: Dict[str, Dict[str, Any]],
                             exe_entry: Dict[str, Any],
                             benchmark_name: str) -> Tuple[Optional[float], Optional[str], Optional[str]]:
        dataset = exe_entry["dataset"]
        candidates = [
            exe_entry["exe_path"].name,
            f"{exe_entry['base_name']}:{dataset}",
            f"{benchmark_name}:{dataset}",
            dataset,
            exe_entry["base_name"],
            benchmark_name,
            DEFAULT_DATASET
        ]
        visited = set()
        for key in candidates:
            if not key:
                continue
            normalized = key.lower()
            if normalized in visited:
                continue
            visited.add(normalized)
            if normalized in runtime_hints:
                entry = runtime_hints[normalized]
                return entry["value"], entry["raw_key"], normalized
        return None, None, None

    def _parse_executable_entry(self,
                                exe_path: Path,
                                kernel_name: str,
                                default_dataset: str,
                                expect_dataset: bool) -> Optional[Dict[str, Any]]:
        parts = exe_path.name.split("_")
        dataset = default_dataset
        compiler = "unknown"
        tokens = parts.copy()
        if expect_dataset and len(tokens) >= 2:
            dataset = tokens[-1]
            tokens = tokens[:-1]
        compiler_idx = None
        for idx in range(len(tokens) - 1, 0, -1):
            token = tokens[idx]
            if token.lower() in KNOWN_COMPILERS:
                compiler = token
                compiler_idx = idx
                tokens = tokens[:idx] + tokens[idx + 1:]
                break
        if compiler_idx is None:
            # Check if last token is a version number (e.g., "v1", "v2") before assuming it's a compiler
            # This handles EX3 executables without compiler suffix (e.g., "b+tree_gpt5.1_v1")
            if len(tokens) >= 2:
                last_token = tokens[-1]
                # If last token looks like a version number, don't treat it as compiler
                if not (last_token.startswith("v") and last_token[1:].isdigit()):
                    compiler = last_token
                    tokens = tokens[:-1]
                else:
                    # No compiler suffix for EX3-style naming
                    compiler = "nvcc"  # Default for CUDA code
            elif len(tokens) == 1 and tokens[0].lower() in KNOWN_COMPILERS:
                compiler = tokens[0]
                tokens = []
        base_name = "_".join(tokens) if tokens else exe_path.stem
        is_baseline = base_name == kernel_name
        model = "baseline" if is_baseline else None
        version = None
        if not is_baseline and base_name.startswith(f"{kernel_name}_"):
            suffix = base_name[len(kernel_name) + 1:]
            if "_v" in suffix:
                model_part, version_part = suffix.rsplit("_v", 1)
                model = model_part or None
                if version_part.isdigit():
                    version = int(version_part)
            else:
                model = suffix or None
        elif not is_baseline:
            model = base_name
        return {
            "exe_path": exe_path,
            "dataset": dataset,
            "compiler": compiler,
            "base_name": base_name,
            "model": model or "baseline",
            "version": version,
            "is_baseline": is_baseline,
        }

    def _collect_executables(self,
                             ex_version: str,
                             benchmark_name: str,
                             models: Optional[List[str]] = None,
                             datasets: Optional[List[str]] = None,
                             versions: Optional[List[int]] = None,
                             all_versions: bool = False) -> List[Dict[str, Any]]:
        opt_dir = self._optimized_dir(ex_version, benchmark_name)
        if not opt_dir.exists():
            raise FileNotFoundError(f"{opt_dir} does not exist. Please build the benchmark first.")
        kernel_name = benchmark_name
        dataset_profile = self._get_dataset_profile(benchmark_name)
        if not dataset_profile:
            dataset_profile = [DEFAULT_DATASET]
        default_dataset = dataset_profile[0]
        expect_dataset = benchmark_name in self.polybench
        model_filter = [m.lower() for m in models] if models else None
        dataset_filter = [d.lower() for d in datasets] if datasets else None
        version_filter = set(versions) if (versions and not all_versions) else None
        discovered_versions: Dict[str, Set[int]] = {}
        executables: List[Dict[str, Any]] = []
        for item in sorted(opt_dir.iterdir()):
            if not item.is_file():
                continue
            suffix = item.suffix
            if suffix:
                # 兼容 gpt5.1 缩写：文件名中包含 "gpt5.1" 时把最后的小数视为模型名的一部分
                allow_suffix = "gpt5.1" in item.name
                if not allow_suffix:
                    # 其他带扩展名的文件（如 .c/.txt）依旧跳过
                    continue
            entry = self._parse_executable_entry(item, kernel_name, default_dataset, expect_dataset)
            if not entry:
                continue
            entry["benchmark"] = benchmark_name
            entry["workdir"] = self._benchmark_dir(ex_version, benchmark_name)
            model_name = entry["model"].lower()
            if model_filter and model_name not in model_filter:
                continue
            if entry["version"] is not None:
                discovered_versions.setdefault(model_name, set()).add(entry["version"])
            if version_filter is not None:
                if entry.get("is_baseline"):
                    # baseline 可执行不参与版本过滤
                    pass
                else:
                    version_val = entry["version"]
                    if version_val is None or version_val not in version_filter:
                        continue
            if expect_dataset:
                if dataset_filter and entry["dataset"].lower() not in dataset_filter:
                    continue
                entry["dataset_options"] = [entry["dataset"]]
            else:
                dataset_candidates = dataset_profile
                if dataset_filter:
                    dataset_candidates = [d for d in dataset_profile if d.lower() in dataset_filter]
                if not dataset_candidates:
                    continue
                entry["dataset_options"] = dataset_candidates
                entry["dataset"] = dataset_candidates[0]
            executables.append(entry)
        expected = self._build_expected_combinations(
            benchmark_name,
            dataset_profile,
            models,
            versions,
            expect_dataset,
            dataset_filter=dataset_filter,
            discovered_versions=discovered_versions if all_versions else None,
            use_discovered_versions=all_versions,
        )
        dataset_order = {name.lower(): idx for idx, name in enumerate(dataset_profile)}
        executables.sort(key=lambda e: self._executable_sort_key(e, dataset_order))
        existing_signatures = {(entry["dataset"].lower() if expect_dataset else DEFAULT_DATASET,
                                entry["model"].lower(),
                                entry["version"]) for entry in executables}
        entries_missing = []
        for item in expected:
            signature = (item["dataset"].lower(), item["model"], item["version"])
            if signature not in existing_signatures:
                entries_missing.append({
                    "dataset": item["dataset"],
                    "model": item["model"],
                    "version": item["version"]
                })
        executables.sort(key=lambda e: self._executable_sort_key(e, dataset_order))
        return executables, entries_missing

    def _thread_options_for_measurement(self, ex_version: str, is_baseline: bool) -> List[Optional[int]]:
        canonical = canonical_ex_version(ex_version) or ex_version
        if canonical and canonical.upper() == "EX2":
            return [1] if is_baseline else [1, 2, 8, 16, 32]
        return [None]

    def _build_expected_combinations(self,
                                     benchmark_name: str,
                                     dataset_profile: List[str],
                                     models: Optional[List[str]],
                                     versions: Optional[List[int]],
                                     expect_dataset: bool,
                                     dataset_filter: Optional[List[str]] = None,
                                     discovered_versions: Optional[Dict[str, Set[int]]] = None,
                                     use_discovered_versions: bool = False) -> List[Dict[str, Any]]:
        expected = []
        dataset_iter: List[str]
        if expect_dataset:
            dataset_iter = dataset_profile if dataset_profile else [DEFAULT_DATASET]
            if dataset_filter:
                dataset_iter = [name for name in dataset_iter if name.lower() in dataset_filter]
        else:
            dataset_iter = [DEFAULT_DATASET]
        model_iter = models if models else ["baseline"]
        for dataset in dataset_iter:
            for model in model_iter:
                normalized_model = model.lower()
                version_values: List[Optional[int]]
                if normalized_model == "baseline":
                    version_values = [None]
                elif use_discovered_versions:
                    referenced_versions = sorted(discovered_versions.get(normalized_model, [])) if discovered_versions else []
                    max_version = max(referenced_versions) if referenced_versions else DEFAULT_AUTO_VERSION_COUNT
                    version_values = list(range(1, max_version + 1))
                elif versions:
                    version_values = versions
                else:
                    version_values = [None]
                for version in version_values:
                    expected.append({
                        "dataset": dataset,
                        "model": normalized_model,
                        "version": version
                    })
        return expected

    def _executable_sort_key(self, entry: Dict[str, Any], dataset_order: Dict[str, int]) -> Tuple[int, int, str, int, str]:
        dataset_idx = dataset_order.get(entry["dataset"].lower(), len(dataset_order))
        model_rank = 0 if entry["model"] == "baseline" else 1
        version_val = entry["version"] if entry["version"] is not None else 0
        return (
            dataset_idx,
            model_rank,
            entry["model"],
            version_val,
            entry["exe_path"].name
        )

    def _benchmark_args(self, benchmark_name: str, dataset: str, ex_version: Optional[str] = None) -> List[str]:
        # Choose the appropriate config based on ex_version
        if ex_version and ex_version.upper() == "EX3":
            arg_map = BENCHMARK_ARG_MAP_EX3
        else:
            arg_map = BENCHMARK_ARG_MAP
        
        config = arg_map.get(benchmark_name)
        if not config:
            return []
        dataset_key = dataset.lower()
        if dataset_key in config:
            return config[dataset_key]
        return config.get("default", [])

    def _run_binary(self,
                    exe_path: Path,
                    env: Dict[str, str],
                    args: Optional[List[str]] = None,
                    cwd: Optional[Path] = None,
                    output_path: Optional[Path] = None,
                    timeout: Optional[float] = None) -> Dict[str, Any]:
        try:
            cmd = [str(exe_path)]
            if args:
                cmd.extend(args)
            workdir = cwd if cwd is not None else exe_path.parent

            # Determine stdout destination
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                stdout_dest = open(output_path, "w", encoding="utf-8")
            else:
                stdout_dest = subprocess.DEVNULL

            try:
                completed = subprocess.run(
                    cmd,
                    stdout=stdout_dest,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=str(workdir),
                    check=False,
                    env=env,
                    timeout=timeout
                )
                return {"returncode": completed.returncode, "stderr": completed.stderr}
            except subprocess.TimeoutExpired:
                return {"returncode": -9, "stderr": f"TIMEOUT after {timeout}s"}
            finally:
                if output_path and stdout_dest != subprocess.DEVNULL:
                    stdout_dest.close()
        except Exception as exc:
            return {"returncode": None, "stderr": str(exc)}

    def _parse_timing_output(self, content: str) -> Dict[str, Optional[float]]:
        kernel_time = None
        total_time = None
        for line in content.splitlines():
            if "KERNEL_TIME" in line:
                try:
                    kernel_time = float(line.split(":")[1].strip())
                except (IndexError, ValueError):
                    kernel_time = None
            elif "TOTAL_TIME" in line:
                try:
                    total_time = float(line.split(":")[1].strip())
                except (IndexError, ValueError):
                    total_time = None
        return {"kernel": kernel_time, "total": total_time}

    def _measure_single_executable(self,
                                   exe_entry: Dict[str, Any],
                                   warmup_runs: int,
                                   measure_runs: int,
                                   writer,
                                   trail: int,
                                   stop_on_error: bool,
                                   hint_key: Optional[str],
                                   threads: Optional[int] = None,
                                   save_outputs: bool = False,
                                   ex_version: Optional[str] = None,
                                   run_timeout: Optional[float] = None) -> Dict[str, Any]:
        exe_path = exe_entry["exe_path"]
        env_template = os.environ.copy()
        if threads is not None:
            env_template["OMP_NUM_THREADS"] = str(threads)
        args = self._benchmark_args(exe_entry.get("benchmark", ""), exe_entry["dataset"], ex_version=ex_version)
        workdir = exe_entry.get("workdir") or exe_path.parent
        
        # Determine output path for EX3 if save_outputs is enabled
        # TEMPORARILY DISABLED due to crashes - use correctness-run instead
        output_path_to_save = None
        # if save_outputs and ex_version and ex_version.upper() == "EX3":
        #     benchmark_name = exe_entry.get("benchmark", "")
        #     model = exe_entry.get("model", "baseline")
        #     version = exe_entry.get("version")
        #     dataset = exe_entry.get("dataset", "default")
        #     
        #     # Build correctness output directory (EX5_correctness for all EX versions)
        #     correctness_dir = self.absolute_path / ex_version / benchmark_name / "EX5_correctness"
        #     correctness_dir.mkdir(parents=True, exist_ok=True)
        #     
        #     # Build output filename similar to correctness-run
        #     if version is not None:
        #         output_filename = f"output_{model}_v{version}_{dataset}.txt"
        #     else:
        #         output_filename = f"output_{model}_{dataset}.txt"
        #     
        #     output_path_to_save = correctness_dir / output_filename
        
        warmup_errors = []
        warmup_attempts = 0
        warmup_aborted = False
        for idx in range(warmup_runs):
            warmup_attempts += 1
            env = env_template.copy()
            env["TIMING_LOG_FILE"] = os.devnull
            result = self._run_binary(exe_path, env, args, cwd=workdir, timeout=run_timeout)
            if result["returncode"] != 0:
                is_timeout = result["returncode"] == -9
                msg = f"Warmup {idx} failed (rc={result['returncode']}): {result['stderr'].strip()}"
                warmup_errors.append(msg)
                print(f"[WARN] {exe_path.name} {msg}")
                if stop_on_error or is_timeout:
                    warmup_aborted = True
                    break

        if warmup_aborted:
            summary = {
                "record_type": "summary",
                "executable": str(exe_path),
                "dataset": exe_entry["dataset"],
                "compiler": exe_entry["compiler"],
                "model": exe_entry["model"],
                "version": exe_entry["version"],
                "is_baseline": exe_entry["is_baseline"],
                "warmup_runs": warmup_attempts,
                "measure_runs": 0,
                "successful_runs": 0,
                "kernel_stats": None,
                "total_stats": None,
                "warmup_errors": warmup_errors,
                "failed_runs": [],
                "hint_key": hint_key,
                "threads": threads,
            }
            if writer:
                writer.write(json.dumps(summary, ensure_ascii=False))
                writer.write("\n")
                writer.flush()
            return summary

        executed_warmups = warmup_attempts

        measurement_records = []
        success_kernel = []
        success_total = []
        measure_attempts = 0
        for run_idx in range(measure_runs):
            measure_attempts += 1
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                timing_path = tmp_file.name
            env = env_template.copy()
            env["TIMING_LOG_FILE"] = timing_path
            
            # Save output only on first measurement run to avoid overwriting
            output_path_for_this_run = output_path_to_save if (run_idx == 0 and output_path_to_save) else None
            result = self._run_binary(exe_path, env, args, cwd=workdir, output_path=output_path_for_this_run, timeout=run_timeout)
            
            # Log to correctness capture log if output was saved
            if run_idx == 0 and output_path_for_this_run and ex_version:
                benchmark_name = exe_entry.get("benchmark", "")
                capture_log_path = self.absolute_path / "results" / "correctness" / benchmark_name / ex_version / "correctness.jsonl"
                capture_log_path.parent.mkdir(parents=True, exist_ok=True)
                capture_record = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "stage": "capture",
                    "executable": exe_path.name,
                    "output": str(output_path_for_this_run),
                    "dataset": exe_entry["dataset"],
                    "model": exe_entry["model"],
                    "version": exe_entry.get("version"),
                    "status": "ok" if result["returncode"] == 0 else "error",
                    "returncode": result["returncode"],
                    "stderr": (result.get("stderr") or "").strip(),
                }
                with capture_log_path.open("a", encoding="utf-8") as fp:
                    fp.write(json.dumps(capture_record, ensure_ascii=False) + "\n")
            
            is_timeout = result["returncode"] == -9
            status = "timeout" if is_timeout else ("ok" if result["returncode"] == 0 else "runtime_error")
            kernel_time = None
            total_time = None
            parse_error = None
            try:
                with open(timing_path, "r", encoding="utf-8") as fp:
                    timing_content = fp.read()
            except Exception as exc:
                status = "timing_io_error"
                parse_error = str(exc)
            else:
                parsed = self._parse_timing_output(timing_content)
                kernel_time = parsed["kernel"]
                total_time = parsed["total"]
                if kernel_time is None or total_time is None:
                    status = "timing_parse_error"
                    parse_error = timing_content.strip()
            finally:
                Path(timing_path).unlink(missing_ok=True)

            record = {
                "record_type": "run",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "executable": str(exe_path),
                "dataset": exe_entry["dataset"],
                "compiler": exe_entry["compiler"],
                "model": exe_entry["model"],
                "version": exe_entry["version"],
                "is_baseline": exe_entry["is_baseline"],
                "run_index": run_idx,
                "trail": trail,
                "status": status,
                "returncode": result["returncode"],
                "kernel_time": kernel_time,
                "total_time": total_time,
                "stderr": (result["stderr"] or "").strip(),
                "error": parse_error,
                "threads": threads,
            }
            writer.write(json.dumps(record, ensure_ascii=False))
            writer.write("\n")
            writer.flush()
            measurement_records.append(record)
            if status == "ok":
                success_kernel.append(kernel_time)
                success_total.append(total_time)
            elif stop_on_error or is_timeout:
                break
        summary = {
            "record_type": "summary",
            "executable": str(exe_path),
            "dataset": exe_entry["dataset"],
            "compiler": exe_entry["compiler"],
            "model": exe_entry["model"],
            "version": exe_entry["version"],
            "is_baseline": exe_entry["is_baseline"],
            "warmup_runs": executed_warmups,
            "measure_runs": measure_attempts,
            "successful_runs": len(success_kernel),
            "kernel_stats": compute_stats(success_kernel),
            "total_stats": compute_stats(success_total),
            "warmup_errors": warmup_errors,
            "failed_runs": [rec for rec in measurement_records if rec["status"] != "ok"],
            "hint_key": hint_key,
            "threads": threads,
        }
        if writer:
            writer.write(json.dumps(summary, ensure_ascii=False))
            writer.write("\n")
            writer.flush()
        return summary

    def time_measurement(self,
                         ex_version: str,
                         benchmark_name: str,
                         model_name: Optional[str] = None,
                         trail: int = 0,
                         warmup_time: Optional[int] = None,
                         num_runs: Optional[int] = None,
                         models: Optional[List[str]] = None,
                         datasets: Optional[List[str]] = None,
                         versions: Optional[List[int]] = None,
                         all_versions: bool = False,
                         limit: Optional[int] = None,
                         dry_run: bool = True,
                         output_path: Optional[str] = None,
                         runtime_hints: Optional[str] = None,
                         stop_on_error: bool = False,
                         activity_log: Optional[str] = None,
                         output_mode: str = "append",
                         fast_run: bool = False,
                         save_outputs: bool = False):
        opt_dir = self._optimized_dir(ex_version, benchmark_name)
        dataset_profile = self._get_dataset_profile(benchmark_name)
        executables, missing_entries = self._collect_executables(
            ex_version,
            benchmark_name,
            models,
            datasets,
            versions,
            all_versions=all_versions,
        )
        tm_activity_log = self._resolve_log_path(
            activity_log,
            "logs/activity_time_measurement.log",
        )

        def log_time_measure(action: str, **details: Any):
            record = {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "action": action,
                "ex_version": ex_version,
                "benchmark": benchmark_name,
            }
            for key, value in details.items():
                if value is None:
                    continue
                if isinstance(value, Path):
                    record[key] = str(value)
                else:
                    record[key] = value
            with tm_activity_log.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")

        if missing_entries:
            missing_file = self._emit_missing_binaries(missing_entries, ex_version, benchmark_name, dataset_profile, output_path)
            print(f"[WARN] {len(missing_entries)} expected executable(s) were missing. See {missing_file} for details.")
            log_time_measure(
                "missing_binaries",
                count=len(missing_entries),
                report_path=missing_file,
            )
        if limit is not None and limit > 0:
            executables = executables[:limit]
        if not executables and not missing_entries:
            print("[INFO] No executable matched the provided filters.")
            return {
                "benchmark": benchmark_name,
                "status": "no_match",
                "plans": 0,
                "successful": 0,
                "failed": 0,
                "missing": 0,
            }

        cache_runtime_map, cache_strategy_map, cache_file = self._load_cached_runtime_strategy(benchmark_name)
        runtime_hints_map, strategy_hints_map, hints_file = self._load_runtime_hints(runtime_hints, benchmark_name)
        used_cached_strategy = False
        if strategy_hints_map and hints_file:
            strategy_target = self._strategy_file_path(benchmark_name)
            needs_update = True
            if cache_strategy_map and cache_file and cache_strategy_map == strategy_hints_map and cache_file == strategy_target:
                needs_update = False
            if needs_update:
                saved_file = self._dump_runtime_strategy(benchmark_name, strategy_hints_map, hints_file)
                print(f"[INFO] Runtime strategy saved to {saved_file}")
        elif cache_strategy_map:
            runtime_hints_map = cache_runtime_map
            strategy_hints_map = cache_strategy_map
            hints_file = cache_file
            used_cached_strategy = True
        if used_cached_strategy and cache_file:
            print(f"[INFO] Using cached runtime strategy for {benchmark_name} from {cache_file}")
        if executables:
            print(f"[INFO] Found {len(executables)} executable(s) in {opt_dir}.")
        plans = []
        for entry in executables:
            dataset_options = entry.get("dataset_options") or [entry["dataset"]]
            for dataset in dataset_options:
                dataset_entry = entry.copy()
                dataset_entry["dataset"] = dataset
                runtime_hint, hint_key, hint_norm = self._lookup_runtime_hint(runtime_hints_map, dataset_entry, benchmark_name)
                if runtime_hint is not None and hint_norm in strategy_hints_map:
                    strategy = strategy_hints_map[hint_norm]
                    warmups = warmup_time if warmup_time is not None else strategy["warmup"]
                    runs = num_runs if num_runs is not None else strategy["runs"]
                    reason = strategy["reason"]
                else:
                    warmups = warmup_time if warmup_time is not None else DEFAULT_WARMUP
                    runs = num_runs if num_runs is not None else DEFAULT_RUNS
                    reason = NO_HINT_REASON
                if fast_run:
                    warmups = 0
                    runs = 1
                thread_options = self._thread_options_for_measurement(ex_version, dataset_entry.get("is_baseline", False))
                for thread_value in thread_options:
                    plan_entry = {
                        "entry": dataset_entry.copy(),
                    "warmups": warmups,
                    "runs": runs,
                    "reason": reason,
                    "runtime_hint": runtime_hint,
                        "hint_key": hint_key,
                        "threads": thread_value,
                    }
                    plans.append(plan_entry)
        if dry_run:
            for plan in plans:
                entry = plan["entry"]
                thread_label = plan.get("threads")
                thread_part = f" threads={thread_label}" if thread_label is not None else ""
                print(f"[DRY-RUN] {entry['exe_path'].name}: dataset={entry['dataset']} "
                      f"model={entry['model']} version={entry['version']} "
                      f"warmup={plan['warmups']} runs={plan['runs']} "
                      f"hint_value={plan['runtime_hint']} hint_key={plan['hint_key']} "
                      f"reason={plan['reason']}{thread_part}")
            log_time_measure(
                "measurement_plan",
                plans=len(plans),
                status="dry_run",
            )
            return {
                "benchmark": benchmark_name,
                "status": "dry_run",
                "plans": len(plans),
                "successful": 0,
                "failed": 0,
                "missing": len(missing_entries),
            }
        if output_mode not in {"append", "overwrite"}:
            raise ValueError("output_mode must be 'append' or 'overwrite'")

        if output_path:
            output_file = Path(output_path)
        else:
            output_file = DEFAULT_RESULTS_DIR / "time_measurements" / benchmark_name / f"{ex_version}_time_measurements.jsonl"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        need_separator = False
        file_mode = "a"
        if output_mode == "overwrite":
            file_mode = "w"
        elif output_file.exists():
            try:
                need_separator = output_file.stat().st_size > 0
            except OSError:
                need_separator = False

        summaries = []
        baseline_runtimes = {}
        with output_file.open(file_mode, encoding="utf-8") as writer:
            if need_separator:
                writer.write("\n")
            for plan in plans:
                entry = plan["entry"]
                hint_label = plan["hint_key"] or "-"
                warmup_runs = plan["warmups"]
                measure_runs = plan["runs"]
                thread_count = plan.get("threads")
                thread_suffix = f", threads={thread_count}" if thread_count is not None else ""
                print(f"[INFO] Measuring {entry['exe_path'].name} "
                      f"(dataset={entry['dataset']}, model={entry['model']}, "
                      f"hint={hint_label}, warmup={warmup_runs}, runs={measure_runs}{thread_suffix})")
                log_time_measure(
                    "measure_start",
                    dataset=entry["dataset"],
                    model=entry["model"],
                    version=entry["version"],
                    executable=str(entry["exe_path"]),
                    hint=hint_label,
                    warmup_runs=warmup_runs,
                    measure_runs=measure_runs,
                    threads=thread_count,
                )
                hint_val = plan.get("runtime_hint")
                dataset_key = entry["dataset"].lower()
                if dataset_key in baseline_runtimes and baseline_runtimes[dataset_key] > 0:
                    per_run_timeout = max(30.0, baseline_runtimes[dataset_key] * 5)
                elif hint_val:
                    per_run_timeout = max(120.0, hint_val * 3)
                else:
                    per_run_timeout = 600.0
                summary = self._measure_single_executable(
                    entry,
                    plan["warmups"],
                    plan["runs"],
                    writer,
                    trail,
                    stop_on_error,
                    plan["hint_key"],
                    threads=thread_count,
                    save_outputs=save_outputs,
                    ex_version=ex_version,
                    run_timeout=per_run_timeout,
                )
                summaries.append(summary)
                if entry.get("is_baseline") and summary.get("total_stats"):
                    baseline_runtimes[dataset_key] = summary["total_stats"]["mean"]
                kernel_stats = summary["kernel_stats"]
                total_stats = summary["total_stats"]
                warmup_summary = summary.get("warmup_runs", warmup_runs)
                measure_summary = summary.get("measure_runs", measure_runs)
                summary_threads = summary.get("threads")
                thread_text = f", threads={summary_threads}" if summary_threads is not None else ""
                if kernel_stats and total_stats:
                    print(f"  -> warmup={warmup_summary}, "
                          f"successful_runs={kernel_stats['count']}/{measure_summary}{thread_text}. "
                          f"Mean kernel={kernel_stats['mean']:.6f}s, total={total_stats['mean']:.6f}s")
                else:
                    print(f"  -> warmup={warmup_summary}, measurement_runs={measure_summary}. "
                          f"No successful samples{thread_text}. Check logs for details.")
                completion_status = "success" if summary.get("successful_runs", 0) > 0 else "failed"
                log_time_measure(
                    "measure_complete",
                    dataset=entry["dataset"],
                    model=entry["model"],
                    version=entry["version"],
                    status=completion_status,
                    successful_runs=summary.get("successful_runs"),
                    warmup_runs=summary.get("warmup_runs"),
                    output_path=output_file,
                    threads=summary_threads,
                )
        successful_execs = sum(1 for item in summaries if item.get("successful_runs", 0) > 0)
        failed_execs = len(summaries) - successful_execs
        if summaries:
            status = "success" if failed_execs == 0 else "partial_failure"
        else:
            status = "missing_only" if missing_entries else "no_runs"

        print(
            f"[INFO] Measurement finished for {benchmark_name}: "
            f"{successful_execs} succeeded, {failed_execs} failed. "
            f"Results written to {output_file}"
        )
        log_time_measure(
            "measurement_finished",
            output_path=output_file,
            plans=len(plans),
            successful=successful_execs,
            failed=failed_execs,
            missing=len(missing_entries),
            status=status,
        )
        return {
            "benchmark": benchmark_name,
            "status": status,
            "plans": len(plans),
            "successful": successful_execs,
            "failed": failed_execs,
            "missing": len(missing_entries),
            "output_file": str(output_file),
        }

    def update_measure_script(self, ex_version: str, benchmark_name: str, compiler: str = "gcc"):
        """Automatically generates a measure_<compiler>_performance_times.sh in the benchmark's subdir,
        collecting all .c files in EX{X}_optimized_codes, and appending execution measuring commands.
        If the {ex_version}_optimized_codes folder does not exist, raise an error and skip."""
        metadata = self.benchmark_dict.get(benchmark_name, {})
        target_file = metadata.get("target_file")

        # Consider use g++ or gcc based on target file extension
        suffix = ""
        if isinstance(target_file, str):
            suffix = Path(target_file).suffix
        elif isinstance(target_file, (list, tuple)) and target_file:
            suffix = Path(target_file[0]).suffix
        if suffix == ".cpp" and compiler == 'gcc':
            compiler = 'g++'

        # Example: D:\CS_PhD\...\EX1\2mm\measure_gcc_performance_times.sh
        benchmark_subdir = self.absolute_path / ex_version / benchmark_name
        measure_script_name = f"measure_{compiler}_performance_times.sh"
        measure_script_path = benchmark_subdir / measure_script_name

        # We'll build up the script lines in memory
        lines = ["#!/bin/bash\n\n"]
        lines.append(f"echo 'Compiling with {compiler} for {ex_version} {benchmark_name} ...'\n")
        #lines.append(f"cd {benchmark_subdir}\n")
        lines.append('cd "$(dirname "$0")"\n')

        # We'll store executable names here
        executable_names = []

        # The folder that contains the optimized .c files
        optimized_dir = benchmark_subdir / f"{ex_version}_optimized_codes"
        target_dir = benchmark_subdir / "EX5_correctness"
        # If the directory does not exist, raise an error and skip
        if not optimized_dir.exists():
            raise FileNotFoundError(f"[ERROR: ScriptManager] {optimized_dir} not found. Skipping script generation.")

        # 1) Generate all compile commands for each .c file in {ex_version}_optimized_codes
        c_files = sorted(optimized_dir.glob("*.c")) + sorted(optimized_dir.glob("*.cpp"))
        for c_file in c_files:
            c_filename = c_file.name  # e.g. 2mm_llama.c
            base_name = Path(c_filename).stem  # e.g. 2mm_llama

            # We'll define the command differently depending on the benchmark name
            compile_command = None

            #if "2mm" in benchmark_name or "3mm" in benchmark_name or "deriche" in benchmark_name:
            if benchmark_name in self.polybench:
                # For example, polybench style
                if ex_version == 'EX1':
                    compile_command = (
                        # f"{compiler} -O3 -I ../utilities/ -I . {ex_version}_optimized_codes/{c_filename} "
                        # f"../utilities/polybench.c -DPOLYBENCH_TIME -lm -o {ex_version}_optimized_codes/{base_name}\n"
                        f"{compiler} -DMINI_DATASET -O3 -I ../../utilities/ -I . {ex_version}_optimized_codes/{c_filename} "
                        f"../../utilities/polybench.c -lm -o {ex_version}_optimized_codes/{base_name}_{compiler}_mini\n"
                        f"{compiler} -DSMALL_DATASET -O3 -I ../../utilities/ -I . {ex_version}_optimized_codes/{c_filename} "
                        f"../../utilities/polybench.c -lm -o {ex_version}_optimized_codes/{base_name}_{compiler}_small\n"
                        f"{compiler} -DMEDIUM_DATASET -O3 -I ../../utilities/ -I . {ex_version}_optimized_codes/{c_filename} "
                        f"../../utilities/polybench.c -lm -o {ex_version}_optimized_codes/{base_name}_{compiler}_medium\n"
                        f"{compiler} -DLARGE_DATASET -O3 -I ../../utilities/ -I . {ex_version}_optimized_codes/{c_filename} "
                        f"../../utilities/polybench.c -lm -o {ex_version}_optimized_codes/{base_name}_{compiler}_large\n"
                        f"{compiler} -DEXTRALARGE_DATASET -O3 -I ../../utilities/ -I . {ex_version}_optimized_codes/{c_filename} "
                        f"../../utilities/polybench.c -lm -o {ex_version}_optimized_codes/{base_name}_{compiler}_extralarge\n"
                )
                elif ex_version == 'EX2':
                    compile_command = (
                            f"{compiler} -DMINI_DATASET -O3 -fopenmp -I ../../utilities/ -I . {ex_version}_optimized_codes/{c_filename} "
                            f"../../utilities/polybench.c -lm -o {ex_version}_optimized_codes/{base_name}_{compiler}_mini\n"
                            f"{compiler} -DSMALL_DATASET -O3 -fopenmp -I ../../utilities/ -I . {ex_version}_optimized_codes/{c_filename} "
                            f"../../utilities/polybench.c -lm -o {ex_version}_optimized_codes/{base_name}_{compiler}_small\n"
                            f"{compiler} -DMEDIUM_DATASET -O3 -fopenmp -I ../../utilities/ -I . {ex_version}_optimized_codes/{c_filename} "
                            f"../../utilities/polybench.c -lm -o {ex_version}_optimized_codes/{base_name}_{compiler}_medium\n"
                            f"{compiler} -DLARGE_DATASET -O3 -fopenmp -I ../../utilities/ -I . {ex_version}_optimized_codes/{c_filename} "
                            f"../../utilities/polybench.c -lm -o {ex_version}_optimized_codes/{base_name}_{compiler}_large\n"
                            f"{compiler} -DEXTRALARGE_DATASET -O3 -fopenmp -I ../../utilities/ -I . {ex_version}_optimized_codes/{c_filename} "
                            f"../../utilities/polybench.c -lm -o {ex_version}_optimized_codes/{base_name}_{compiler}_extralarge\n"
                    )
            elif benchmark_name in self.rodinia:
                # Another style of compile command (placeholder)
                if ex_version == 'EX1':
                    if benchmark_name == 'particlefilter':
                        compile_command = (
                            f"{compiler} -w -O3 -I {ex_version}_optimized_codes/{c_filename} "
                            f"-lm -o {ex_version}_optimized_codes/{base_name}_{compiler}\n"
                        )
                    elif benchmark_name == 'hotspot.cpp':
                        compile_command = (
                            f"{compiler} -O3 {ex_version}_optimized_codes/{c_filename} "
                            f"-lm -o {ex_version}_optimized_codes/{base_name}_{compiler}\n"
                        )
                    elif benchmark_name == 'hotspot3D':
                        compile_command = (
                            f"{compiler} -O3 {ex_version}_optimized_codes/{c_filename} "
                            f"-lm -o {ex_version}_optimized_codes/{base_name}_{compiler}\n"
                        )
                    elif benchmark_name == 'srad.cpp':
                        compile_command = (
                            f"{compiler} -w -O3 {ex_version}_optimized_codes/{c_filename} "
                            f"-lm -o {ex_version}_optimized_codes/{base_name}_{compiler}\n"
                        )
                elif ex_version == 'EX2':
                    if benchmark_name == 'particlefilter':
                        compile_command = (
                            f"{compiler} -w -O3 -fopenmp {ex_version}_optimized_codes/{c_filename} "
                            f"-lm -o {ex_version}_optimized_codes/{base_name}_{compiler}\n"
                        )
                    elif benchmark_name == 'hotspot.cpp':
                        compile_command = (
                            f"{compiler} -O3 -fopenmp {ex_version}_optimized_codes/{c_filename} "
                            f"-lm -o {ex_version}_optimized_codes/{base_name}_{compiler}\n"
                        )
                    elif benchmark_name == 'hotspot3D':
                        compile_command = (
                            f"{compiler} -O3 - fopenmp {ex_version}_optimized_codes/{c_filename} "
                            f"-lm -o {ex_version}_optimized_codes/{base_name}_{compiler}\n"
                        )
                    elif benchmark_name == 'srad.cpp':
                        compile_command = (
                            f"{compiler} -w -O3 - fopenmp {ex_version}_optimized_codes/{c_filename} "
                            f"-lm -o {ex_version}_optimized_codes/{base_name}_{compiler}\n"
                        )
            elif benchmark_name == 'matmul':
                # Another style of compile command (placeholder)
                if ex_version == 'EX1':
                    compile_command = (
                        f"{compiler} {ex_version}_optimized_codes/{c_filename} lib/matrix.c lib/clock.c -I lib -I include"
                        f"{ex_version}_optimized_codes/{base_name}_{compiler} -O3\n"
                    )
                elif ex_version == 'EX2':
                    compile_command = (
                        f"{compiler} -fopenmp {ex_version}_optimized_codes/{c_filename} lib/matrix.c lib/clock.c -I lib -I include"
                        f"{ex_version}_optimized_codes/{base_name}_{compiler} -O3\n"
                    )
            elif benchmark_name == 'pi':
                # Another style of compile command (placeholder)
                if ex_version == 'EX1':
                    compile_command = (
                        f"{compiler} {ex_version}_optimized_codes/{c_filename} -lm -o "
                        f"{ex_version}_optimized_codes/{base_name}_{compiler}\n -O3"
                    )
                elif ex_version == 'EX2':
                    compile_command = (
                        f"{compiler} -fopenmp {ex_version}_optimized_codes/{c_filename} -lm -o "
                        f"{ex_version}_optimized_codes/{base_name}_{compiler} -O3\n"
                    )
            elif benchmark_name == 'atmux':
                # Another style of compile command (placeholder)
                if ex_version == 'EX1':
                    compile_command = (
                        f"{compiler} {ex_version}_optimized_codes/{c_filename} lib/CRSMatrix.c lib/Matrix2D.c lib/Vector.c -I lib -lm -o "
                        f"{ex_version}_optimized_codes/{base_name}_{compiler} -O3\n"
                    )
                elif ex_version == 'EX2':
                    compile_command = (
                        f"{compiler} -fopenmp {ex_version}_optimized_codes/{c_filename} lib/CRSMatrix.c lib/Matrix2D.c lib/Vector.c -I lib -lm -o "
                        f"{ex_version}_optimized_codes/{base_name}_{compiler} -O3\n"
                    )
            elif benchmark_name == 'coulomb':
                # Another style of compile command (placeholder)
                if ex_version == 'EX1':
                    compile_command = (
                        f"{compiler} -w {ex_version}_optimized_codes/{c_filename} lib/Matrix2D.c lib/Vector.c -I lib -I include -lm -o "
                        f"{ex_version}_optimized_codes/{base_name}_{compiler} -O3\n"
                    )
                elif ex_version == 'EX2':
                    compile_command = (
                        f"{compiler} -fopenmp -w {ex_version}_optimized_codes/{c_filename} lib/Matrix2D.c lib/Vector.c -I lib -I include -lm -o "
                        f"{ex_version}_optimized_codes/{base_name}_{compiler} -O3\n"
                    )
            elif benchmark_name == 'haccmk':
                # Another style of compile command (placeholder)
                if ex_version == 'EX1':
                    compile_command = (
                        f"{compiler} {ex_version}_optimized_codes/{c_filename} lib/Step10_orig.c lib/mysecond.c -I lib -lm -o "
                        f"{ex_version}_optimized_codes/{base_name}_{compiler} -O3\n"
                    )
                elif ex_version == 'EX2':
                    compile_command = (
                        f"{compiler} -fopenmp {ex_version}_optimized_codes/{c_filename} lib/Step10_orig.c lib/mysecond.c -I lib -lm -o "
                        f"{ex_version}_optimized_codes/{base_name}_{compiler} -O3\n"
                    )
            elif benchmark_name == 'TBD':
                # Another style of compile command (placeholder)
                compile_command = (
                    f"{compiler} -o3 -I ../utilities/ -I . {ex_version}_optimized_codes/{c_filename} "
                    f"/path/to/matmal_lib.c -lm -o {ex_version}_optimized_codes/{base_name}\n"
                )
            elif benchmark_name == 'TBD':
                # Another style of compile command (placeholder)
                compile_command = (
                    f"{compiler} -o3 -I ../utilities/ -I . {ex_version}_optimized_codes/{c_filename} "
                    f"/path/to/matmal_lib.c -lm -o {ex_version}_optimized_codes/{base_name}\n"
                )
            elif benchmark_name == 'TBD':
                # Another style of compile command (placeholder)
                compile_command = (
                    f"{compiler} -o3 -I ../utilities/ -I . {ex_version}_optimized_codes/{c_filename} "
                    f"/path/to/matmal_lib.c -lm -o {ex_version}_optimized_codes/{base_name}\n"
                )
            elif benchmark_name == 'TBD':
                # Another style of compile command (placeholder)
                compile_command = (
                    f"{compiler} -o3 -I ../utilities/ -I . {ex_version}_optimized_codes/{c_filename} "
                    f"/path/to/matmal_lib.c -lm -o {ex_version}_optimized_codes/{base_name}\n"
                )
            elif benchmark_name == 'TBD':
                # Another style of compile command (placeholder)
                compile_command = (
                    f"{compiler} -o3 -I ../utilities/ -I . {ex_version}_optimized_codes/{c_filename} "
                    f"/path/to/matmal_lib.c -lm -o {ex_version}_optimized_codes/{base_name}\n"
                )
            elif benchmark_name == 'TBD':
                # Another style of compile command (placeholder)
                compile_command = (
                    f"{compiler} -o3 -I ../utilities/ -I . {ex_version}_optimized_codes/{c_filename} "
                    f"/path/to/matmal_lib.c -lm -o {ex_version}_optimized_codes/{base_name}\n"
                )
            elif benchmark_name == 'TBD':
                # Another style of compile command (placeholder)
                compile_command = (
                    f"{compiler} -o3 -I ../utilities/ -I . {ex_version}_optimized_codes/{c_filename} "
                    f"/path/to/matmal_lib.c -lm -o {ex_version}_optimized_codes/{base_name}\n"
                )
            elif benchmark_name == 'TBD':
                # Another style of compile command (placeholder)
                compile_command = (
                    f"{compiler} -o3 -I ../utilities/ -I . {ex_version}_optimized_codes/{c_filename} "
                    f"/path/to/matmal_lib.c -lm -o {ex_version}_optimized_codes/{base_name}\n"
                )
            else:
                raise ValueError(f"[ERROR] Unknown benchmark: {benchmark_name}. Please define metadata.")
            lines.append(compile_command)
            for data_type in ['mini', 'small', 'medium', 'large', 'extralarge']:
                executable_names.append(f"{base_name}_{compiler}_{data_type}")

        lines.append("\n")
        lines.append("# Define executable names\n")
        quoted_names = [f"\"{n}\"" for n in executable_names]
        array_line = "executables=(" + " ".join(quoted_names) + ")\n"
        lines.append(array_line)

        lines.append("\n# Number of runs\n")
        lines.append("num_runs=5\n")

        lines.append("\n# Create a log file to store execution times\n")
        log_file_name = f"execution_times_{compiler}.log"
        lines.append(f"log_file=\"{log_file_name}\"\n")
        lines.append("> $log_file  # Clear log file\n\n")

        # 2) Add the loop to measure execution times
        if ex_version == "EX1":
            lines.append("# Run each executable num_runs times\n")
            lines.append("for exe in \"${executables[@]}\"; do\n")
            lines.append("  echo \"Running $exe...\" | tee -a $log_file\n")
            lines.append("  total_time=0\n")
            lines.append("  for ((i=1; i<=num_runs; i++)); do\n")
            lines.append("    start_time=$(date +%s.%N)\n")
            if benchmark_name in self.polybench:
                lines.append(f"    ./{ex_version}_optimized_codes/$exe > /dev/null 2>&1\n")
            elif benchmark_name in self.rodinia:
                if benchmark_name == 'particlefilter':
                    lines.append(f"    ./{ex_version}_optimized_codes/$exe -x 128 -y 128 -z 10 -np 10000 > /dev/null 2>&1\n")
                elif benchmark_name == 'hotspot.cpp':
                    lines.append(f"    ./{ex_version}_optimized_codes/$exe 2048 2048 512 1 inputs/temp_2048.txt inputs/power_2048.txt > /dev/null 2>&1\n")
                elif benchmark_name == 'hotspot3D':
                    lines.append(f"    ./{ex_version}_optimized_codes/$exe 1024 8 100 inputs/power_1024x1024x8.txt inputs/temp_1024x1024x8.txt > /dev/null 2>&1\n")
                elif benchmark_name == 'srad.cpp':
                    lines.append(f"    ./{ex_version}_optimized_codes/$exe 2048 2048 0 127 0 127 1 0.5 50 > /dev/null 2>&1\n")
            elif benchmark_name == 'matmul':
                lines.append(f"    ./{ex_version}_optimized_codes/$exe 2000 > /dev/null 2>&1\n")
            elif benchmark_name == 'pi':
                lines.append(f"    ./{ex_version}_optimized_codes/$exe 800000000 > /dev/null 2>&1\n")
            elif benchmark_name == 'atmux':
                lines.append(f"    ./{ex_version}_optimized_codes/$exe 25000 > /dev/null 2>&1\n")
            elif benchmark_name == 'coulomb':
                lines.append(f"    ./{ex_version}_optimized_codes/$exe 350 > /dev/null 2>&1\n")
            elif benchmark_name == 'haccmk':
                lines.append(f"    ./{ex_version}_optimized_codes/$exe > /dev/null 2>&1\n")
            elif benchmark_name == 'TBD':
                pass
            lines.append("    end_time=$(date +%s.%N)\n")
            lines.append("    elapsed_time=$(echo \"$end_time - $start_time\" | bc)\n")
            lines.append("    total_time=$(echo \"$total_time + $elapsed_time\" | bc)\n")
            lines.append("    echo \"Run $i: $elapsed_time seconds\" | tee -a $log_file\n")
            lines.append("  done\n")
            lines.append("  avg_time=$(echo \"scale=5; $total_time / $num_runs\" | bc)\n")
            lines.append("  echo \"$exe average time: $avg_time seconds\" | tee -a $log_file\n")
            lines.append("  echo \"------------------------\" | tee -a $log_file\n")
            lines.append("done\n\n")

        elif ex_version == "EX2":
            # binary_file_list = [f.name for f in optimized_dir.iterdir() if f.is_file() and f.suffix == '']
            # pattern = re.compile(r'^[^_]+_gcc_[^\.]+$')
            # result = [s for s in binary_file_list
            #           if pattern.match(s) and all(x not in s for x in ['_llama_', '_claude_', '_o4_'])]
            # bash_list = ' '.join(f'"{x}"' for x in result)
            lines.append("thread_counts=(1 2 4 16 32)\n\n")
            lines.append("# Run each executable with multiple thread counts\n")
            lines.append("for exe in \"${executables[@]}\"; do\n")
            lines.append(f"  if [ \"$exe\" == \"2mm\" ]; then\n")
            lines.append("    echo \"Running $exe with 1 threads...\" | tee -a $log_file\n")
            lines.append("    total_time=0\n")
            lines.append("    export OMP_NUM_THREADS=1\n")
            lines.append("    for ((i=1; i<=num_runs; i++)); do\n")
            lines.append(f"      start_time=$(date +%s.%N)\n")
            lines.append(f"      ./{ex_version}_optimized_codes/$exe > /dev/null 2>&1\n")
            lines.append(f"      end_time=$(date +%s.%N)\n")
            lines.append("      elapsed_time=$(echo \"$end_time - $start_time\" | bc)\n")
            lines.append("      total_time=$(echo \"$total_time + $elapsed_time\" | bc)\n")
            lines.append("    done\n")
            lines.append("    avg_time=$(echo \"scale=5; $total_time / $num_runs\" | bc)\n")
            lines.append("    echo \"$exe average time with 1 threads: $avg_time seconds\" | tee -a $log_file\n")
            lines.append("    echo \"------------------------\" | tee -a $log_file\n")
            lines.append("  else\n")
            lines.append("    for threads in \"${thread_counts[@]}\"; do\n")
            lines.append("      echo \"Running $exe with $threads threads...\" | tee -a $log_file\n")
            lines.append("      total_time=0\n")
            lines.append("      export OMP_NUM_THREADS=$threads\n")
            lines.append("      for ((i=1; i<=num_runs; i++)); do\n")
            lines.append(f"        start_time=$(date +%s.%N)\n")
            if benchmark_name in self.polybench:
                lines.append(f"    ./{ex_version}_optimized_codes/$exe > /dev/null 2>&1\n")
            elif benchmark_name in self.rodinia:
                if benchmark_name == 'particlefilter':
                    lines.append(f"    ./{ex_version}_optimized_codes/$exe -x 128 -y 128 -z 10 -np 10000 > /dev/null 2>&1\n")
                elif benchmark_name == 'hotspot.cpp':
                    lines.append(f"    ./{ex_version}_optimized_codes/$exe 2048 2048 512 1 inputs/temp_2048.txt inputs/power_2048.txt > /dev/null 2>&1\n")
                elif benchmark_name == 'hotspot3D':
                    lines.append(f"    ./{ex_version}_optimized_codes/$exe 1024 8 100 inputs/power_1024x1024x8.txt inputs/temp_1024x1024x8.txt > /dev/null 2>&1\n")
            elif benchmark_name == 'matmul':
                lines.append(f"    ./{ex_version}_optimized_codes/$exe 2000 > /dev/null 2>&1\n")
            elif benchmark_name == 'pi':
                lines.append(f"    ./{ex_version}_optimized_codes/$exe 800000000 > /dev/null 2>&1\n")
            elif benchmark_name == 'atmux':
                lines.append(f"    ./{ex_version}_optimized_codes/$exe 25000 > /dev/null 2>&1\n")
            elif benchmark_name == 'coulomb':
                lines.append(f"    ./{ex_version}_optimized_codes/$exe 350 > /dev/null 2>&1\n")
            elif benchmark_name == 'haccmk':
                lines.append(f"    ./{ex_version}_optimized_codes/$exe > /dev/null 2>&1\n")
            elif benchmark_name == 'TBD':
                pass
            lines.append(f"        end_time=$(date +%s.%N)\n")
            lines.append("        elapsed_time=$(echo \"$end_time - $start_time\" | bc)\n")
            lines.append("        total_time=$(echo \"$total_time + $elapsed_time\" | bc)\n")
            lines.append("      done\n")
            lines.append("      avg_time=$(echo \"scale=5; $total_time / $num_runs\" | bc)\n")
            lines.append("      echo \"$exe average time with $threads threads: $avg_time seconds\" | tee -a $log_file\n")
            lines.append("      echo \"------------------------\" | tee -a $log_file\n")
            lines.append("    done\n")
            lines.append("  fi\n")
            lines.append("done\n\n")

        lines.append("echo \"Execution completed. Check $log_file for details.\" | tee -a $log_file\n")

        # Write out the script
        with measure_script_path.open("w", encoding="utf-8") as f:
            f.writelines(lines)

        os.chmod(measure_script_path, 0o755)
        print(f"[INFO: ScriptManager] Updated measure script: {measure_script_path}")

    def update_measure_script_c(self,ex_version: str, benchmark_name: str):
        compile_list = ["gcc", "clang"]
        for compiler in compile_list:
            self.update_measure_script(ex_version, benchmark_name, compiler)
    
    def update_measure_script_ce(self, benchmark):
        ex_version_list = ["EX1", "EX2", "EX3"]
        for ex_version in ex_version_list:
            self.update_measure_script_c(ex_version, benchmark)
    
    def update_measure_script_all(self):
        for benchmark in self.benchmark_list:
            self.update_measure_script_ce(benchmark)

    def compare_outputs(self, ex_version: str, benchmark_name: str, log_file: str = "comparison_results.log", write_mode: str = "w"):
        """
        Compare the <all> the output files in EX5_correctness, given the ex_version and benchmark_name
        If a baseline file (with exactly one underscore) is found, compare all others against it
        Save PASS and FAIL results to log_file
        """
        try:
            benchmark_subdir = self.absolute_path / ex_version / benchmark_name
            ex5_dir = benchmark_subdir / "EX5_correctness"

            if not ex5_dir.exists():
                raise FileNotFoundError(f"[ERROR] <compare_outputs> {ex5_dir} not found.")

            # 1. Get all .txt files in the EX5_correctness directory
            txt_files = list(ex5_dir.glob("*.txt"))
            if not txt_files:
                raise FileNotFoundError(f"[ERROR] <compare_outputs> No .txt files found in {ex5_dir}")

            # 2. Identify the baseline file (filename containing exactly one underscore)
            baseline_file = None
            for f in txt_files:
                if f.name.count("_") == 2:
                    baseline_file = f
                    break

            if not baseline_file:
                raise ValueError("[ERROR] <compare_outputs> No baseline .txt file found (expecting a file with exactly one underscore)")

            print(f"[INFO] <compare_outputs> Baseline output file: {baseline_file.name}")

            # 3. Get the comparison files
            compare_files = [f for f in txt_files if f != baseline_file]

            # 4. Get the benchmark-specific threshold
            if benchmark_name not in self.benchmark_dict:
                raise ValueError(f"[ERROR] <compare_outputs> Threshold not found for benchmark: {benchmark_name}")

            threshold = self.benchmark_dict[benchmark_name].get("threshold", 0.002)

            # 5. Compare baseline against all other output files
            print(f"[INFO] <compare_outputs> Starting numeric comparison in {ex_version}/{benchmark_name} with threshold={threshold:.3f}...\n")

            # Open the log file to write results
            with open(log_file, write_mode, encoding="utf-8") as log:
                log.write("="*80 + "\n")
                log.write(f"[COMPARE] {ex_version}/{benchmark_name}\n")
                log.write(f"Baseline: {baseline_file.name}\n")
                log.write(f"Threshold: {threshold}\n\n")

                for compare_file in compare_files:
                    ok, msg = compare_files_with_numpy(
                        str(baseline_file), str(compare_file), threshold
                    )
                    if ok:
                        result = f"[PASS] {compare_file.name} => {msg}"
                    else:
                        result = f"[FAIL] {compare_file.name} => {msg}"

                    print(result)  # Print to console
                    log.write(result + "\n")  # Append to log file

                log.write("="*80 + "\n\n")

        except Exception as e:
            # Use log_error to record the error
            self.log_error(
                ex_version=ex_version,
                benchmark=benchmark_name,
                model=None,
                path=str(self.log_file_path),
                error=e,
                level="ERROR"
            )
            
            # Optional: Print the error message to the console
            print(f"[ERROR] <compare_outputs> has Exception - Benchmark: {benchmark_name}, EX version: {ex_version}, Error: {str(e)}")

    def update_correctness_script(self, ex_version: str, benchmark_name: str):
        """
        Creates measure_correctness.sh inside the benchmark's subdir, referencing EX5_correctness as output folder.
        Uses the single file without '_' as baseline, and compares all others against it.
        """
        # The path of the target script
        try:
            benchmark_subdir = self.absolute_path / ex_version / benchmark_name
            script_path = benchmark_subdir / "measure_correctness.sh"

            run_dir = benchmark_subdir / f"{ex_version}_optimized_codes"
            ex5_dir = benchmark_subdir / "EX5_correctness"
            if not ex5_dir.exists() or not run_dir.exists():
                raise FileNotFoundError(f"[ERROR] <update_correctness_script> {ex5_dir} or {run_dir} not found, cannot create correctness script.")

            # Traverse the EX5_correctness directory to locate all executable files
            # Assume an executable is "no extension + potentially executable permission" or .out
            exec_files = []
            for f in run_dir.iterdir():
                # Here we only demonstrate that if there's no extension, we consider it an executable
                # You could also check if f.suffix == '.out' or verify os.access(f, os.X_OK)
                if f.is_file() and f.suffix == "" and ("gcc" in f.name or "g++" in f.name):
                    exec_files.append(f.name)

            # Find the single file without '_' as the baseline, and everything else for comparison
            baseline_name = None
            comparison_names = []
            for name in exec_files:
                #print(f"[INFO ScriptManager] Found executable: {name}", flush=True)
                #if "_" not in name:
                if name.count("_") == 1:
                    # We assume there's exactly one such baseline
                    if baseline_name is not None:
                        raise ValueError("[ERROR] <update_correctness_script> Found multiple files with only one '_'. Expected exactly one baseline!")
                    baseline_name = name
                else:
                    comparison_names.append(name)

            if baseline_name is None:
                raise ValueError("[ERROR] <update_correctness_script> No file found with only one '_', cannot determine baseline.")

            # Construct the script
            lines = ["#!/bin/bash\n\n"]
            lines.append("# Automatically discovered executables from EX5_correctness\n")
            lines.append('cd "$(dirname "$0")"\n')

            # Put baseline first, and sort the comparison_names
            quoted_names = ' '.join(f'"{name}"' for name in [baseline_name] + sorted(comparison_names))
            lines.append(f"executables=({quoted_names})\n\n")

            lines.append("# Define output directory\n")
            lines.append('output_dir="EX5_correctness"\n\n')

            # Loop through executables
            lines.append("# Loop through executables and run with the same parameters\n")
            lines.append('for exe in "${executables[@]}"; do\n')
            lines.append(f'  exe_path="{ex_version}_optimized_codes/$exe"\n\n')

            # Check if executable exists and is executable
            lines.append('  # Check if the executable file exists and has execute permissions \n')
            lines.append('  if [[ -x "$exe_path" ]]; then\n')
            lines.append('    echo "Running $exe..."\n')
            # Run executable with fixed parameters
            if benchmark_name in self.polybench:
                lines.append('    "$exe_path" > "$output_dir/${exe}_output.txt" 2>&1\n')
            elif benchmark_name in self.rodinia:
                if benchmark_name == 'particlefilter':
                    lines.append('    "$exe_path" -x 128 -y 128 -z 10 -np 10000 > "$output_dir/${exe}_output.txt" 2>&1\n')
                elif benchmark_name == 'hotspot.cpp':
                    lines.append('    "$exe_path" 2048 2048 512 1 inputs/temp_2048.txt inputs/power_2048.txt > "$output_dir/${exe}_output.txt" 2>&1\n')
                elif benchmark_name == 'hotspot3D':
                    lines.append('    "$exe_path" 1024 8 100 inputs/power_1024x1024x8.txt inputs/temp_1024x1024x8.txt > "$output_dir/${exe}_output.txt" 2>&1\n')
                elif benchmark_name == 'srad.cpp':
                    lines.append('    "$exe_path" 2048 2048 0 127 0 127 1 0.5 50 > "$output_dir/${exe}_output.txt" 2>&1\n')
            elif benchmark_name == 'matmul':
                lines.append('    "$exe_path" 2000 > "$output_dir/${exe}_output.txt" 2>&1\n')
            elif benchmark_name == 'pi':
                lines.append('    "$exe_path" 800000000 > "$output_dir/${exe}_output.txt" 2>&1\n')
            elif benchmark_name == 'atmux':
                lines.append('    "$exe_path" 25000 > "$output_dir/${exe}_output.txt" 2>&1\n')
            elif benchmark_name == 'coulomb':
                lines.append('    "$exe_path" 350 > "$output_dir/${exe}_output.txt" 2>&1\n')
            elif benchmark_name == 'haccmk':
                lines.append('    "$exe_path" > "$output_dir/${exe}_output.txt" 2>&1\n')
            elif benchmark_name == 'TBD':
                pass

            # lines.append('    "$exe_path" 2048 2048 512 1 inputs/temp_2048.txt inputs/power_2048.txt > "$output_dir/${exe}_output.txt"\n')
            lines.append('  else\n')
            lines.append('    echo "Executable $exe_path not found or not executable."\n')
            lines.append('  fi\n')
            lines.append('done\n')

            with script_path.open("w", encoding="utf-8") as f:
                f.writelines(lines)

            # Make the script executable
            os.chmod(script_path, 0o755)
            print(f"[INFO] <update_correctness_script> Updated correctness script: {script_path}")
        except Exception as e:
            # Use log_error to record the error
            self.log_error(
                ex_version=ex_version,
                benchmark=benchmark_name,
                model=None,
                path=str(self.log_file_path),
                error=e,
                level="ERROR"
            )
            
            # Optional: Print the error message to the console
            print(f"[ERROR] <update_correctness_script> has Exception - Benchmark: {benchmark_name}, EX version: {ex_version}, Error: {str(e)}")

    def update_correctness_script_e(self, benchmark):
        ex_version_list = ["EX1", "EX2", "EX3"]
        for ex_version in ex_version_list:
            self.update_correctness_script(ex_version, benchmark)

    def update_correctness_script_all(self):
        for benchmark in self.benchmark_list:
            self.update_correctness_script_e(benchmark)

    def remove_measure_script(self, ex_version: str, benchmark_name: str, compiler: str):
        benchmark_subdir = self.absolute_path / ex_version / benchmark_name
        measure_script_name = f"measure_{compiler}_performance_times.sh"
        measure_script_path = benchmark_subdir / measure_script_name

        if measure_script_path.exists():
            measure_script_path.unlink()
            print(f"[INFO] <remove_measure_script> Deleted measure script: {measure_script_path}")
        else:
            print(f"[ERROR] <remove_measure_script> Script not found, skipping: {measure_script_path}")

    def remove_all_measure_scripts(self, ex_version: str, benchmark_name: str, compiler_list=None):
        if compiler_list is None:
            compiler_list = ["gcc", "clang"]
        for compiler in compiler_list:
            self.remove_measure_script(ex_version, benchmark_name, compiler)

    def remove_measure_scripts_bulk(self, ex_version_list: list, benchmark_name_list: list, compiler_list: list, delete_all_llms: bool = False):
        """
        Removes measure_<compiler>_performance_times.sh for multiple EX versions and benchmarks.
        If ex_versions is None, defaults to ["EX1", "EX2", "EX3"].
        If benchmark_names is None, only proceed if delete_all_llms == True.
        If compilers is None, defaults to ["gcc", "clang"].
        """
        if ex_version_list is None:
            ex_version_list = ["EX1", "EX2", "EX3"]
        if compiler_list is None:
            compiler_list = ["gcc", "clang"]
        if not delete_all_llms and len(benchmark_name_list) == len(self.benchmark_list):
            raise ValueError("[ERROR] <remove_measure_scripts_bulk> To remove measure scripts for all benchmarks, set delete_all_llms=True.")
        if benchmark_name_list is None:
            benchmark_names = self.benchmark_list

        for ex_version in ex_version_list:
            for bm in benchmark_names:
                for c in compiler_list:
                    self.remove_measure_script(ex_version, bm, c)

        print("[INFO] <remove_measure_scripts_bulk> Bulk removal of measure scripts completed.")

    def remove_correctness_script(self, ex_version: str = "EX1", benchmark_name: str = None):
        """Automatically removes correctness_check.sh script."""
        if benchmark_name is None:
            raise ValueError("[ERROR] <remove_correctness_script> benchmark_name must be provided.")
        correctness_script_path = self.absolute_path / ex_version / benchmark_name / "correctness_check.sh"
        #correctness_script_path = self.absolute_path / ex_version / "measure_correctness.sh"
        
        if correctness_script_path.exists():
            correctness_script_path.unlink()
            print(f"[INFO] <remove_correctness_script> Deleted {correctness_script_path}")
        else:
            print(f"[ERROR] <remove_correctness_script> Script not found, skipping: {correctness_script_path}")
    
    def remove_correctness_scripts_bulk(self, ex_version: list = None, benchmark_name_list: list = None, delete_all_llms: bool = False):
        """
        Removes correctness_check.sh for multiple EX versions and benchmarks.
        If ex_versions is None, defaults to ["EX1", "EX2", "EX3"].
        If benchmark_names is None, only proceed if delete_all_llms == True.
        """
        if ex_version_list is None:
            ex_version_list = ["EX1", "EX2", "EX3"]
        if not delete_all_llms and len(benchmark_name_list) == len(self.benchmark_list):
            raise ValueError("[ERROR] <remove_correctness_scripts_bulk> To remove measure scripts for all benchmarks, set delete_all_llms=True.")
        if benchmark_name_list is None:
            benchmark_names = self.benchmark_list
        
        for ex_version in ex_version_list:
            for bm in benchmark_names:
                self.remove_correctness_script(ex_version, bm)

    def run_measure_script(self, ex_version: str, benchmark_name: str):
        """
        Run the measure_<compiler>_performance_times.sh script if it exists.
        """
        try:
            # for compiler in ["gcc", "clang"]:
            for compiler in ["gcc"]:
                measure_script_path = self.absolute_path / ex_version / benchmark_name / f"measure_{compiler}_performance_times.sh"
                if not measure_script_path.exists():
                    print(f"[INFO: Runner] {measure_script_path} not found, skip running measure script.")
                    return
                print(f"[INFO] <run_measure_script> Running measure script: {measure_script_path}")
                # Need to send Path to subprocess.run
                # subprocess.run(['bash', str(script_path)], cwd=script_path.parent, check=True)
                subprocess.run(['bash', str(measure_script_path)], check=True)
                print(f"[INFO] <run_measure_script> measure script completed")
        except Exception as e:
            # Use log_error to record the error
            self.log_error(
                ex_version=ex_version,
                benchmark=benchmark_name,
                model=None,
                path=str(self.log_file_path),
                error=e,
                level="ERROR"
            )
            
            # Optional: Print the error message to the console
            print(f"[ERROR] <run_measure_script> has Exception - Benchmark: {benchmark_name}, EX version: {ex_version}, Error: {str(e)}")

    def run_correctness_script(self, ex_version: str, benchmark_name: str):
        """
        Run the measure_correctness.sh script if it exists, using pathlib for path operations.   
        """
        try: 
            correctness_script_path = self.absolute_path / ex_version / benchmark_name / "measure_correctness.sh"
            if not correctness_script_path.exists():
                print(f"[INFO: Runner] {correctness_script_path} not found, skip running correctness script.")
                return
            
            print(f"[INFO: Runner] Running correctness script: {correctness_script_path}")
            subprocess.run(['bash', str(correctness_script_path)], check=True)
            print(f"[INFO: Runner] Correctness script running completed")
            self.compare_outputs(ex_version, benchmark_name)
            print(f"[INFO: Compare] Correctness script comparison completed")
        except Exception as e:
            # Use log_error to record the error
            self.log_error(
                ex_version=ex_version,
                benchmark=benchmark_name,
                model=None,
                path=str(self.log_file_path),
                error=e,
                level="ERROR"
            )
            
            # Optional: Print the error message to the console
            print(f"[ERROR] <run_correctness_script> has Exception - Benchmark: {benchmark_name}, EX version: {ex_version}, Error: {str(e)}")
    
    def run_measure_script_e(self, benchmark_name: str):
        """
        Iterate EX1, EX2, EX3 for run_measure_script
        """
        for ex_version in ["EX1", "EX2", "EX3"]:
            self.run_measure_script(ex_version, benchmark_name)
    
    def run_correctness_script_e(self, benchmark_name: str):
        """
        Iterate EX1, EX2, EX3 for run_correctness_script
        """
        for ex_version in ["EX1", "EX2", "EX3"]:
            self.run_correctness_script(ex_version, benchmark_name)
    
    def run_measure_scripts_all(self):
        for benchmark_name in self.benchmark_list:
            self.run_measure_script_e(benchmark_name)
    
    def run_correctness_scripts_all(self):
        for benchmark_name in self.benchmark_list:
            self.run_correctness_script_e(benchmark_name)
    
    def run_scripts_all(self):
        self.run_all_measure_scripts()
        self.run_all_correctness_scripts()

"""
# Reference Format for class Benchmarks
if __name__ == "__main__":
    benchmark_runner = Benchmarks(absolute_path='D:\\CS_PhD\\research\\NeurIPS\\target_code\\NeurIPS_25', llm_model='deepseek-ai/DeepSeek-R1', llm_abbrev='dpR1')
    benchmark_runner.run_all_benchmarks()  # Run all benchmarks
    benchmark_runner.write_error_log()  # Log any errors encountered during the runs
    print("[INFO: ScriptManager] All scripts executed successfully.")  # Final message
"""


def _common_cli_options() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
        help="Repository root (defaults to repository base directory)",
    )
    common.add_argument(
        "--llm-model",
        default="o4-mini-2025-04-16",
        help="Default LLM model identifier",
    )
    common.add_argument(
        "--llm-abbrev",
        default="o4",
        help="Abbreviation for the LLM model",
    )
    common.add_argument(
        "--model-family",
        default="openai",
        help="LLM provider family (openai, together.ai, etc.)",
    )
    return common


def build_cli_parser():
    common = _common_cli_options()
    benchmark_parent = argparse.ArgumentParser(add_help=False)
    benchmark_parent.add_argument(
        "--llm-guard",
        action="store_true",
        help="Enable approximate token-limit guard when preparing LLM prompts",
    )
    benchmark_parent.add_argument(
        "--llm-guard-limit",
        type=int,
        default=12000,
        help="Maximum approximate token count allowed when --llm-guard is enabled",
    )
    benchmark_parent.add_argument(
        "--log-file",
        help="Custom path for benchmark error/activity logs",
    )
    benchmark_parent.add_argument(
        "--activity-log",
        help="Path for verbose activity logs (JSON lines)",
    )
    benchmark_parent.add_argument(
        "--write-state",
        action="store_true",
        help="Persist optimization explanations alongside generated code",
    )

    parser = argparse.ArgumentParser(description="Benchmark automation toolkit")
    subparsers = parser.add_subparsers(dest="command")

    tm = subparsers.add_parser(
        "time_measurement",
        parents=[common],
        help="Scan and measure already-built executables",
    )
    tm.add_argument(
        "--ex-version",
        required=True,
        choices=TIME_MEASUREMENT_EX_CHOICES,
        help="Experiment directory to inspect (EX1, EX2, or EX3).",
    )
    tm.add_argument(
        "--benchmark",
        required=True,
        help="Benchmark name(s) (comma-separated or 'all', e.g., 2mm,3mm)",
    )
    tm.add_argument("--models", help="Comma-separated model filters (use 'baseline' for originals)")
    tm.add_argument("--datasets", help="Comma-separated dataset names (e.g., mini,small,medium)")
    tm.add_argument("--versions", help="Comma-separated version numbers (e.g., 1,2,3)")
    tm.add_argument("--limit", type=int, help="Maximum number of executables to process")
    tm.add_argument("--output", help="Destination JSONL file (default: results/time_measurements/<ex>/<benchmark>/...)")
    tm.add_argument("--runtime-hints", help="Path to runtime_hints.json (relative to repo root or absolute)")
    tm.add_argument("--warmup", type=int, dest="warmup_override", help="Override warm-up iterations")
    tm.add_argument("--runs", type=int, dest="runs_override", help="Override measurement iterations")
    tm.add_argument("--trail", type=int, default=0, help="Trail/experiment identifier recorded in JSON")
    tm.add_argument("--stop-on-error", action="store_true", help="Stop immediately if any run fails")
    tm.add_argument("--dry-run", dest="dry_run", action="store_true", help="Only print the plan (default)")
    tm.add_argument("--no-dry-run", dest="dry_run", action="store_false", help="Execute the measurements")
    tm.set_defaults(dry_run=True)
    tm.add_argument(
        "--activity-log",
        help="Path for verbose activity logs (JSON lines). Default: logs/activity_time_measurement.log",
    )
    tm.add_argument(
        "--output-mode",
        choices=["append", "overwrite"],
        default="append",
        help="Append to or overwrite the measurement JSONL output (default: append)",
    )
    tm.add_argument(
        "--fast-run",
        action="store_true",
        help="Skip warmups and record only one measurement run (overrides --warmup/--runs)",
    )
    tm.add_argument(
        "--save-outputs",
        action="store_true",
        help="[EX3 only] Save stdout to EX3_correctness during time measurement for later correctness checks",
    )

    bench = subparsers.add_parser(
        "benchmark",
        parents=[common, benchmark_parent],
        help="Run LLM-based optimization workflows",
    )
    bench_sub = bench.add_subparsers(dest="benchmark_action")
    bench_sub.required = True

    list_cmd = bench_sub.add_parser(
        "list",
        parents=[common, benchmark_parent],
        help="Display resolved benchmark names for a selection",
    )
    list_cmd.add_argument(
        "--benchmarks",
        default="all",
        help="Comma-separated benchmark names or 'all' (default)",
    )

    auto = bench_sub.add_parser(
        "auto-run",
        parents=[common, benchmark_parent],
        help="Invoke auto_run_EX1/EX2 for selected benchmarks",
    )
    auto.add_argument(
        "--ex-version",
        required=True,
        choices=AUTO_RUN_EX_CHOICES,
        help="Experiment directory (EX1 or EX2)",
    )
    auto.add_argument("--benchmarks", required=True,
                      help="Comma-separated benchmark names or 'all'")
    auto.add_argument("--max-attempts", type=int, default=10,
                      help="Maximum number of LLM attempts per benchmark")

    build_cmd = bench_sub.add_parser(
        "build",
        parents=[common, benchmark_parent],
        help="Run make in each benchmark directory before measurements",
    )
    build_cmd.add_argument(
        "--benchmarks",
        default="all",
        help="Comma-separated benchmark names or 'all' (default)",
    )
    build_cmd.add_argument(
        "--ex-versions",
        required=True,
        help="Comma-separated EX directories to build, e.g. EX1,EX2,EX3",
    )
    build_cmd.add_argument(
        "--log-root",
        default="make_results",
        help="Location under results/ for build logs (default: results/make_results)",
    )
    build_cmd.add_argument(
        "--no-logs",
        action="store_true",
        help="Disable writing build logs and summary files",
    )
    build_cmd.add_argument(
        "--clean",
        action="store_true",
        help="Run `make clean` before the default target for each benchmark",
    )
    build_cmd.add_argument(
        "--clean-only",
        action="store_true",
        help="Only run `make clean` (skip the default `make` target)",
    )

    check_cmd = bench_sub.add_parser(
        "check",
        parents=[common, benchmark_parent],
        help="Run correctness validation for selected benchmarks",
    )
    check_cmd.add_argument(
        "--benchmarks",
        default="all",
        help="Comma-separated benchmark names or 'all' (default)",
    )
    check_cmd.add_argument(
        "--ex-versions",
        required=True,
        help="Comma-separated EX directories to check, e.g. EX1,EX2,EX3",
    )
    check_cmd.add_argument(
        "--datasets",
        help="Comma-separated dataset filters (mini,small,...) or 'all'",
    )

    capture_cmd = bench_sub.add_parser(
        "correctness-run",
        parents=[common, benchmark_parent],
        help="Run executables once and capture outputs for correctness checks",
    )
    capture_cmd.add_argument(
        "--benchmarks",
        default="all",
        help="Comma-separated benchmark names or 'all' (default)",
    )
    capture_cmd.add_argument(
        "--ex-versions",
        required=True,
        help="Comma-separated EX directories to run, e.g. EX1,EX2,EX3",
    )
    capture_cmd.add_argument(
        "--models",
        help="Comma-separated model filters (e.g., baseline,gpt5.1)",
    )
    capture_cmd.add_argument(
        "--datasets",
        help="Comma-separated dataset filters (mini,small,...) or 'all'",
    )
    capture_cmd.add_argument(
        "--versions",
        help="Comma-separated version numbers (e.g., 1,2,3) or 'all'",
    )
    capture_cmd.add_argument(
        "--capture-activity-log",
        help="Custom path for correctness capture activity log (default: logs/activity_correctness_capture.log)",
    )
    capture_cmd.add_argument(
        "--capture-error-log",
        help="Custom path for correctness capture error log (default: logs/correctness_capture_errors.log)",
    )

    clean = bench_sub.add_parser(
        "cleanup",
        parents=[common, benchmark_parent],
        help="Delete generated optimization artifacts",
    )
    clean.add_argument("--benchmarks", required=True,
                       help="Comma-separated benchmark names or 'all'")
    clean.add_argument("--ex-versions", required=True,
                       help="Comma-separated EX directories to clean, e.g. EX1,EX2,EX3")
    clean.add_argument("--delete-all-llms", action="store_true",
                       help="Remove files for every LLM abbreviation instead of the current one only")

    return parser


def _resolve_benchmark_names(runner, raw_value: str, parser: argparse.ArgumentParser, ex_version: Optional[str] = None, ex_versions: Optional[List[str]] = None) -> List[str]:
    available = runner.available_benchmarks(ex_version=ex_version, ex_versions=ex_versions)
    requested = parse_csv_list(raw_value)
    if not requested or any(item.lower() == "all" for item in requested):
        return available
    missing = [item for item in requested if item not in available]
    if missing:
        parser.error(f"Unknown benchmark(s): {', '.join(missing)}")
    return requested


def handle_benchmark_cli(args, parser: argparse.ArgumentParser):
    from scripts import Benchmarks

    runner = Benchmarks(
        absolute_path=args.root,
        llm_model=args.llm_model,
        llm_abbrev=args.llm_abbrev,
        model_family=args.model_family,
        write_state=args.write_state,
        log_file=args.log_file,
        activity_log=args.activity_log,
        llm_guard_enabled=args.llm_guard,
        llm_guard_limit=args.llm_guard_limit,
    )
    runner.configure_guard(args.llm_guard, args.llm_guard_limit)
    action = args.benchmark_action
    
    # Resolve benchmarks based on EX version(s)
    ex_version_for_resolve = None
    ex_versions_for_resolve = None
    if action == "auto-run":
        ex_version_for_resolve = getattr(args, 'ex_version', None)
    elif action == "build":
        # For build, resolve benchmarks based on specified EX versions
        ex_versions_for_resolve = parse_csv_list(getattr(args, 'ex_versions', None))
    
    targets = _resolve_benchmark_names(runner, args.benchmarks, parser, ex_version=ex_version_for_resolve, ex_versions=ex_versions_for_resolve)

    if action == "list":
        ordered = sorted(targets)
        print(f"[INFO] Resolved {len(ordered)} benchmark(s):")
        for name in ordered:
            print(f" - {name}")
        return

    if action == "auto-run":
        selected_ex = args.ex_version
        if selected_ex == "EX1":
            for name in targets:
                runner.auto_run_EX1(name, max_attempts=args.max_attempts, ex_version=selected_ex)
        elif selected_ex == "EX2":
            for name in targets:
                runner.auto_run_EX2(name, max_attempts=args.max_attempts, ex_version=selected_ex)
        elif selected_ex == "EX3":
            for name in targets:
                runner.auto_run_EX3(name, max_attempts=args.max_attempts, ex_version=selected_ex)
        else:
            parser.error(f"auto-run does not support EX version '{selected_ex}'.")
        return

    if action == "build":
        ex_versions = parse_csv_list(args.ex_versions)
        if not ex_versions:
            parser.error("--ex-versions must specify at least one experiment name (e.g., EX1,EX2,EX3)")
        clean_flag = bool(args.clean or args.clean_only)
        log_enabled = not getattr(args, "no_logs", False)
        for name in targets:
            runner.run_make(
                name,
                ex_versions,
                log_root=args.log_root,
                clean=clean_flag,
                clean_only=args.clean_only,
                enable_logs=log_enabled,
            )
        log_root_path = None
        if log_enabled:
            log_root_path = runner.log_build_summary(
                benchmarks=targets,
                ex_versions=ex_versions,
                log_root=args.log_root,
                clean=clean_flag,
                clean_only=args.clean_only,
            )
        mode_desc = (
            "clean-only" if args.clean_only else ("clean+build" if args.clean else "build")
        )
        log_msg = (
            f"Logs stored in {log_root_path}."
            if log_root_path
            else "Logs disabled (--no-logs)."
        )
        print(
            f"[INFO] Benchmark build ({mode_desc}) finished for {len(targets)} benchmark(s) "
            f"on {', '.join(ex_versions)}. {log_msg}"
        )
        return

    if action == "check":
        ex_versions = parse_csv_list(args.ex_versions)
        if not ex_versions:
            parser.error("--ex-versions must specify at least one experiment name (e.g., EX1,EX2,EX3)")
        dataset_tokens = parse_csv_list(getattr(args, "datasets", None))
        dataset_list = None
        if dataset_tokens:
            lowered = [token.lower() for token in dataset_tokens]
            if any(token == "all" for token in lowered):
                if len(dataset_tokens) > 1:
                    parser.error("--datasets cannot mix 'all' with explicit names.")
                dataset_list = None
            else:
                dataset_list = dataset_tokens
        checker_runner = CorrectnessRunner(
            absolute_path=args.root,
            log_file=args.log_file,
            activity_log=args.activity_log,
        )
        summary = checker_runner.run_checks(targets, ex_versions, datasets=dataset_list)
        print(
            f"[INFO] Correctness summary: total={summary['total']}, "
            f"pass={summary['pass']}, skip={summary['skip']}, "
            f"fail={summary['fail']}, error={summary['error']}."
        )
        file_summary = summary.get("file_summary", {})
        print(
            "[INFO] Correctness summary (files): "
            f"total={file_summary.get('total', 0)}, "
            f"pass={file_summary.get('pass', 0)}, "
            f"fail={file_summary.get('fail', 0)}, "
            f"skip={file_summary.get('skip', 0)}, "
            f"error={file_summary.get('error', 0)}, "
            f"unknown={file_summary.get('unknown', 0)}."
        )
        if summary["fail"] > 0 or summary["error"] > 0:
            sys.exit(1)
        return

    if action == "correctness-run":
        ex_versions = parse_csv_list(args.ex_versions)
        if not ex_versions:
            parser.error("--ex-versions must specify at least one experiment name (e.g., EX1,EX2,EX3)")
        model_list = parse_csv_list(args.models)
        dataset_tokens = parse_csv_list(args.datasets)
        dataset_list = None
        if dataset_tokens:
            lowered_datasets = [token.lower() for token in dataset_tokens]
            if any(token == "all" for token in lowered_datasets):
                if len(dataset_tokens) > 1:
                    parser.error("--datasets cannot mix 'all' with explicit names.")
                dataset_list = None
            else:
                dataset_list = dataset_tokens
        version_tokens = parse_csv_list(args.versions)
        version_list = None
        versions_all = False
        if version_tokens:
            lowered_versions = [token.lower() for token in version_tokens]
            if any(token == "all" for token in lowered_versions):
                if len(version_tokens) > 1:
                    parser.error("--versions cannot mix 'all' with explicit integers.")
                versions_all = True
            else:
                try:
                    version_list = [int(token) for token in version_tokens]
                except ValueError:
                    parser.error("--versions must be a comma-separated list of integers (e.g., 1,2,3)")
        runner.configure_correctness_logging(args.capture_activity_log, args.capture_error_log)
        for name in targets:
            runner.capture_correctness_outputs(
                name,
                ex_versions,
                models=model_list,
                datasets=dataset_list,
                versions=version_list,
                versions_all=versions_all,
            )
        return

    if action == "cleanup":
        ex_versions = parse_csv_list(args.ex_versions)
        if not ex_versions:
            parser.error("--ex-versions must specify at least one experiment name (e.g., EX1,EX2,EX3)")
        for name in targets:
            runner.cleanup_generated_files(name, ex_versions, delete_all_llms=args.delete_all_llms)
        return

    parser.error(f"Unsupported benchmark action: {action}")


def main():
    parser = build_cli_parser()
    args = parser.parse_args()

    if args.command == "time_measurement":
        script_manager = ScriptManager(absolute_path=args.root)
        model_list = parse_csv_list(args.models)
        dataset_list = parse_csv_list(args.datasets)
        if dataset_list:
            lowered_datasets = [token.lower() for token in dataset_list]
            if any(token == "all" for token in lowered_datasets):
                if len(dataset_list) > 1:
                    parser.error("--datasets cannot mix 'all' with explicit names.")
                dataset_list = None
        version_tokens = parse_csv_list(args.versions)
        version_list = None
        versions_all = False
        if version_tokens:
            lowered_versions = [token.lower() for token in version_tokens]
            if any(token == "all" for token in lowered_versions):
                if len(version_tokens) > 1:
                    parser.error("--versions cannot mix 'all' with explicit integers.")
                versions_all = True
            else:
                try:
                    version_list = [int(token) for token in version_tokens]
                except ValueError:
                    parser.error("--versions must be a comma-separated list of integers (e.g., 1,2,3)")
        # Get available benchmarks for the specified ex_version
        from scripts import Benchmarks
        temp_runner = Benchmarks(
            absolute_path=args.root,
            llm_model=getattr(args, 'llm_model', 'deepseek-ai/DeepSeek-R1'),
            llm_abbrev=getattr(args, 'llm_abbrev', 'dpR1'),
            model_family=getattr(args, 'model_family', 'together.ai'),
            write_state=getattr(args, 'write_state', False),
        )
        available_for_ex = temp_runner.available_benchmarks(ex_version=args.ex_version)
        
        benchmark_tokens = parse_csv_list(args.benchmark) or [args.benchmark]
        if any(token.lower() == "all" for token in benchmark_tokens if token):
            target_benchmarks = available_for_ex
        else:
            target_benchmarks = [token for token in benchmark_tokens if token]
        if not target_benchmarks:
            parser.error("--benchmark must specify at least one benchmark name or 'all'.")

        available_set = set(available_for_ex)
        missing = [name for name in target_benchmarks if name not in available_set]
        if missing:
            parser.error(f"Unknown benchmark(s): {', '.join(missing)}")

        if args.output and len(target_benchmarks) > 1:
            parser.error("--output can only be specified when targeting a single benchmark.")

        overall_results = []
        failure_statuses = {"partial_failure", "missing_only", "no_runs", "failed"}
        has_failure = False
        for benchmark_name in target_benchmarks:
            summary = script_manager.time_measurement(
                ex_version=args.ex_version,
                benchmark_name=benchmark_name,
                models=model_list,
                datasets=dataset_list,
                versions=version_list,
                all_versions=versions_all,
                limit=args.limit,
                dry_run=args.dry_run,
                output_path=args.output,
                runtime_hints=args.runtime_hints,
                warmup_time=args.warmup_override,
                num_runs=args.runs_override,
                trail=args.trail,
                stop_on_error=args.stop_on_error,
                activity_log=args.activity_log,
                output_mode=args.output_mode,
                fast_run=args.fast_run,
                save_outputs=args.save_outputs,
            )
            if summary:
                overall_results.append(summary)
                if summary.get("status") in failure_statuses:
                    has_failure = True

        if overall_results:
            total_plans = sum(item.get("plans", 0) for item in overall_results)
            total_success = sum(item.get("successful", 0) for item in overall_results)
            total_failed = sum(item.get("failed", 0) for item in overall_results)
            total_missing = sum(item.get("missing", 0) for item in overall_results)
            print(
                f"[INFO] Time measurement summary: {len(overall_results)} benchmark(s), "
                f"plans={total_plans}, successful_execs={total_success}, "
                f"failed_execs={total_failed}, missing={total_missing}."
            )

        if has_failure:
            sys.exit(1)
        return
    if args.command == "benchmark":
        handle_benchmark_cli(args, parser)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
