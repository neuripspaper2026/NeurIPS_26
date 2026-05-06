from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import json

DEFAULT_ROOT = Path(__file__).resolve().parent.parent


def load_benchmark_arg_map(config_path: Optional[Path] = None) -> Dict[str, Dict[str, List[str]]]:
    """
    Load benchmark runtime arguments from JSON.
    Expected schema: { "benchmark": { "dataset": ["arg1", ...] } }.
    """
    path = Path(config_path) if config_path else DEFAULT_ROOT / "configs" / "benchmark_args.json"
    if not path.exists():
        print(f"[INFO] Benchmark argument config {path} not found; using empty map.")
        return {}
    try:
        with path.open("r", encoding="utf-8") as fp:
            raw_data = json.load(fp)
    except Exception as exc:
        print(f"[WARN] Failed to load benchmark argument config {path}: {exc}")
        return {}
    if not isinstance(raw_data, dict):
        print(f"[WARN] Benchmark argument config {path} must be a JSON object.")
        return {}
    normalized: Dict[str, Dict[str, List[str]]] = {}
    for bench_name, dataset_map in raw_data.items():
        if not isinstance(dataset_map, dict):
            continue
        bench_entry: Dict[str, List[str]] = {}
        for dataset_key, args in dataset_map.items():
            if not isinstance(args, list):
                continue
            bench_entry[str(dataset_key).lower()] = [str(item) for item in args]
        if bench_entry:
            normalized[str(bench_name)] = bench_entry
    return normalized


BENCHMARK_ARG_MAP: Dict[str, Dict[str, List[str]]] = load_benchmark_arg_map()


def load_benchmark_arg_map_ex3(config_path: Optional[Path] = None) -> Dict[str, Dict[str, List[str]]]:
    """
    Load EX3 benchmark runtime arguments from JSON.
    Expected schema: { "benchmark": { "dataset": ["arg1", ...] } }.
    """
    path = Path(config_path) if config_path else DEFAULT_ROOT / "configs" / "EX3_configs" / "benchmark_args_ex3.json"
    if not path.exists():
        print(f"[INFO] EX3 Benchmark argument config {path} not found; using empty map.")
        return {}
    try:
        with path.open("r", encoding="utf-8") as fp:
            raw_data = json.load(fp)
    except Exception as exc:
        print(f"[WARN] Failed to load EX3 benchmark argument config {path}: {exc}")
        return {}
    if not isinstance(raw_data, dict):
        print(f"[WARN] EX3 Benchmark argument config {path} must be a JSON object.")
        return {}
    normalized: Dict[str, Dict[str, List[str]]] = {}
    for bench_name, dataset_map in raw_data.items():
        if not isinstance(dataset_map, dict):
            continue
        bench_entry: Dict[str, List[str]] = {}
        for dataset_key, args in dataset_map.items():
            if not isinstance(args, list):
                continue
            bench_entry[str(dataset_key).lower()] = [str(item) for item in args]
        if bench_entry:
            normalized[str(bench_name)] = bench_entry
    return normalized


BENCHMARK_ARG_MAP_EX3: Dict[str, Dict[str, List[str]]] = load_benchmark_arg_map_ex3()


