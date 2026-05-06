#!/usr/bin/env python3
"""
汇总多个 benchmark 的 speedup（kernel/total）统计：按 dataset+model 计算均值/最小/最大。

示例：
    python scripts/analysis/aggregate_speedups.py --benchmarks srad_v2,3mm --ex-version EX1
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.analysis_time_measurements import (
    parse_list,
    discover_benchmarks,
    load_summary_records,
    summarize_records,
    load_missing_entries,
)

DATASET_ORDER = {
    "mini": 0,
    "small": 1,
    "medium": 2,
    "large": 3,
    "extra-large": 4,
    "default": 5,
    "all": 6,
}


def dataset_rank(name: Optional[str]) -> int:
    if not name:
        return DATASET_ORDER["default"]
    return DATASET_ORDER.get(name, len(DATASET_ORDER))

def aggregate_speedup_rows(rows: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, str, Optional[int]], Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, Optional[int]], Dict[str, Any]] = {}
    for row in rows:
        if row.get("is_baseline"):
            continue
        dataset = row.get("dataset") or "default"
        model = row.get("model") or "-"
        threads = row.get("threads")
        key = (dataset, model, threads)
        bucket = grouped.setdefault(
            key,
            {
                "dataset": dataset,
                "model": model,
                "threads": threads,
                "versions": set(),
                "kernel_speedups": [],
                "total_speedups": [],
                "sample_count": 0,
            },
        )
        if row.get("version") is not None:
            bucket["versions"].add(row.get("version"))
        bucket["sample_count"] += 1
        if row.get("kernel_speedup") is not None:
            bucket["kernel_speedups"].append(row["kernel_speedup"])
        if row.get("total_speedup") is not None:
            bucket["total_speedups"].append(row["total_speedup"])
    return grouped


def calc_stats(values: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not values:
        return None, None, None
    return mean(values), min(values), max(values)


def annotate_missing(grouped: Dict[Tuple[str, str, Optional[int]], Dict[str, Any]],
                     missing_entries: Iterable[Dict[str, Any]]) -> None:
    for entry in missing_entries:
        dataset = entry.get("dataset") or "all"
        if dataset == "default":
            dataset = "all"
        model = entry.get("model")
        if not dataset or not model:
            continue
        key = (dataset, model, None)
        bucket = grouped.setdefault(
            key,
            {
                "dataset": dataset,
                "model": model,
                "threads": None,
                "versions": set(),
                "kernel_speedups": [],
                "total_speedups": [],
                "sample_count": 0,
            },
        )
        misses = bucket.setdefault("missing_versions", [])
        misses.append(entry.get("version"))


def print_table(per_benchmark_stats: Dict[str, List[Dict[str, Any]]]) -> None:
    for benchmark, stats in per_benchmark_stats.items():
        print(f"\n=== Aggregated Speedup: {benchmark} ===")
        if not stats:
            print("  (no data)")
            continue
        header = (
            f"{'dataset':<12} {'model':<12} {'thr':>5} {'samples':>8} "
            f"{'k_avg':>8} {'k_min':>8} {'k_max':>8} "
            f"{'t_avg':>8} {'t_min':>8} {'t_max':>8} {'miss_versions':>15}"
        )
        print(header)
        for entry in stats:
            k_avg, k_min, k_max = calc_stats(entry["kernel_speedups"])
            t_avg, t_min, t_max = calc_stats(entry["total_speedups"])
            missing_versions = entry.get("missing_versions")
            miss_str = ",".join(str(v) for v in sorted(missing_versions)) if missing_versions else "-"
            print(
                f"{entry['dataset']:<12} {entry['model']:<12} {str(entry.get('threads') if entry.get('threads') is not None else '-'):>5} "
                f"{entry['sample_count']:>8d} "
                f"{(f'{k_avg:.2f}' if k_avg is not None else '-'):>8} "
                f"{(f'{k_min:.2f}' if k_min is not None else '-'):>8} "
                f"{(f'{k_max:.2f}' if k_max is not None else '-'):>8} "
                f"{(f'{t_avg:.2f}' if t_avg is not None else '-'):>8} "
                f"{(f'{t_min:.2f}' if t_min is not None else '-'):>8} "
                f"{(f'{t_max:.2f}' if t_max is not None else '-'):>8} "
                f"{miss_str:>15}"
            )


def resolve_speedup_dir(base: Optional[Path]) -> Optional[Path]:
    if base is None:
        return None
    if base.name.lower() == "speedup":
        base.mkdir(parents=True, exist_ok=True)
        return base
    target = base / "speedup"
    target.mkdir(parents=True, exist_ok=True)
    return target


def maybe_write_csv(output_dir: Optional[Path], benchmark: str, stats: List[Dict[str, Any]]) -> None:
    if not output_dir or not stats:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{benchmark}_speedup_stats.csv"
    fieldnames = [
        "dataset",
        "model",
        "threads",
        "samples",
        "kernel_speedup_avg",
        "kernel_speedup_min",
        "kernel_speedup_max",
        "total_speedup_avg",
        "total_speedup_min",
        "total_speedup_max",
        "missing_versions",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in stats:
            k_avg, k_min, k_max = calc_stats(entry["kernel_speedups"])
            t_avg, t_min, t_max = calc_stats(entry["total_speedups"])
            writer.writerow(
                {
                    "dataset": entry["dataset"],
                    "model": entry["model"],
                    "threads": entry.get("threads"),
                    "samples": entry["sample_count"],
                    "kernel_speedup_avg": k_avg,
                    "kernel_speedup_min": k_min,
                    "kernel_speedup_max": k_max,
                    "total_speedup_avg": t_avg,
                    "total_speedup_min": t_min,
                    "total_speedup_max": t_max,
                    "missing_versions": ",".join(str(v) for v in sorted(entry.get("missing_versions", []))) or "",
                }
            )
    print(f"[INFO] CSV written: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate speedup stats by dataset/model.")
    parser.add_argument("--benchmarks", default="all", help="Comma-separated benchmark names or 'all'.")
    parser.add_argument(
        "--ex-version",
        default="EX1",
        help="Experiment version to use (e.g., EX1, EX2).",
    )
    parser.add_argument("--results-root", default="results/time_measurements", help="Root directory for measurement JSONLs.")
    parser.add_argument("--missing-root", default="results/missing_bins", help="Root directory for missing bin reports.")
    parser.add_argument("--trails", help="Comma-separated trail IDs to include (default: all).")
    parser.add_argument("--output-dir", help="Optional directory to write CSV output.")
    args = parser.parse_args()

    benchmarks = discover_benchmarks(Path(args.results_root), parse_list(args.benchmarks))
    if not benchmarks:
        print(f"[WARN] No benchmarks found under {args.results_root}")
        return
    trail_ids = [int(token) for token in parse_list(args.trails)] if args.trails else None
    base_output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    if base_output_dir:
        base_output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = resolve_speedup_dir(base_output_dir)

    per_benchmark_stats: Dict[str, List[Dict[str, Any]]] = {}
    for benchmark in benchmarks:
        measurement_path = Path(args.results_root) / benchmark / f"{args.ex_version}_time_measurements.jsonl"
        missing_path = Path(args.missing_root) / benchmark / f"{args.ex_version}_missing_bins.json"
        summaries = load_summary_records(measurement_path, trail_ids)
        rows = summarize_records(summaries)
        grouped = aggregate_speedup_rows(rows)
        missing_entries = load_missing_entries(missing_path)
        annotate_missing(grouped, missing_entries)
        stats_list = list(grouped.values())
        stats_list.sort(key=lambda entry: (dataset_rank(entry["dataset"]), entry["model"], entry.get("threads") if entry.get("threads") is not None else -1))
        per_benchmark_stats[benchmark] = stats_list
        maybe_write_csv(output_dir, benchmark, stats_list)

    print_table(per_benchmark_stats)


if __name__ == "__main__":
    main()

