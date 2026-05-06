#!/usr/bin/env python3
"""
批量为指定 benchmark 生成两类分析结果：
1. 详细 summary：与 analysis_time_measurements 相同字段，写入 CSV。
2. speedup 汇总：与 aggregate_speedups 相同字段，写入 CSV。

示例：
    python scripts/analysis/generate_benchmark_summaries.py \
        --benchmarks 3mm,srad_v2 \
        --ex-version EX1 \
        --trails 1 \
        --output-dir analysis_summaries
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.analysis_time_measurements import (
    parse_list,
    discover_benchmarks,
    load_summary_records,
    summarize_records,
    load_missing_entries,
    dataset_rank,
)
from scripts.analysis.aggregate_speedups import (
    aggregate_speedup_rows,
    annotate_missing,
    calc_stats,
    dataset_rank as speedup_dataset_rank,
)


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_category_dir(base: Optional[Path], category: str) -> Optional[Path]:
    if base is None:
        return None
    if base.name.lower() == category:
        base.mkdir(parents=True, exist_ok=True)
        return base
    target = base / category
    return ensure_output_dir(target)


def write_detail_csv(output_dir: Optional[Path], benchmark: str, ex_version: str, rows: List[Dict[str, Any]]) -> Optional[Path]:
    if not output_dir or not rows:
        return None
    ordered_rows = sorted(
        rows,
        key=lambda row: (
            dataset_rank(row["dataset"]),
            row["trail"],
            -1 if row.get("threads") is None else row.get("threads"),
            0 if row.get("is_baseline") else 1,
            row.get("model"),
            row.get("version") or 0,
            row.get("sequence_id") or 0,
        ),
    )
    detail_path = ensure_output_dir(output_dir) / f"{benchmark}_{ex_version}_detail.csv"
    fieldnames = [
        "dataset",
        "trail",
        "model",
        "threads",
        "version",
        "compiler",
        "is_baseline",
        "kernel_mean_s",
        "total_mean_s",
        "kernel_speedup",
        "total_speedup",
        "success_rate",
        "warmup_runs",
        "measure_runs",
        "hint_key",
        "executable",
    ]
    with detail_path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in ordered_rows:
            writer.writerow(
                {
                    "dataset": row["dataset"],
                    "trail": row["trail"],
                    "model": row["model"],
                    "threads": row.get("threads"),
                    "version": row.get("version"),
                    "compiler": row.get("compiler"),
                    "is_baseline": row.get("is_baseline"),
                    "kernel_mean_s": row.get("kernel_mean_s"),
                    "total_mean_s": row.get("total_mean_s"),
                    "kernel_speedup": row.get("kernel_speedup"),
                    "total_speedup": row.get("total_speedup"),
                    "success_rate": row.get("success_rate"),
                    "warmup_runs": row.get("warmup_runs"),
                    "measure_runs": row.get("measure_runs"),
                    "hint_key": row.get("hint_key"),
                    "executable": row.get("executable"),
                }
            )
    return detail_path


def write_speedup_csv(output_dir: Optional[Path], benchmark: str, ex_version: str, grouped_stats: List[Dict[str, Any]]) -> Optional[Path]:
    if not output_dir or not grouped_stats:
        return None
    ordered_stats = sorted(
        grouped_stats,
        key=lambda entry: (
            speedup_dataset_rank(entry["dataset"]),
            entry["model"],
            entry.get("threads") if entry.get("threads") is not None else -1,
        ),
    )
    out_path = ensure_output_dir(output_dir) / f"{benchmark}_{ex_version}_speedup_stats.csv"
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
        for entry in ordered_stats:
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
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate detail + speedup summaries for multiple benchmarks.")
    parser.add_argument("--benchmarks", required=True, help="Comma-separated benchmark list (e.g., 3mm,srad_v2).")
    parser.add_argument(
        "--ex-version",
        default="EX1",
        help="Experiment version (EX1, EX2, or EX3).",
    )
    parser.add_argument("--results-root", default="results/time_measurements", help="Root for measurement JSONL files.")
    parser.add_argument("--missing-root", default="results/missing_bins", help="Root for missing bin JSON files.")
    parser.add_argument("--trails", help="Comma-separated trail IDs to include (default: all).")
    parser.add_argument(
        "--output-dir",
        default="analysis_summaries",
        help="Base directory to store outputs (detail in 'summaries/', aggregate in 'speedup/').",
    )
    parser.add_argument(
        "--output-dir-detail",
        help="Override directory for detail CSVs (if unset, falls back to <output-dir>/summaries).",
    )
    parser.add_argument(
        "--output-dir-aggregate",
        help="Override directory for aggregate CSVs (if unset, falls back to <output-dir>/speedup).",
    )
    args = parser.parse_args()

    base_output_dir = ensure_output_dir(Path(args.output_dir).resolve()) if args.output_dir else None
    if args.output_dir_detail:
        detail_output_dir = ensure_output_dir(Path(args.output_dir_detail).resolve())
    else:
        detail_output_dir = resolve_category_dir(base_output_dir, "summaries")
    if args.output_dir_aggregate:
        speedup_output_dir = ensure_output_dir(Path(args.output_dir_aggregate).resolve())
    else:
        speedup_output_dir = resolve_category_dir(base_output_dir, "speedup")
    trail_ids = [int(token) for token in parse_list(args.trails)] if args.trails else None
    benchmarks = discover_benchmarks(Path(args.results_root), parse_list(args.benchmarks))
    if not benchmarks:
        print("[WARN] No benchmark matched the provided list.")
        return

    for benchmark in benchmarks:
        measurement_path = Path(args.results_root) / benchmark / f"{args.ex_version}_time_measurements.jsonl"
        missing_path = Path(args.missing_root) / benchmark / f"{args.ex_version}_missing_bins.json"
        summaries = load_summary_records(measurement_path, trail_ids)
        rows = summarize_records(summaries)
        detail_path = write_detail_csv(detail_output_dir, benchmark, args.ex_version, rows)

        grouped = aggregate_speedup_rows(rows)
        missing_entries = load_missing_entries(missing_path)
        annotate_missing(grouped, missing_entries)
        speedup_path = write_speedup_csv(speedup_output_dir, benchmark, args.ex_version, list(grouped.values()))

        if detail_path:
            print(f"[INFO] Detail summary saved: {detail_path}")
        else:
            print(f"[WARN] No summary rows found for {benchmark}.")
        if speedup_path:
            print(f"[INFO] Speedup summary saved: {speedup_path}")
        else:
            print(f"[WARN] No speedup data available for {benchmark}.")


if __name__ == "__main__":
    main()

