#!/usr/bin/env python3
"""
按照 benchmark 所属 motif 统计平均 speedup。

示例：
    python scripts/analysis/motif_speedup.py \
        --aggregate-root analysis_summaries/speedup/aggregate \
        --ex-versions EX1,EX2 \
        --output-dir analysis_summaries/motif_speedup
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.motif import BENCHMARK_TO_MOTIF, motif_of  # noqa: E402

DATASET_ORDER: Dict[str, int] = {
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


def parse_list(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    tokens = []
    for chunk in raw.replace(";", ",").split(","):
        item = chunk.strip()
        if item:
            tokens.append(item)
    return tokens or None


def load_aggregate_frames(aggregate_root: Path, ex_version: str) -> pd.DataFrame:
    target_dir = aggregate_root / ex_version
    if not target_dir.exists():
        print(f"[WARN] Aggregate directory not found: {target_dir}")
        return pd.DataFrame()
    frames: List[pd.DataFrame] = []
    suffix = f"_{ex_version}_speedup_stats.csv"
    for csv_file in sorted(target_dir.glob(f"*{suffix}")):
        benchmark = csv_file.name[: -len(suffix)]
        df = pd.read_csv(csv_file)
        if df.empty:
            continue
        df["benchmark"] = benchmark
        df["motif"] = motif_of(benchmark)
        df["ex_version"] = ex_version
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize_by_motif(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["ex_version", "motif", "rows", "benchmarks", "kernel_speedup_avg", "total_speedup_avg"])
    grouped = (
        df.groupby(["ex_version", "motif"])
        .agg(
            rows=("benchmark", "count"),
            benchmarks=("benchmark", "nunique"),
            kernel_speedup_avg=("kernel_speedup_avg", "mean"),
            total_speedup_avg=("total_speedup_avg", "mean"),
        )
        .reset_index()
        .sort_values(by=["ex_version", "motif"])
    )
    return grouped


def summarize_by_dataset(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=["ex_version", "motif", "dataset", "rows", "kernel_speedup_avg", "total_speedup_avg"]
        )
    grouped = (
        df.groupby(["ex_version", "motif", "dataset"])
        .agg(
            rows=("benchmark", "count"),
            kernel_speedup_avg=("kernel_speedup_avg", "mean"),
            total_speedup_avg=("total_speedup_avg", "mean"),
        )
        .reset_index()
    )
    grouped["dataset_rank"] = grouped["dataset"].apply(dataset_rank)
    grouped = grouped.sort_values(by=["ex_version", "motif", "dataset_rank"])
    return grouped.drop(columns=["dataset_rank"])


def print_overall_table(summary: pd.DataFrame) -> None:
    if summary.empty:
        print("[WARN] No motif summary data available.")
        return
    print("\n=== Motif average speedup (overall) ===")
    header = f"{'EX':<6} {'motif':<32} {'rows':>6} {'benchmarks':>11} {'k_avg':>10} {'t_avg':>10}"
    print(header)
    for _, row in summary.iterrows():
        print(
            f"{row['ex_version']:<6} "
            f"{row['motif']:<32} "
            f"{int(row['rows']):>6d} "
            f"{int(row['benchmarks']):>11d} "
            f"{row['kernel_speedup_avg']:>10.3f} "
            f"{row['total_speedup_avg']:>10.3f}"
        )


def print_dataset_table(summary: pd.DataFrame) -> None:
    if summary.empty:
        print("[WARN] No motif+dataset summary data available.")
        return
    print("\n=== Motif average speedup by dataset ===")
    header = f"{'EX':<6} {'motif':<32} {'dataset':<12} {'rows':>6} {'k_avg':>10} {'t_avg':>10}"
    print(header)
    for _, row in summary.iterrows():
        print(
            f"{row['ex_version']:<6} "
            f"{row['motif']:<32} "
            f"{(row['dataset'] or '-'): <12} "
            f"{int(row['rows']):>6d} "
            f"{row['kernel_speedup_avg']:>10.3f} "
            f"{row['total_speedup_avg']:>10.3f}"
        )


def maybe_write_csv(df: pd.DataFrame, path: Optional[Path], label: str) -> None:
    if path is None or df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[INFO] {label} written to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate speedup stats by motif.")
    parser.add_argument(
        "--ex-versions",
        help="Comma-separated EX directories (e.g., EX1,EX2,EX3). If omitted, all EX* sub-directories are used.",
    )
    parser.add_argument(
        "--aggregate-root",
        default="analysis_summaries/speedup/aggregate",
        help="Directory containing aggregate CSVs (expects <root>/<EX>/<benchmark>_EX_speedup_stats.csv).",
    )
    parser.add_argument("--output-dir", help="Optional directory to store CSV summaries.")
    args = parser.parse_args()

    aggregate_root = Path(args.aggregate_root).resolve()
    if not aggregate_root.exists():
        print(f"[WARN] Aggregate root does not exist: {aggregate_root}")
        return
    ex_versions = parse_list(args.ex_versions)
    if not ex_versions:
        ex_versions = [
            p.name
            for p in sorted(aggregate_root.iterdir())
            if p.is_dir() and p.name.upper().startswith("EX")
        ]
        if not ex_versions:
            print(f"[WARN] No EX directories detected under {aggregate_root}")
            return
        print(f"[INFO] Auto-detected EX versions: {', '.join(ex_versions)}")
    frames = [load_aggregate_frames(aggregate_root, ex) for ex in ex_versions]
    frames = [df for df in frames if not df.empty]
    if not frames:
        print("[WARN] No aggregate CSVs found for the requested EX versions.")
        return
    union_df = pd.concat(frames, ignore_index=True)

    overall = summarize_by_motif(union_df)
    per_dataset = summarize_by_dataset(union_df)
    print_overall_table(overall)
    print_dataset_table(per_dataset)

    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    if output_dir:
        overall_path = output_dir / "motif_overview.csv"
        dataset_path = output_dir / "motif_by_dataset.csv"
        maybe_write_csv(overall, overall_path, "Overall motif summary")
        maybe_write_csv(per_dataset, dataset_path, "Motif+dataset summary")


if __name__ == "__main__":
    main()

