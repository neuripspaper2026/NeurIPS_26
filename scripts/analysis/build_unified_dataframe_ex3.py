#!/usr/bin/env python3
"""
Build unified DataFrame for EX3 evaluation data.

This script aggregates correctness and speedup results from EX3
into a single tidy/long-form DataFrame for downstream analysis.

Usage:
    python scripts/analysis/build_unified_dataframe_ex3.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Constants
DATASET_SIZES = ["mini", "small", "medium", "large", "extra-large"]
EXPECTED_TRIALS = list(range(1, 11))  # 1-10
EX_VERSION = "EX3"


def load_correctness_data(correctness_file: Path) -> pd.DataFrame:
    """加载 EX3 correctness 数据"""
    if not correctness_file.exists():
        print(f"[WARN] Correctness file not found: {correctness_file}")
        return pd.DataFrame(columns=["benchmark", "model", "version", "correctness"])
    
    df = pd.read_csv(correctness_file)
    # 确保 correctness 列是布尔值（处理 "true"/"false" 字符串或布尔值）
    if df["correctness"].dtype == bool:
        df["is_correct"] = df["correctness"]
    else:
        df["is_correct"] = df["correctness"].astype(str).str.lower() == "true"
    return df


def load_speedup_detail_data(detail_dir: Path) -> pd.DataFrame:
    """加载 EX3 详细 speedup 数据"""
    all_data = []
    
    for csv_file in sorted(detail_dir.glob("*_EX3_summary.csv")):
        benchmark = csv_file.stem.replace("_EX3_summary", "")
        
        try:
            df = pd.read_csv(csv_file)
            df["benchmark"] = benchmark
            all_data.append(df)
        except Exception as e:
            print(f"[WARN] Error loading {csv_file}: {e}")
            continue
    
    if not all_data:
        print("[WARN] No speedup detail data found")
        return pd.DataFrame()
    
    return pd.concat(all_data, ignore_index=True)


def parse_speedup_value(speedup_str) -> Optional[float]:
    """解析 speedup 字符串 (如 '1.23x') 为浮点数"""
    if pd.isna(speedup_str) or speedup_str == "-":
        return None
    
    if isinstance(speedup_str, (int, float)):
        return float(speedup_str)
    
    if isinstance(speedup_str, str):
        try:
            return float(speedup_str.rstrip("x"))
        except:
            return None
    
    return None


def build_unified_dataframe(
    correctness_file: Path,
    speedup_detail_dir: Path,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """构建 EX3 统一数据框"""
    
    print(f"Loading data for {EX_VERSION}...")
    
    # 加载 correctness 数据
    df_correctness = load_correctness_data(correctness_file)
    print(f"  Loaded {len(df_correctness)} correctness records")
    
    # 加载 speedup 数据
    df_speedup = load_speedup_detail_data(speedup_detail_dir)
    print(f"  Loaded {len(df_speedup)} speedup records")
    
    if df_speedup.empty:
        print("[ERROR] No speedup data available")
        return pd.DataFrame()
    
    # 标准化 speedup 数据
    df_speedup["dataset"] = df_speedup["dataset"].str.lower()
    df_speedup["trail"] = df_speedup.get("trail", 1).fillna(1).astype(int)
    
    # 版本号转换：保留为 float 以便后续统一处理
    df_speedup["version"] = pd.to_numeric(df_speedup["version"], errors='coerce')
    
    # 解析 speedup 值
    df_speedup["kernel_speedup"] = df_speedup["kernel_speedup"].apply(parse_speedup_value)
    df_speedup["total_speedup"] = df_speedup["total_speedup"].apply(parse_speedup_value)
    
    # 确保两个 dataframe 的 version 类型一致
    # 统一使用 float 类型进行合并
    df_correctness["version"] = df_correctness["version"].astype(float)
    df_speedup["version"] = df_speedup["version"].astype(float)
    
    # 合并 correctness 数据
    df_merged = df_speedup.merge(
        df_correctness[["benchmark", "model", "version", "is_correct"]],
        on=["benchmark", "model", "version"],
        how="left"
    )
    
    # Debug: 打印合并统计
    print(f"\n[DEBUG] Merge statistics:")
    print(f"  Before merge: {len(df_speedup)} speedup records")
    print(f"  Correctness records: {len(df_correctness)}")
    print(f"  After merge: {len(df_merged)} records")
    print(f"  Records with correctness data: {df_merged['is_correct'].notna().sum()}")
    print(f"  Records marked correct: {(df_merged['is_correct'] == True).sum()}")
    
    # 填充缺失的 correctness 值（默认为 False）
    df_merged["is_correct"] = df_merged["is_correct"].fillna(False)
    
    # 添加 EX 版本列
    df_merged["ex_version"] = EX_VERSION
    
    # 重新排列列
    columns = [
        "benchmark",
        "ex_version",
        "model",
        "version",
        "dataset",
        "trail",
        "is_correct",
        "kernel_speedup",
        "total_speedup",
        "success_rate"
    ]
    
    # 只保留存在的列
    columns = [col for col in columns if col in df_merged.columns]
    df_unified = df_merged[columns].copy()
    
    # 排序
    df_unified = df_unified.sort_values(
        ["benchmark", "model", "version", "dataset", "trail"]
    ).reset_index(drop=True)
    
    print(f"\n[INFO] Unified dataframe built: {len(df_unified)} rows")
    print(f"  Benchmarks: {df_unified['benchmark'].nunique()}")
    print(f"  Models: {df_unified['model'].nunique()}")
    print(f"  Versions: {df_unified['version'].nunique()}")
    
    # 保存
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_unified.to_csv(output_path, index=False)
        print(f"\n[SAVED] {output_path}")
    
    return df_unified


def compute_fast_at_k_and_speedup_at_k(
    df: pd.DataFrame,
    k: int = 3,
    output_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    计算 Fast@k 和 Speedup@k 指标
    
    Fast@k: 在前 k 个版本中是否有至少一个版本 speedup > 1.0 且 correct
    Speedup@k: 前 k 个版本中正确版本的最大 speedup
    """
    
    results = []
    
    # 按 benchmark, model, dataset 分组
    for (benchmark, model, dataset), group in df.groupby(["benchmark", "model", "dataset"]):
        # 只保留正确的版本
        correct_group = group[group["is_correct"] == True].copy()
        
        if len(correct_group) == 0:
            # 没有正确的版本
            results.append({
                "benchmark": benchmark,
                "model": model,
                "dataset": dataset,
                "ex_version": EX_VERSION,
                f"fast_at_{k}": 0,
                f"speedup_at_{k}": 1.0,  # baseline
                "num_correct_versions": 0,
                "num_total_versions": len(group["version"].unique())
            })
            continue
        
        # 按 version 排序，取前 k 个
        correct_group = correct_group.sort_values("version")
        versions_in_k = correct_group["version"].unique()[:k]
        
        # 只保留前 k 个版本的数据
        topk_group = correct_group[correct_group["version"].isin(versions_in_k)]
        
        # Fast@k: 是否有至少一个 kernel_speedup > 1.0
        kernel_speedups = topk_group["kernel_speedup"].dropna()
        fast_at_k = 1 if (kernel_speedups > 1.0).any() else 0
        
        # Speedup@k: 最大 kernel_speedup
        max_speedup = kernel_speedups.max() if len(kernel_speedups) > 0 else 1.0
        
        results.append({
            "benchmark": benchmark,
            "model": model,
            "dataset": dataset,
            "ex_version": EX_VERSION,
            f"fast_at_{k}": fast_at_k,
            f"speedup_at_{k}": max_speedup,
            "num_correct_versions": len(correct_group["version"].unique()),
            "num_total_versions": len(group["version"].unique())
        })
    
    df_metrics = pd.DataFrame(results)
    
    if output_dir:
        output_path = output_dir / f"fast_speedup_at_k{k}_EX3.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_metrics.to_csv(output_path, index=False)
        print(f"\n[SAVED] Fast@{k} and Speedup@{k} metrics: {output_path}")
    
    return df_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Build unified DataFrame for EX3 evaluation data"
    )
    parser.add_argument(
        "--correctness-file",
        type=Path,
        default=Path("analysis_summaries/correctness/EX3_correctness_summary.csv"),
        help="Path to EX3 correctness summary CSV"
    )
    parser.add_argument(
        "--speedup-detail-dir",
        type=Path,
        default=Path("analysis_summaries/speedup/detail/EX3"),
        help="Path to EX3 speedup detail directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_summaries/unified_data"),
        help="Output directory for unified dataframe"
    )
    
    args = parser.parse_args()
    
    # 构建统一数据框
    output_path = args.output_dir / "unified_df_EX3.csv"
    df_unified = build_unified_dataframe(
        correctness_file=args.correctness_file,
        speedup_detail_dir=args.speedup_detail_dir,
        output_path=output_path
    )
    
    if df_unified.empty:
        print("[ERROR] Failed to build unified dataframe")
        return
    
    # 计算 Fast@k 和 Speedup@k
    for k in [1, 3, 5]:
        compute_fast_at_k_and_speedup_at_k(
            df_unified,
            k=k,
            output_dir=args.output_dir
        )
    
    print("\n[DONE] EX3 unified dataframe and metrics computed successfully")


if __name__ == "__main__":
    main()
