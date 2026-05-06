#!/usr/bin/env python3
"""
Build unified DataFrame for HPC-Bench evaluation data.

This script aggregates correctness and speedup results from all EX versions
into a single tidy/long-form DataFrame for downstream analysis.

Usage:
    python scripts/analysis/build_unified_dataframe.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import numpy as np

# Add project root to path to import internal modules
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.difficulty import BENCHMARK_TO_DIFFICULTY
from scripts.analysis.motif import BENCHMARK_TO_MOTIF

# Constants
DATASET_SIZES = ["mini", "small", "medium", "large", "extra-large"]
EXPECTED_TRIALS = list(range(1, 11))  # 1-10


def extract_model_and_version_from_filename(filename: str) -> Tuple[Optional[str], Optional[int]]:
    """
    从文件名中提取 model 和 version 信息。
    
    示例：
        "output_claude_v10_mini_output.txt" -> ("claude", 10)
        "output_gpt5.1_v3_small_output.txt" -> ("gpt5.1", 3)
        "output_qwen_v7_large_output.data" -> ("qwen", 7)
    
    Args:
        filename: 文件名字符串
    
    Returns:
        (model, version) 元组，如果无法提取则返回 (None, None)
    """
    if not filename or pd.isna(filename):
        return (None, None)
    
    # 尝试匹配模式：output_{model}_v{version}_{dataset}_output.*
    # 支持的 model: claude, gpt4, gpt5.1, qwen 等
    pattern = r'output_([a-zA-Z0-9.]+)_v(\d+)_'
    match = re.search(pattern, str(filename))
    
    if match:
        model = match.group(1)
        version = int(match.group(2))
        return (model, version)
    
    return (None, None)


def normalize_version(version_value) -> Optional[int]:
    """
    将 version 值标准化为整数。
    
    示例：
        "v10" -> 10
        "v1" -> 1
        10 -> 10
        "10" -> 10
        None -> None
    
    Args:
        version_value: 版本值（可能是字符串或整数）
    
    Returns:
        标准化后的整数版本号，如果无法转换则返回 None
    """
    if version_value is None or pd.isna(version_value):
        return None
    
    # 如果是整数，直接返回
    if isinstance(version_value, int):
        return version_value
    
    # 如果是字符串
    if isinstance(version_value, str):
        # 去掉开头的 'v' 或 'V'
        cleaned = version_value.strip().lstrip('vV')
        # 尝试转换为整数
        try:
            return int(cleaned)
        except ValueError:
            return None
    
    # 其他类型，尝试直接转换
    try:
        return int(version_value)
    except (ValueError, TypeError):
        return None


def load_correctness_data(ex_version: str, root_dir: Path) -> pd.DataFrame:
    """
    Load all correctness detail CSVs for a given EX version.
    
    Args:
        ex_version: EX version (e.g., "EX1", "EX2")
        root_dir: Project root directory
        
    Returns:
        DataFrame with columns: benchmark, ex, model, trial_id, dataset_size,
                                correctness, correctness_tested, checker, correctness_message
    """
    correctness_dir = root_dir / "analysis_summaries" / "correctness" / "detail" / ex_version
    
    if not correctness_dir.exists():
        print(f"[WARN] Correctness directory not found: {correctness_dir}")
        return pd.DataFrame()
    
    all_data = []
    csv_files = list(correctness_dir.glob("*_correctness_detail.csv"))
    
    print(f"Loading correctness data for {ex_version}: {len(csv_files)} benchmarks")
    
    for csv_file in csv_files:
        # Extract benchmark name from filename: "2mm_EX1_correctness_detail.csv" -> "2mm"
        benchmark = csv_file.stem.replace(f"_{ex_version}_correctness_detail", "")
        
        try:
            df = pd.read_csv(csv_file)
            
            # Only keep check stage rows
            if 'stage' in df.columns:
                df = df[df['stage'] == 'check'].copy()
            
            # Rename columns to match our schema
            df['benchmark'] = benchmark
            df['ex'] = ex_version
            
            # Fix model and version by extracting from file_name if needed
            if 'file_name' in df.columns:
                for idx, row in df.iterrows():
                    model = row.get('model')
                    version = row.get('version')
                    file_name = row.get('file_name', '')
                    
                    # If model or version is missing/null, extract from file_name
                    if (pd.isna(model) or model == '' or pd.isna(version) or version == '') and file_name:
                        extracted_model, extracted_version = extract_model_and_version_from_filename(file_name)
                        if pd.isna(model) or model == '':
                            df.at[idx, 'model'] = extracted_model
                        if pd.isna(version) or version == '':
                            df.at[idx, 'version'] = extracted_version
                    
                    # Normalize version (remove 'v' prefix if present)
                    current_version = df.at[idx, 'version']
                    normalized = normalize_version(current_version)
                    if normalized is not None:
                        df.at[idx, 'version'] = normalized
            
            # Extract trial_id from version column
            if 'version' in df.columns:
                df['trial_id'] = pd.to_numeric(df['version'], errors='coerce').fillna(0).astype(int)
            else:
                df['trial_id'] = 0  # Baseline or unknown
            
            # Rename dataset to dataset_size
            if 'dataset' in df.columns:
                df.rename(columns={'dataset': 'dataset_size'}, inplace=True)
            
            # Map status to correctness (1=pass, 0=fail/skip/error)
            if 'status' in df.columns:
                df['correctness'] = (df['status'].str.lower() == 'pass').astype(int)
            else:
                df['correctness'] = 0
            
            # Mark as explicitly tested
            df['correctness_tested'] = True
            
            # Keep relevant columns
            columns_to_keep = ['benchmark', 'ex', 'model', 'trial_id', 'dataset_size',
                              'correctness', 'correctness_tested', 'checker', 'message']
            existing_columns = [col for col in columns_to_keep if col in df.columns]
            df = df[existing_columns].copy()
            
            if 'message' in df.columns:
                df.rename(columns={'message': 'correctness_message'}, inplace=True)
            
            all_data.append(df)
            
        except Exception as e:
            print(f"[ERROR] Failed to load {csv_file}: {e}")
            continue
    
    if not all_data:
        return pd.DataFrame()
    
    result = pd.concat(all_data, ignore_index=True)
    
    # Filter out baseline rows (trial_id == 0 or model is empty/None)
    result = result[(result['trial_id'] > 0) & (result['model'].notna()) & (result['model'] != '')].copy()
    
    print(f"  Loaded {len(result)} correctness records for {ex_version}")
    
    return result


def load_speedup_data(ex_version: str, root_dir: Path) -> pd.DataFrame:
    """
    Load all speedup detail CSVs for a given EX version.
    
    Args:
        ex_version: EX version (e.g., "EX1", "EX2")
        root_dir: Project root directory
        
    Returns:
        DataFrame with columns: benchmark, ex, model, trial_id, dataset_size, threads,
                                compiler, kernel_mean_s, total_mean_s, 
                                baseline_kernel_mean_s, baseline_total_mean_s,
                                warmup_runs, measure_runs, hint_key, success_rate, executable_path
    """
    speedup_dir = root_dir / "analysis_summaries" / "speedup" / "detail" / ex_version
    
    if not speedup_dir.exists():
        print(f"[WARN] Speedup directory not found: {speedup_dir}")
        return pd.DataFrame()
    
    all_data = []
    csv_files = list(speedup_dir.glob("*_detail.csv"))
    
    print(f"Loading speedup data for {ex_version}: {len(csv_files)} benchmarks")
    
    for csv_file in csv_files:
        # Extract benchmark name from filename: "2mm_EX1_detail.csv" -> "2mm"
        benchmark = csv_file.stem.replace(f"_{ex_version}_detail", "")
        
        try:
            df = pd.read_csv(csv_file)
            
            df['benchmark'] = benchmark
            df['ex'] = ex_version
            
            # Extract baseline rows separately
            if 'is_baseline' in df.columns:
                baseline_df = df[df['is_baseline'] == True].copy()
                optimized_df = df[df['is_baseline'] == False].copy()
            else:
                baseline_df = df[df['model'] == 'baseline'].copy()
                optimized_df = df[df['model'] != 'baseline'].copy()
            
            if baseline_df.empty:
                print(f"  [WARN] No baseline found for {benchmark} in {ex_version}")
                continue
            
            # Prepare baseline lookup: key = (dataset_size, threads)
            # For EX1 (no threads), we'll use NaN as the threads value
            baseline_lookup = {}
            for _, row in baseline_df.iterrows():
                dataset = row.get('dataset', row.get('dataset_size', 'mini'))
                threads = row.get('threads', np.nan)
                # Normalize NaN threads
                if pd.isna(threads):
                    threads = None
                key = (dataset, threads)
                baseline_lookup[key] = {
                    'baseline_kernel_mean_s': row.get('kernel_mean_s', np.nan),
                    'baseline_total_mean_s': row.get('total_mean_s', np.nan),
                }
            
            # Add baseline values to optimized rows
            for idx, row in optimized_df.iterrows():
                dataset = row.get('dataset', row.get('dataset_size', 'mini'))
                threads = row.get('threads', np.nan)
                # Normalize NaN threads
                if pd.isna(threads):
                    threads = None
                
                # Try to find baseline with matching logic:
                # 1. Exact match (dataset, threads)
                # 2. For EX2+: match (dataset, 1) - baseline is always single-threaded
                # 3. Fallback to (dataset, None) - for EX1 without threads
                key = (dataset, threads)
                if key not in baseline_lookup:
                    # Try with threads=1 (for EX2 where baseline is single-threaded)
                    key = (dataset, 1)
                    if key not in baseline_lookup:
                        # Try with None threads (for EX1)
                        key = (dataset, None)
                
                if key in baseline_lookup:
                    optimized_df.at[idx, 'baseline_kernel_mean_s'] = baseline_lookup[key]['baseline_kernel_mean_s']
                    optimized_df.at[idx, 'baseline_total_mean_s'] = baseline_lookup[key]['baseline_total_mean_s']
                else:
                    # Only warn if we really can't find any baseline for this dataset
                    if not any(k[0] == dataset for k in baseline_lookup.keys()):
                        print(f"  [WARN] No baseline match for {benchmark} {dataset} threads={threads}")
                    optimized_df.at[idx, 'baseline_kernel_mean_s'] = np.nan
                    optimized_df.at[idx, 'baseline_total_mean_s'] = np.nan
            
            # Rename columns to match our schema
            if 'dataset' in optimized_df.columns:
                optimized_df.rename(columns={'dataset': 'dataset_size'}, inplace=True)
            
            # Handle trial_id: speedup CSV has 'version' column that maps to trial_id
            if 'version' in optimized_df.columns:
                optimized_df['trial_id'] = pd.to_numeric(optimized_df['version'], errors='coerce').fillna(0).astype(int)
            elif 'trail' in optimized_df.columns:
                # Old fallback - but 'trail' is usually always 1, not the actual trial ID
                optimized_df['trial_id'] = pd.to_numeric(optimized_df['trail'], errors='coerce').fillna(0).astype(int)
            
            # Ensure trial_id exists and is > 0
            if 'trial_id' not in optimized_df.columns:
                print(f"  [WARN] No trial_id/version column found in {benchmark}, setting to 1")
                optimized_df['trial_id'] = 1
            
            # Keep relevant columns
            columns_to_keep = [
                'benchmark', 'ex', 'model', 'trial_id', 'dataset_size', 'threads', 'compiler',
                'kernel_mean_s', 'total_mean_s', 'baseline_kernel_mean_s', 'baseline_total_mean_s',
                'warmup_runs', 'measure_runs', 'hint_key', 'success_rate', 'executable'
            ]
            existing_columns = [col for col in columns_to_keep if col in optimized_df.columns]
            optimized_df = optimized_df[existing_columns].copy()
            
            if 'executable' in optimized_df.columns:
                optimized_df.rename(columns={'executable': 'executable_path'}, inplace=True)
            
            all_data.append(optimized_df)
            
        except Exception as e:
            print(f"[ERROR] Failed to load {csv_file}: {e}")
            continue
    
    if not all_data:
        return pd.DataFrame()
    
    result = pd.concat(all_data, ignore_index=True)
    
    # Filter out baseline rows again (just in case)
    result = result[(result['trial_id'] > 0) & (result['model'].notna()) & (result['model'] != '')].copy()
    
    print(f"  Loaded {len(result)} speedup records for {ex_version}")
    
    return result


def broadcast_correctness(correctness_df: pd.DataFrame) -> pd.DataFrame:
    """
    Broadcast correctness results across all dataset sizes.
    
    If a trial has correctness tested for any dataset size, replicate that
    result to all 5 dataset sizes. Mark only explicitly tested sizes with
    correctness_tested=True.
    
    Args:
        correctness_df: Correctness DataFrame
        
    Returns:
        Expanded DataFrame with broadcasted correctness
    """
    if correctness_df.empty:
        return correctness_df
    
    print("Broadcasting correctness across dataset sizes...")
    
    # Group by (benchmark, ex, model, trial_id)
    grouped = correctness_df.groupby(['benchmark', 'ex', 'model', 'trial_id'])
    
    all_broadcasted = []
    
    for (benchmark, ex, model, trial_id), group in grouped:
        # Get the correctness value (should be consistent within group)
        # Use the first non-NaN correctness value, prefer 'pass' if multiple
        correctness_val = group['correctness'].max()  # 1 if any pass, else 0
        
        # Get explicitly tested dataset sizes
        tested_sizes = set(group['dataset_size'].unique())
        
        # Get other metadata from first row
        first_row = group.iloc[0]
        checker = first_row.get('checker', None)
        correctness_message = first_row.get('correctness_message', None)
        
        # Create rows for all 5 dataset sizes
        # If any dataset_size was tested for this trial, mark all as tested (broadcast)
        for dataset_size in DATASET_SIZES:
            row = {
                'benchmark': benchmark,
                'ex': ex,
                'model': model,
                'trial_id': trial_id,
                'dataset_size': dataset_size,
                'correctness': correctness_val,
                'correctness_tested': True,  # 广播：只要有任何 dataset_size 被测试，所有都标记为 True
                'checker': checker,
                'correctness_message': correctness_message if dataset_size in tested_sizes else None,
            }
            all_broadcasted.append(row)
    
    result = pd.DataFrame(all_broadcasted)
    
    print(f"  Broadcasted to {len(result)} records (from {len(correctness_df)})")
    
    return result


def generate_full_cartesian_product(
    benchmarks: List[str],
    ex_version: str,
    models: List[str] = ["claude", "gpt5.1", "qwen"],
    trial_ids: List[int] = list(range(1, 11)),
    dataset_sizes: List[str] = ["mini", "small", "medium", "large", "extra-large"],
) -> pd.DataFrame:
    """
    Generate full Cartesian product of all possible combinations.
    
    This creates the "expected" universe of all trials, which we'll compare
    against actual data to identify compilation failures (NC).
    
    Args:
        benchmarks: List of benchmark names
        ex_version: EX version
        models: List of model names
        trial_ids: List of trial IDs (1-10)
        dataset_sizes: List of dataset sizes
        
    Returns:
        DataFrame with all possible combinations
    """
    import itertools
    
    # Generate all combinations
    combinations = list(itertools.product(
        benchmarks,
        [ex_version],
        models,
        trial_ids,
        dataset_sizes,
    ))
    
    df = pd.DataFrame(combinations, columns=[
        'benchmark', 'ex', 'model', 'trial_id', 'dataset_size'
    ])
    
    # Add threads for EX2
    if ex_version == "EX2":
        # Each combination should exist for multiple thread counts
        thread_counts = [1, 2, 8, 16, 32]
        expanded = []
        for threads in thread_counts:
            df_copy = df.copy()
            df_copy['threads'] = threads
            expanded.append(df_copy)
        df = pd.concat(expanded, ignore_index=True)
    else:
        df['threads'] = np.nan
    
    return df


def merge_data(
    correctness_df: pd.DataFrame,
    speedup_df: pd.DataFrame,
    ex_version: str,
    all_benchmarks: List[str],
) -> pd.DataFrame:
    """
    Merge correctness and speedup data, generating full Cartesian product.
    
    Args:
        correctness_df: Correctness DataFrame (already broadcasted)
        speedup_df: Speedup DataFrame (with baseline values)
        ex_version: EX version for context
        all_benchmarks: List of all benchmarks
        
    Returns:
        Merged DataFrame with compile_status
    """
    print(f"Merging correctness and speedup data for {ex_version}...")
    
    # Get unique models and benchmarks from actual data
    if not speedup_df.empty:
        actual_models = speedup_df['model'].unique().tolist()
        benchmarks_with_speedup = speedup_df['benchmark'].unique().tolist()
    else:
        actual_models = ["claude", "gpt5.1", "qwen"]
        benchmarks_with_speedup = []
    
    if not correctness_df.empty:
        benchmarks_with_correctness = correctness_df['benchmark'].unique().tolist()
    else:
        benchmarks_with_correctness = []
    
    # Union of benchmarks with any data
    benchmarks_to_use = sorted(set(benchmarks_with_speedup) | set(benchmarks_with_correctness))
    
    if not benchmarks_to_use:
        return pd.DataFrame()
    
    print(f"  Generating full Cartesian product for {len(benchmarks_to_use)} benchmarks, {len(actual_models)} models...")
    full_product = generate_full_cartesian_product(
        benchmarks=benchmarks_to_use,
        ex_version=ex_version,
        models=actual_models,
    )
    
    print(f"  Full product: {len(full_product):,} expected combinations")
    
    # Merge with speedup data (left join to keep all expected combinations)
    merge_keys = ['benchmark', 'ex', 'model', 'trial_id', 'dataset_size']
    if ex_version == "EX2":
        merge_keys.append('threads')
    
    merged = pd.merge(
        full_product,
        speedup_df,
        on=merge_keys,
        how='left',
        suffixes=('', '_speed')
    )
    
    # Add compile_status based on speedup data (初步判断)
    # success: 有性能数据
    # NC: 没有性能数据（可能是编译失败或运行失败）
    merged['compile_status'] = merged['kernel_mean_s'].notna().map({
        True: 'success',
        False: 'NC'
    })
    
    print(f"  Initial compile status: {(merged['compile_status'] == 'success').sum():,} success, "
          f"{(merged['compile_status'] == 'NC').sum():,} NC")
    
    # Merge with correctness data
    if not correctness_df.empty:
        correctness_merge = pd.merge(
            merged,
            correctness_df[['benchmark', 'ex', 'model', 'trial_id', 'dataset_size', 
                           'correctness', 'correctness_tested', 'checker', 'correctness_message']],
            on=['benchmark', 'ex', 'model', 'trial_id', 'dataset_size'],
            how='left',
            suffixes=('', '_corr')
        )
        
        # Update correctness columns
        for col in ['correctness', 'correctness_tested', 'checker', 'correctness_message']:
            if col in correctness_merge.columns:
                merged[col] = correctness_merge[col]
        
        # 重要修正：区分三种状态
        # 1. success: 有 speedup 数据
        # 2. runtime_failure: 有 correctness 数据但没有 speedup 数据（编译成功但性能测量失败）
        # 3. NC: 既没有 speedup 也没有 correctness 数据（编译失败）
        has_correctness = merged['correctness_tested'] == True
        has_speedup = merged['kernel_mean_s'].notna()
        
        # 有 correctness 但没有 speedup → runtime_failure
        merged.loc[has_correctness & ~has_speedup, 'compile_status'] = 'runtime_failure'
        
        # 有 correctness 且有 speedup → success (保持不变)
        merged.loc[has_correctness & has_speedup, 'compile_status'] = 'success'
        
        # 既没有 correctness 也没有 speedup → NC (保持不变)
        
        print(f"  After correctness merge: "
              f"{(merged['compile_status'] == 'success').sum():,} success, "
              f"{(merged['compile_status'] == 'runtime_failure').sum():,} runtime_failure, "
              f"{(merged['compile_status'] == 'NC').sum():,} NC")
    
    # Fill missing correctness_tested
    if 'correctness_tested' not in merged.columns:
        merged['correctness_tested'] = False
    else:
        merged['correctness_tested'].fillna(False, inplace=True)
    
    # For NC rows, ensure correctness is NaN and correctness_tested is False
    # 编译失败的情况下不可能进行 correctness 测试
    nc_rows = merged['compile_status'] == 'NC'
    merged.loc[nc_rows, 'correctness'] = np.nan
    merged.loc[nc_rows, 'correctness_tested'] = False
    
    print(f"  Merged to {len(merged):,} records")
    
    return merged


def enrich_metadata(df: pd.DataFrame, root_dir: Path) -> pd.DataFrame:
    """
    Add difficulty, motif, and suite metadata to DataFrame.
    
    Args:
        df: Merged DataFrame
        root_dir: Project root directory
        
    Returns:
        Enriched DataFrame
    """
    print("Enriching with metadata (difficulty, motif, suite)...")
    
    if df.empty:
        return df
    
    # Add difficulty
    df['difficulty'] = df['benchmark'].map(BENCHMARK_TO_DIFFICULTY)
    df['difficulty'].fillna('unknown', inplace=True)
    
    # Add motif
    df['motif'] = df['benchmark'].map(BENCHMARK_TO_MOTIF)
    df['motif'].fillna('unknown', inplace=True)
    
    # Add suite - need to load from output_specs
    try:
        from scripts.benchmark_catalog import (
            BENCHMARK_POLYBENCH,
            BENCHMARK_RODINIA,
            BENCHMARK_CODEE,
            BENCHMARK_PARBOIL,
            BENCHMARK_MIBENCH,
            BENCHMARK_PARSEC,
            BENCHMARK_MACHSUITE,
        )
        
        benchmark_to_suite = {}
        for name in BENCHMARK_POLYBENCH.keys():
            benchmark_to_suite[name] = 'polybench'
        for name in BENCHMARK_RODINIA.keys():
            benchmark_to_suite[name] = 'rodinia'
        for name in BENCHMARK_CODEE.keys():
            benchmark_to_suite[name] = 'codee'
        for name in BENCHMARK_PARBOIL.keys():
            benchmark_to_suite[name] = 'parboil'
        for name in BENCHMARK_MIBENCH.keys():
            benchmark_to_suite[name] = 'mibench'
        for name in BENCHMARK_PARSEC.keys():
            benchmark_to_suite[name] = 'parsec'
        for name in BENCHMARK_MACHSUITE.keys():
            benchmark_to_suite[name] = 'machsuite'
        
        df['suite'] = df['benchmark'].map(benchmark_to_suite)
        df['suite'].fillna('unknown', inplace=True)
    except Exception as e:
        print(f"  [WARN] Could not load suite info: {e}")
        df['suite'] = 'unknown'
    
    print(f"  Enriched {len(df)} records")
    
    return df


def calculate_speedups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate fresh speedup values from baseline and optimized runtimes.
    
    Args:
        df: DataFrame with runtime columns
        
    Returns:
        DataFrame with speedup columns added
    """
    print("Calculating speedups from runtime data...")
    
    if df.empty:
        return df
    
    # Calculate kernel speedup
    df['kernel_speedup'] = df['baseline_kernel_mean_s'] / df['kernel_mean_s']
    
    # Calculate total speedup
    df['total_speedup'] = df['baseline_total_mean_s'] / df['total_mean_s']
    
    # Replace inf with NaN (avoid inplace warning)
    df['kernel_speedup'] = df['kernel_speedup'].replace([np.inf, -np.inf], np.nan)
    df['total_speedup'] = df['total_speedup'].replace([np.inf, -np.inf], np.nan)
    
    print(f"  Calculated speedups for {len(df)} records")
    
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """
    Validate that DataFrame matches expected schema.
    
    Args:
        df: DataFrame to validate
        
    Raises:
        ValueError if validation fails
    """
    print("Validating schema...")
    
    required_columns = [
        'benchmark', 'ex', 'model', 'trial_id', 'dataset_size',
        'correctness', 'difficulty', 'motif'
    ]
    
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Check correctness values
    invalid_correctness = df[~df['correctness'].isin([0, 1])]['correctness'].unique()
    if len(invalid_correctness) > 0:
        print(f"  [WARN] Invalid correctness values found: {invalid_correctness}")
    
    # Check for negative runtimes
    if 'kernel_mean_s' in df.columns:
        negative = df[df['kernel_mean_s'] < 0]
        if len(negative) > 0:
            print(f"  [WARN] {len(negative)} rows with negative kernel_mean_s")
    
    print("  Schema validation passed")


def export_unified_data(df: pd.DataFrame, output_path: Path) -> None:
    """
    Export unified DataFrame to CSV with summary statistics.
    
    Args:
        df: Unified DataFrame
        output_path: Output CSV path
    """
    print(f"Exporting unified DataFrame to {output_path}...")
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Define column order
    column_order = [
        # Core identifiers
        'benchmark', 'ex', 'model', 'trial_id', 'dataset_size', 'compiler', 'threads', 'suite',
        # Metadata
        'difficulty', 'motif',
        # Compilation status
        'compile_status',
        # Correctness
        'correctness', 'correctness_tested',
        # Baseline runtime
        'baseline_kernel_mean_s', 'baseline_total_mean_s',
        # Optimized runtime
        'kernel_mean_s', 'total_mean_s',
        # Speedups
        'kernel_speedup', 'total_speedup',
        # Experimental metadata
        'warmup_runs', 'measure_runs', 'hint_key', 'success_rate',
        'checker', 'correctness_message', 'executable_path',
    ]
    
    # Keep only columns that exist
    existing_columns = [col for col in column_order if col in df.columns]
    df_export = df[existing_columns].copy()
    
    # Export to CSV
    df_export.to_csv(output_path, index=False)
    
    print(f"  Exported {len(df_export)} rows × {len(df_export.columns)} columns")
    print(f"\nSummary Statistics:")
    print(f"  Total rows: {len(df_export):,}")
    print(f"  Benchmarks: {df_export['benchmark'].nunique()}")
    print(f"  EX versions: {df_export['ex'].nunique()}")
    print(f"  Models: {df_export['model'].nunique()}")
    print(f"  Dataset sizes: {df_export['dataset_size'].nunique()}")
    
    if 'compile_status' in df_export.columns:
        print(f"  Compile success: {(df_export['compile_status'] == 'success').sum():,} "
              f"({(df_export['compile_status'] == 'success').sum()/len(df_export)*100:.1f}%)")
        print(f"  Compile failed (NC): {(df_export['compile_status'] == 'NC').sum():,} "
              f"({(df_export['compile_status'] == 'NC').sum()/len(df_export)*100:.1f}%)")
    
    # Pass rate only for successfully compiled trials
    if 'compile_status' in df_export.columns:
        compiled = df_export[df_export['compile_status'] == 'success']
        if len(compiled) > 0 and 'correctness' in compiled.columns:
            pass_rate = compiled['correctness'].mean()
            print(f"  Pass rate (of compiled): {pass_rate:.2%}")
    else:
        pass_rate = df_export['correctness'].mean()
        print(f"  Pass rate: {pass_rate:.2%}")
    
    if 'kernel_speedup' in df_export.columns:
        valid_speedups = df_export['kernel_speedup'].dropna()
        if len(valid_speedups) > 0:
            print(f"  Mean kernel speedup: {valid_speedups.mean():.3f}x")
            print(f"  Median kernel speedup: {valid_speedups.median():.3f}x")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Build unified DataFrame from HPC-Bench evaluation results"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "final_data" / "unified_results.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--ex-versions",
        nargs='+',
        default=["EX1", "EX2"],
        help="EX versions to process"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Building Unified HPC-Bench DataFrame")
    print("=" * 80)
    
    all_merged = []
    
    for ex_version in args.ex_versions:
        print(f"\n{'=' * 80}")
        print(f"Processing {ex_version}")
        print(f"{'=' * 80}\n")
        
        # 1. Load data
        correctness_df = load_correctness_data(ex_version, args.root)
        speedup_df = load_speedup_data(ex_version, args.root)
        
        if correctness_df.empty and speedup_df.empty:
            print(f"[WARN] No data found for {ex_version}, skipping...")
            continue
        
        # 2. Broadcast correctness
        if not correctness_df.empty:
            correctness_df = broadcast_correctness(correctness_df)
        
        # 3. Merge correctness + speedup (no need to pass benchmarks, function gets it from data)
        merged = merge_data(correctness_df, speedup_df, ex_version, [])
        
        if not merged.empty:
            all_merged.append(merged)
    
    if not all_merged:
        print("\n[ERROR] No data to export!")
        return 1
    
    # 4. Combine all EX versions
    print(f"\n{'=' * 80}")
    print("Combining all EX versions")
    print(f"{'=' * 80}\n")
    unified = pd.concat(all_merged, ignore_index=True)
    print(f"Combined to {len(unified)} total records")
    
    # 5. Enrich with metadata
    unified = enrich_metadata(unified, args.root)
    
    # 6. Calculate speedups
    unified = calculate_speedups(unified)
    
    # 7. Validate and export
    validate_schema(unified)
    export_unified_data(unified, args.output)
    
    print(f"\n{'=' * 80}")
    print("✓ Unified DataFrame built successfully!")
    print(f"{'=' * 80}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

