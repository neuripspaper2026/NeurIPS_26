#!/usr/bin/env python3
"""
HPC-Bench Error Statistics Table Generator

统计各个 benchmark 的错误类型：
1. Compilation Failure (CF): 编译失败
2. Runtime Failure (RF): 运行时失败（编译成功但运行崩溃）
3. Correctness Error (CE): 正确性错误（运行成功但输出不正确）
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path


# ==================== 配置参数 ====================

WORKSPACE_ROOT = Path(__file__).parent.parent.parent
TABLE_DIR = WORKSPACE_ROOT / "analysis_summaries" / "tables"
OUTPUT_DIR = WORKSPACE_ROOT / "analysis_summaries" / "error_tables"

# 实验配置
EX_VERSIONS = ['EX1', 'EX2']
DATASET_SIZES = ['mini', 'small', 'medium', 'large', 'extra-large']
MODELS = ['claude', 'gpt5.1', 'qwen']
N_TRIALS = 10  # 每个配置的试验次数


# ==================== 核心统计函数 ====================

def analyze_errors_per_benchmark(table_a_df, ex_version, dataset_size, threads_filter=None):
    """
    统计每个 benchmark 的错误类型
    
    Args:
        table_a_df: Table A 数据（trial-level）
        ex_version: EX 版本
        dataset_size: 数据集大小
        threads_filter: EX2 的线程数筛选
        
    Returns:
        DataFrame with columns: benchmark, model, total_trials, CF, RF, CE, correct
    """
    
    # 筛选数据
    df = table_a_df[(table_a_df['ex'] == ex_version) & 
                    (table_a_df['dataset_size'] == dataset_size)].copy()
    
    # EX2 线程数筛选
    if ex_version == 'EX2' and threads_filter is not None:
        df = df[df['threads'] == float(threads_filter)]
    
    if len(df) == 0:
        return pd.DataFrame()
    
    results = []
    
    # 对每个 (benchmark, model) 组合统计
    for benchmark in sorted(df['benchmark'].unique()):
        df_bench = df[df['benchmark'] == benchmark]
        
        for model in MODELS:
            df_model = df_bench[df_bench['model'] == model]
            
            if len(df_model) == 0:
                continue
            
            total_trials = len(df_model)
            
            # 统计各类错误
            # 1. Compilation Failure (CF): compile_status == 'NC'
            cf_count = len(df_model[df_model['compile_status'] == 'NC'])
            
            # 2. Runtime Failure (RF): compile_status == 'success' 但 correctness 是 NaN
            #    (编译成功但运行失败，没有 correctness 结果)
            rf_mask = (df_model['compile_status'] == 'success') & (df_model['correctness'].isna())
            rf_count = len(df_model[rf_mask])
            
            # 3. Correctness Error (CE): correctness == 0
            #    (运行成功但输出不正确)
            ce_count = len(df_model[df_model['correctness'] == 0])
            
            # 4. Correct: correctness == 1
            correct_count = len(df_model[df_model['correctness'] == 1])
            
            results.append({
                'benchmark': benchmark,
                'model': model,
                'total_trials': total_trials,
                'CF': cf_count,
                'RF': rf_count,
                'CE': ce_count,
                'correct': correct_count
            })
    
    return pd.DataFrame(results)


def generate_summary_table(error_df):
    """
    生成汇总表：按 benchmark 汇总所有模型的错误
    
    Args:
        error_df: 包含 benchmark, model, CF, RF, CE, correct 的 DataFrame
        
    Returns:
        汇总后的 DataFrame
    """
    
    if len(error_df) == 0:
        return pd.DataFrame()
    
    # 按 benchmark 聚合
    summary = error_df.groupby('benchmark').agg({
        'total_trials': 'sum',
        'CF': 'sum',
        'RF': 'sum',
        'CE': 'sum',
        'correct': 'sum'
    }).reset_index()
    
    # 计算百分比
    summary['CF_pct'] = (summary['CF'] / summary['total_trials'] * 100).round(2)
    summary['RF_pct'] = (summary['RF'] / summary['total_trials'] * 100).round(2)
    summary['CE_pct'] = (summary['CE'] / summary['total_trials'] * 100).round(2)
    summary['correct_pct'] = (summary['correct'] / summary['total_trials'] * 100).round(2)
    
    return summary


def generate_model_breakdown_table(error_df):
    """
    生成模型分解表：每个 benchmark 在不同模型下的错误
    
    Args:
        error_df: 包含 benchmark, model, CF, RF, CE, correct 的 DataFrame
        
    Returns:
        宽格式的 DataFrame（每个模型一列）
    """
    
    if len(error_df) == 0:
        return pd.DataFrame()
    
    # 为每个模型创建一个子表
    model_dfs = []
    
    for model in MODELS:
        df_model = error_df[error_df['model'] == model][['benchmark', 'CF', 'RF', 'CE', 'correct']].copy()
        
        # 重命名列
        df_model.columns = ['benchmark', f'{model}_CF', f'{model}_RF', f'{model}_CE', f'{model}_correct']
        
        model_dfs.append(df_model)
    
    # 合并所有模型
    result = model_dfs[0]
    for df in model_dfs[1:]:
        result = result.merge(df, on='benchmark', how='outer')
    
    return result


# ==================== 主函数 ====================

def main():
    """主函数：生成错误统计表"""
    
    print("="*80)
    print("HPC-Bench Error Statistics Table Generator")
    print("="*80)
    print()
    
    # 确保输出目录存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {OUTPUT_DIR}")
    print()
    
    # 遍历每个 EX 和 dataset_size
    for ex in EX_VERSIONS:
        print(f"{'='*80}")
        print(f"处理 {ex}")
        print(f"{'='*80}")
        print()
        
        for ds in DATASET_SIZES:
            print(f"  处理 {ex} - {ds}...")
            
            # 读取 Table A
            table_a_file = TABLE_DIR / f"table_a_trial_level_{ex}_{ds}.csv"
            
            if not table_a_file.exists():
                print(f"    ⚠️  未找到: {table_a_file.name}")
                continue
            
            df = pd.read_csv(table_a_file)
            print(f"    ✓ 读取数据: {len(df)} 行")
            
            # 分析错误
            threads_filter = 16 if ex == 'EX2' else None
            error_df = analyze_errors_per_benchmark(df, ex, ds, threads_filter)
            
            if len(error_df) == 0:
                print(f"    ⚠️  无数据")
                continue
            
            # ===== 1. 详细表（每个 benchmark × model） =====
            detail_file = OUTPUT_DIR / f"error_detail_{ex}_{ds}.csv"
            error_df.to_csv(detail_file, index=False)
            print(f"    ✓ 详细表: {detail_file.name}")
            
            # ===== 2. 汇总表（按 benchmark 汇总） =====
            summary_df = generate_summary_table(error_df)
            summary_file = OUTPUT_DIR / f"error_summary_{ex}_{ds}.csv"
            summary_df.to_csv(summary_file, index=False)
            print(f"    ✓ 汇总表: {summary_file.name}")
            
            # ===== 3. 模型分解表（宽格式） =====
            model_breakdown_df = generate_model_breakdown_table(error_df)
            breakdown_file = OUTPUT_DIR / f"error_model_breakdown_{ex}_{ds}.csv"
            model_breakdown_df.to_csv(breakdown_file, index=False)
            print(f"    ✓ 模型分解表: {breakdown_file.name}")
            
            print()
        
        print()
    
    # ===== 生成跨 EX 的总表 =====
    print(f"{'='*80}")
    print("生成跨 EX 总表")
    print(f"{'='*80}")
    print()
    
    for ds in DATASET_SIZES:
        print(f"  处理 dataset: {ds}...")
        
        all_ex_dfs = []
        
        for ex in EX_VERSIONS:
            detail_file = OUTPUT_DIR / f"error_detail_{ex}_{ds}.csv"
            
            if detail_file.exists():
                df = pd.read_csv(detail_file)
                df['ex'] = ex
                all_ex_dfs.append(df)
        
        if len(all_ex_dfs) == 0:
            print(f"    ⚠️  无数据")
            continue
        
        # 合并 EX1 和 EX2
        combined_df = pd.concat(all_ex_dfs, ignore_index=True)
        
        # 保存合并表
        combined_file = OUTPUT_DIR / f"error_detail_all_ex_{ds}.csv"
        combined_df.to_csv(combined_file, index=False)
        print(f"    ✓ 总表: {combined_file.name}")
        
        # 生成汇总（按 EX 和 benchmark）
        summary = combined_df.groupby(['ex', 'benchmark']).agg({
            'total_trials': 'sum',
            'CF': 'sum',
            'RF': 'sum',
            'CE': 'sum',
            'correct': 'sum'
        }).reset_index()
        
        # 计算百分比
        summary['CF_pct'] = (summary['CF'] / summary['total_trials'] * 100).round(2)
        summary['RF_pct'] = (summary['RF'] / summary['total_trials'] * 100).round(2)
        summary['CE_pct'] = (summary['CE'] / summary['total_trials'] * 100).round(2)
        summary['correct_pct'] = (summary['correct'] / summary['total_trials'] * 100).round(2)
        
        summary_file = OUTPUT_DIR / f"error_summary_all_ex_{ds}.csv"
        summary.to_csv(summary_file, index=False)
        print(f"    ✓ 汇总表: {summary_file.name}")
        
        print()
    
    print("="*80)
    print("✅ 错误统计表生成完成！")
    print("="*80)
    print()
    print("生成的文件类型：")
    print("  1. error_detail_*.csv       - 详细表（benchmark × model）")
    print("  2. error_summary_*.csv      - 汇总表（按 benchmark）")
    print("  3. error_model_breakdown_*  - 模型分解表（宽格式）")
    print("  4. error_detail_all_ex_*    - 跨 EX 总表")
    print("  5. error_summary_all_ex_*   - 跨 EX 汇总表")
    print()
    print("错误类型说明：")
    print("  - CF (Compilation Failure):  编译失败")
    print("  - RF (Runtime Failure):      运行时失败（编译成功但运行崩溃）")
    print("  - CE (Correctness Error):    正确性错误（运行成功但输出不正确）")
    print("  - correct:                   完全正确")
    print()


if __name__ == "__main__":
    main()
