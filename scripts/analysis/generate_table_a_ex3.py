#!/usr/bin/env python3
"""
将 unified_df_EX3.csv 转换为 table_a_trial_level_EX3_*.csv 格式

用于 k-sensitivity 分析
"""

import pandas as pd
from pathlib import Path

# 配置
WORKSPACE_ROOT = Path(__file__).parent.parent.parent
UNIFIED_DATA_DIR = WORKSPACE_ROOT / "analysis_summaries" / "unified_data"
OUTPUT_DIR = WORKSPACE_ROOT / "analysis_summaries" / "tables"

DATASET_SIZES = ['mini', 'small', 'medium', 'large', 'extra-large']

def convert_unified_to_table_a():
    """
    将 unified_df_EX3.csv 转换为 table_a_trial_level 格式
    """
    print("="*80)
    print("生成 EX3 的 table_a_trial_level 文件")
    print("="*80)
    
    # 读取 unified_df_EX3.csv
    unified_file = UNIFIED_DATA_DIR / "unified_df_EX3.csv"
    
    if not unified_file.exists():
        print(f"❌ 未找到文件: {unified_file}")
        return
    
    print(f"读取: {unified_file}")
    df = pd.read_csv(unified_file)
    print(f"总行数: {len(df)}")
    print()
    
    # 只保留 LLM 模型的数据（排除 baseline）
    df = df[df['is_baseline'] == False].copy()
    print(f"排除 baseline 后: {len(df)} 行")
    
    # 列名映射和转换
    df = df.rename(columns={
        'dataset': 'dataset_size',
        'trail': 'trial_id',
        'is_correct': 'correctness',
        'executable': 'executable_path'
    })
    
    # 添加缺失的列
    df['ex'] = 'EX3'
    df['compile_status'] = 'success'  # unified_df 中的数据都是成功编译的
    df['correctness_tested'] = True
    df['suite'] = 'rodinia'  # EX3 主要是 Rodinia benchmarks
    df['difficulty'] = 'unknown'
    df['motif'] = 'unknown'
    df['checker'] = 'output_check'
    df['correctness_message'] = ''
    df['baseline_kernel_mean_s'] = None  # 需要从 baseline 数据获取
    df['baseline_total_mean_s'] = None
    
    # 转换 correctness 为 1.0/0.0
    df['correctness'] = df['correctness'].astype(float)
    
    # 为每个数据集大小生成单独的文件
    for ds in DATASET_SIZES:
        df_ds = df[df['dataset_size'] == ds].copy()
        
        if len(df_ds) == 0:
            print(f"⚠️  {ds}: 没有数据")
            continue
        
        # 按 benchmark 获取 baseline 数据
        unified_full = pd.read_csv(unified_file)
        baseline_data = unified_full[
            (unified_full['is_baseline'] == True) & 
            (unified_full['dataset'] == ds)
        ][['benchmark', 'kernel_mean_s', 'total_mean_s']].rename(columns={
            'kernel_mean_s': 'baseline_kernel_mean_s',
            'total_mean_s': 'baseline_total_mean_s'
        })
        
        # 合并 baseline 数据
        df_ds = df_ds.merge(baseline_data, on='benchmark', how='left', suffixes=('', '_baseline'))
        
        # 如果合并后有重复列，使用 baseline 的值
        if 'baseline_kernel_mean_s_baseline' in df_ds.columns:
            df_ds['baseline_kernel_mean_s'] = df_ds['baseline_kernel_mean_s_baseline']
            df_ds['baseline_total_mean_s'] = df_ds['baseline_total_mean_s_baseline']
            df_ds = df_ds.drop(columns=['baseline_kernel_mean_s_baseline', 'baseline_total_mean_s_baseline'])
        
        # 选择需要的列（与 EX1/EX2 的 table_a 格式一致）
        columns = [
            'benchmark', 'ex', 'model', 'trial_id', 'dataset_size', 'compiler', 
            'threads', 'suite', 'difficulty', 'motif', 'compile_status', 
            'correctness', 'correctness_tested', 'baseline_kernel_mean_s', 
            'baseline_total_mean_s', 'kernel_mean_s', 'total_mean_s', 
            'kernel_speedup', 'total_speedup', 'warmup_runs', 'measure_runs', 
            'hint_key', 'success_rate', 'checker', 'correctness_message', 
            'executable_path'
        ]
        
        # 确保所有列都存在
        for col in columns:
            if col not in df_ds.columns:
                df_ds[col] = None
        
        df_ds = df_ds[columns]
        
        # 保存
        output_file = OUTPUT_DIR / f"table_a_trial_level_EX3_{ds}.csv"
        df_ds.to_csv(output_file, index=False)
        
        print(f"✓ {ds}: {len(df_ds)} 行 -> {output_file.name}")
    
    print()
    print("="*80)
    print("✅ 完成！")
    print("="*80)


if __name__ == "__main__":
    convert_unified_to_table_a()
