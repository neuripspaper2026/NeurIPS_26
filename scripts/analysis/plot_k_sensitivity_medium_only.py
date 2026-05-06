#!/usr/bin/env python3
"""
HPC-Bench k-Sensitivity Analysis - Medium Dataset Only

只生成 medium 数据集的 k-sensitivity 图，每个 EX 单独一张图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations
from math import comb


# ==================== 配置参数 ====================

# 目录配置
WORKSPACE_ROOT = Path(__file__).parent.parent.parent
TABLE_DIR = WORKSPACE_ROOT / "analysis_summaries" / "tables"
OUTPUT_DIR = WORKSPACE_ROOT / "analysis_summaries" / "beautiful_figure" / "main_paper" / "sensitivity_analysis"

# 确保输出目录存在
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 实验配置
ALL_EX_VERSIONS = ['EX1', 'EX2', 'EX3']
DATASET_SIZE = 'medium'  # 只处理 medium

# 模型配置
ALL_MODELS = ['claude', 'gpt5.1', 'qwen']
MODEL_LABELS = {
    'claude': 'Claude',
    'gpt5.1': 'GPT-4o',
    'qwen': 'Qwen'
}

MODEL_COLORS = {
    'claude': '#2E86AB',
    'gpt5.1': '#A23B72', 
    'qwen': '#F18F01'
}

# Sensitivity 分析配置
N = 10  # 总试验次数
K_RANGE = range(1, 11)  # k 从 1 到 10
DEFAULT_P_THR = 1.05  # 固定 p_thr


# ==================== 核心计算函数 ====================

def compute_fast_at_k(c_p, N, k):
    """计算 Fast@k（闭式解）"""
    if c_p < 0 or N <= 0 or k <= 0 or k > N:
        return np.nan
    
    if c_p >= k:
        return 1.0
    
    if c_p == 0:
        return 0.0
    
    numerator = comb(N - c_p, k)
    denominator = comb(N, k)
    
    if denominator == 0:
        return np.nan
    
    return 1.0 - (numerator / denominator)


def compute_speedup_at_k_exact(speedups_correct, p_thr, N, k):
    """计算 Speedup@k（精确枚举）"""
    if len(speedups_correct) == 0:
        return 0.0
    
    if k > N or k <= 0:
        return np.nan
    
    # 用 0 填充到 N 个样本
    speedups = list(speedups_correct) + [0.0] * (N - len(speedups_correct))
    
    if k > len(speedups):
        return np.nan
    
    # 精确计算所有组合
    total_speedup = 0.0
    count = 0
    
    for combo in combinations(speedups, k):
        max_speedup = max(combo)
        if max_speedup >= p_thr:
            total_speedup += max_speedup
        count += 1
    
    return total_speedup / count if count > 0 else 0.0


def compute_metrics_by_k(table_a_df, ex_version, p_thr, k_range, threads_filter=None):
    """计算不同 k 值下的 Fast@k 和 Speedup@k"""
    
    print(f"  处理: {ex_version} - {DATASET_SIZE}")
    
    # 筛选数据
    df = table_a_df[(table_a_df['ex'] == ex_version) & 
                    (table_a_df['dataset_size'] == DATASET_SIZE)].copy()
    
    # EX2 线程数筛选
    if ex_version == 'EX2' and threads_filter is not None:
        df = df[df['threads'] == float(threads_filter)]
    
    if len(df) == 0:
        print(f"    没有数据")
        return pd.DataFrame()
    
    results = []
    total_tasks = len(ALL_MODELS) * len(k_range)
    current_task = 0
    
    for model in ALL_MODELS:
        df_model = df[df['model'] == model]
        
        if len(df_model) == 0:
            continue
        
        for k in k_range:
            current_task += 1
            print(f"    [{current_task}/{total_tasks}] {model} - k={k}")
            
            fast_k_list = []
            speedup_k_list = []
            
            for benchmark in df_model['benchmark'].unique():
                df_bench = df_model[df_model['benchmark'] == benchmark]
                
                # 限制到前 N 个版本（某些 benchmark 可能有更多版本）
                actual_n = min(len(df_bench), N)
                df_bench = df_bench.head(N)  # 只取前 10 个版本
                
                # 获取所有 trial 的 speedup（正确的）
                speedups_correct = df_bench[df_bench['correctness'] == 1.0]['kernel_speedup'].values
                c_p = len(speedups_correct)
                
                # 计算 Fast@k（使用实际的 N）
                fast_k = compute_fast_at_k(c_p, actual_n, k)
                if not np.isnan(fast_k):
                    fast_k_list.append(fast_k)
                
                # 计算 Speedup@k（使用实际的 N）
                if c_p > 0:
                    speedup_k = compute_speedup_at_k_exact(speedups_correct, p_thr, actual_n, k)
                    if not np.isnan(speedup_k):
                        speedup_k_list.append(speedup_k)
            
            # Macro-average over benchmarks
            if len(fast_k_list) > 0:
                results.append({
                    'model': model,
                    'k': k,
                    'Fast_at_k': np.mean(fast_k_list),
                    'Speedup_at_k': np.mean(speedup_k_list) if len(speedup_k_list) > 0 else 0.0
                })
    
    print(f"    完成: {len(results)} 个数据点")
    return pd.DataFrame(results)


# ==================== 画图函数 ====================

def plot_single_ex_beautiful(df_results, ex_version, metric_name, ylabel, filename):
    """画单个 EX 的美化图"""
    
    if len(df_results) == 0:
        print(f"⚠️  {ex_version} 没有数据，跳过")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 为每个模型画线
    for model in ALL_MODELS:
        df_model = df_results[df_results['model'] == model].sort_values('k')
        
        if len(df_model) == 0:
            continue
        
        ax.plot(
            df_model['k'],
            df_model[metric_name],
            color=MODEL_COLORS[model],
            linewidth=2.5,
            marker=None,
            alpha=0.85,
            label=MODEL_LABELS[model]
        )
    
    # 网格（只保留水平网格，且更淡）
    ax.grid(True, axis='y', alpha=0.15, linestyle='-', linewidth=0.4, color='gray')
    ax.set_axisbelow(True)
    
    # 设置范围和标签
    if 'Fast' in metric_name:
        ax.set_ylim(-0.05, 1.05)
    else:
        ax.set_ylim(bottom=0)
    
    ax.set_xlim(0.5, 10.5)
    ax.set_xticks(range(1, 11))
    
    ax.set_xlabel(r'$k$', fontsize=14, fontweight='normal')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='normal')
    
    # 标题
    ex_labels = {'EX1': 'EX1 (Serial)', 'EX2': 'EX2 (OpenMP)', 'EX3': 'EX3 (CUDA)'}
    ax.set_title(f'{ex_labels[ex_version]} - Medium Dataset', 
                fontsize=14, fontweight='bold', pad=15)
    
    # 图例（右下角，半透明）
    ax.legend(loc='lower right', frameon=True, fontsize=11, 
             framealpha=0.7, edgecolor='gray')
    
    ax.tick_params(axis='both', labelsize=10)
    
    plt.tight_layout()
    
    # 保存
    output_file = OUTPUT_DIR / filename
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"✓ 保存: {output_file}")
    
    plt.close()


# ==================== 主函数 ====================

def main():
    """主函数"""
    
    print("="*80)
    print("生成 k-Sensitivity 图（仅 Medium 数据集）")
    print("="*80)
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"数据集大小: {DATASET_SIZE}")
    print(f"固定 p_thr: {DEFAULT_P_THR}")
    print(f"k 范围: {list(K_RANGE)}")
    print()
    
    # 读取 Table A 数据（只读 medium）
    print("加载 Table A 数据...")
    dfs = []
    
    for ex in ALL_EX_VERSIONS:
        filepath = TABLE_DIR / f"table_a_trial_level_{ex}_{DATASET_SIZE}.csv"
        
        if filepath.exists():
            df = pd.read_csv(filepath)
            dfs.append(df)
            print(f"  ✓ 找到: {filepath.name} ({len(df)} 行)")
        else:
            print(f"  ⚠️  未找到: {filepath.name}")
    
    if len(dfs) == 0:
        print("❌ 没有找到任何数据！")
        return
    
    table_a_df = pd.concat(dfs, ignore_index=True)
    print(f"总行数: {len(table_a_df)}")
    print()
    
    # 为每个 EX 生成图
    for ex in ALL_EX_VERSIONS:
        print("="*80)
        print(f"生成 {ex} 的图...")
        print("="*80)
        
        # 计算指标
        df_results = compute_metrics_by_k(
            table_a_df, ex, DEFAULT_P_THR, K_RANGE, threads_filter=16
        )
        
        if len(df_results) == 0:
            print(f"⚠️  {ex} 没有数据")
            continue
        
        # Fast@k 图
        plot_single_ex_beautiful(
            df_results, ex,
            metric_name='Fast_at_k',
            ylabel='Fast@k',
            filename=f'fast_at_k_vs_k_{ex}_{DATASET_SIZE}.pdf'
        )
        
        # Speedup@k 图
        plot_single_ex_beautiful(
            df_results, ex,
            metric_name='Speedup_at_k',
            ylabel='Speedup@k',
            filename=f'speedup_at_k_vs_k_{ex}_{DATASET_SIZE}.pdf'
        )
        
        print()
    
    print("="*80)
    print("✅ 完成！")
    print("="*80)


if __name__ == "__main__":
    main()
