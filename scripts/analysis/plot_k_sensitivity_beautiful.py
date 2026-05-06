#!/usr/bin/env python3
"""
HPC-Bench k-Sensitivity Analysis - Beautiful Version

分析固定 p_thr 下，k 值变化（1-10）对 Fast@k 和 Speedup@k 的影响

美化优化：
1. 去掉 markers，只保留线条
2. 网格只保留水平方向，透明度更低
3. 图例移到顶部中央
4. 行标签竖着放在左侧（EX1 (Serial), EX2 (OpenMP), EX3 (CUDA)）
5. 只在外侧显示坐标轴标签
"""

import os
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
DATASET_SIZES = ['mini', 'small', 'medium', 'large', 'extra-large']

# 模型配置（与 beautiful 版本一致）
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

# Dataset 标签
DATASET_LABELS = {
    'mini': 'Mini',
    'small': 'Small',
    'medium': 'Medium',
    'large': 'Large',
    'extra-large': 'Extra-Large'
}

# Sensitivity 分析配置
N = 10  # 总试验次数
K_RANGE = range(1, 11)  # k 从 1 到 10
DEFAULT_P_THR = 1.05  # 固定 p_thr


# ==================== 核心计算函数 ====================

def compute_fast_at_k(c_p, N, k):
    """
    计算 Fast@k（闭式解）
    
    Fast@k = 1 - C(N - c_p, k) / C(N, k)
    
    其中 c_p 是正确样本数（speedup >= p_thr）
    """
    if c_p < 0 or N <= 0 or k <= 0 or k > N:
        return np.nan
    
    if c_p >= k:
        # 至少有 k 个正确样本，Fast@k = 1
        return 1.0
    
    if c_p == 0:
        # 没有正确样本，Fast@k = 0
        return 0.0
    
    # 一般情况
    numerator = comb(N - c_p, k)
    denominator = comb(N, k)
    
    if denominator == 0:
        return np.nan
    
    return 1.0 - (numerator / denominator)


def compute_speedup_at_k_exact(speedups_correct, p_thr, N, k):
    """
    计算 Speedup@k（精确枚举）
    
    E[Speedup@k] = Σ(所有k组合) max(S_i in 组合) / C(N, k)
    """
    if len(speedups_correct) == 0:
        return 0.0
    
    if k > N or k <= 0:
        return np.nan
    
    # 用 0 填充到 N 个样本（不正确的样本 speedup = 0）
    speedups = list(speedups_correct) + [0.0] * (N - len(speedups_correct))
    
    if k > len(speedups):
        return np.nan
    
    # 精确计算所有组合
    total_speedup = 0.0
    count = 0
    
    for combo in combinations(speedups, k):
        max_speedup = max(combo)
        # 只有当 max_speedup >= p_thr 时才计入
        if max_speedup >= p_thr:
            total_speedup += max_speedup
        count += 1
    
    return total_speedup / count if count > 0 else 0.0


def compute_metrics_by_k(table_a_df, ex_version, dataset_size, p_thr, k_range, threads_filter=None):
    """
    计算不同 k 值下的 Fast@k 和 Speedup@k（benchmark 平均）
    
    Args:
        table_a_df: Table A 数据
        ex_version: EX 版本
        dataset_size: 数据集大小
        p_thr: 性能阈值
        k_range: k 值范围
        threads_filter: EX2 的线程数筛选
        
    Returns:
        DataFrame with columns: model, k, Fast_at_k, Speedup_at_k
    """
    
    print(f"  处理: {ex_version} - {dataset_size}")
    
    # 筛选数据
    df = table_a_df[(table_a_df['ex'] == ex_version) & 
                    (table_a_df['dataset_size'] == dataset_size)].copy()
    
    # EX2 线程数筛选
    if ex_version == 'EX2' and threads_filter is not None:
        df = df[df['threads'] == float(threads_filter)]
    
    if len(df) == 0:
        print(f"    没有数据")
        return pd.DataFrame()
    
    results = []
    
    for model in ALL_MODELS:
        df_model = df[df['model'] == model]
        
        if len(df_model) == 0:
            continue
        
        # 对每个 benchmark 计算 Fast@k 和 Speedup@k
        for k in k_range:
            fast_k_list = []
            speedup_k_list = []
            
            for benchmark in df_model['benchmark'].unique():
                df_bench = df_model[df_model['benchmark'] == benchmark]
                
                # 获取所有 trial 的 speedup（正确的）
                speedups_correct = df_bench[df_bench['correctness'] == 1.0]['kernel_speedup'].values
                c_p = len(speedups_correct)
                
                # 计算 Fast@k
                fast_k = compute_fast_at_k(c_p, N, k)
                if not np.isnan(fast_k):
                    fast_k_list.append(fast_k)
                
                # 计算 Speedup@k (只在有正确样本时计算)
                if c_p > 0:
                    speedup_k = compute_speedup_at_k_exact(speedups_correct, p_thr, N, k)
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


# ==================== 美化版画图函数 ====================

def plot_all_datasets_beautiful(
        table_a_df,
        metric_name='Fast_at_k',
        ylabel='Fast@k',
        filename='fast_at_k_vs_k_beautiful.pdf',
        p_thr=1.05,
        threads_filter=16):
    """
    画所有数据集大小和EX版本的美化网格图
    
    布局：3行（EX1, EX2, EX3）× 5列（dataset sizes）
    
    Args:
        table_a_df: Table A 数据
        metric_name: 指标名称（'Fast_at_k' 或 'Speedup_at_k'）
        ylabel: Y轴标签
        filename: 输出文件名
        p_thr: p_thr 值
        threads_filter: EX2 的线程数
    """
    
    # 创建 3×5 的子图
    fig, axes = plt.subplots(3, 5, figsize=(20, 10))
    
    # 调整子图间距
    plt.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.08, 
                       wspace=0.15, hspace=0.25)
    
    print()
    total_plots = len(ALL_EX_VERSIONS) * len(DATASET_SIZES)
    current_plot = 0
    
    for row_idx, ex in enumerate(ALL_EX_VERSIONS):
        for col_idx, ds in enumerate(DATASET_SIZES):
            ax = axes[row_idx, col_idx]
            current_plot += 1
            
            print(f"[{current_plot}/{total_plots}] 计算 {ex} - {ds}")
            
            # 计算指标
            df_results = compute_metrics_by_k(
                table_a_df, ex, ds, p_thr, K_RANGE, threads_filter
            )
            
            if len(df_results) == 0:
                # 没有数据，显示提示
                ax.text(0.5, 0.5, 'No Data', 
                       ha='center', va='center', fontsize=10, color='gray')
                ax.set_xlim(0, 11)
                ax.set_ylim(0, 1)
            else:
                # 为每个模型画线
                for model in ALL_MODELS:
                    df_model = df_results[df_results['model'] == model].sort_values('k')
                    
                    if len(df_model) == 0:
                        continue
                    
                    ax.plot(
                        df_model['k'],
                        df_model[metric_name],
                        color=MODEL_COLORS[model],
                        linewidth=2,
                        marker=None,  # 不显示 marker
                        markersize=0,
                        alpha=0.85
                    )
            
            # 网格（只保留水平网格，且更淡）
            ax.grid(True, axis='y', alpha=0.15, linestyle='-', linewidth=0.4, color='gray')
            ax.set_axisbelow(True)
            
            # 设置 y 轴范围
            if 'Fast' in metric_name:
                ax.set_ylim(-0.05, 1.05)
            else:
                ax.set_ylim(bottom=0)
            
            # x 轴范围和刻度
            ax.set_xlim(0.5, 10.5)
            ax.set_xticks(range(1, 11, 2))  # 只显示 1, 3, 5, 7, 9
            
            # 列标题（只在第一行显示）
            if row_idx == 0:
                ax.set_title(DATASET_LABELS[ds], fontsize=11, fontweight='normal', pad=15)
            
            # 只在最左列显示 y 轴标签
            if col_idx == 0:
                ax.set_ylabel('')
                ax.tick_params(axis='y', labelsize=8)
            else:
                ax.set_yticklabels([])
            
            # 只在最底行显示 x 轴标签
            if row_idx == 2:
                ax.tick_params(axis='x', labelsize=8)
            else:
                ax.set_xticklabels([])
            
            # 调整刻度字体大小
            ax.tick_params(axis='both', labelsize=8)
    
    # 添加行标签（竖着放在左侧）
    row_labels = ['EX1 (Serial)', 'EX2 (OpenMP)', 'EX3 (CUDA)']
    row_positions = [0.77, 0.50, 0.23]  # 对应三行的中心位置
    for row_idx, label in enumerate(row_labels):
        fig.text(
            0.01, row_positions[row_idx],
            label,
            fontsize=12,
            fontweight='bold',
            ha='center',
            va='center',
            rotation=90
        )
    
    # 添加总的 x 轴和 y 轴标签
    fig.text(0.52, 0.02, r'$k$',
             ha='center', fontsize=13, fontweight='normal')
    fig.text(0.005, 0.5, ylabel,
             ha='center', va='center', rotation=90, fontsize=13, fontweight='normal')
    
    # Legend 放在顶部中央
    handles = [
        plt.Line2D([0], [0], color=MODEL_COLORS[m], linewidth=2.5,
                   marker='o', markersize=5, label=MODEL_LABELS[m])
        for m in ALL_MODELS
    ]
    
    fig.legend(
        handles=handles,
        loc='upper center',
        ncol=3,
        frameon=True,
        fontsize=11,
        bbox_to_anchor=(0.5, 1.02),
        columnspacing=1.5,
        handlelength=2.5
    )
    
    # 保存
    output_file = OUTPUT_DIR / filename
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"✓ 保存: {output_file}")
    
    plt.close()


# ==================== 主函数 ====================

def main():
    """主函数：生成美化版 k-Sensitivity 图"""
    
    print("="*80)
    print("生成美化版 k-Sensitivity 图")
    print("="*80)
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"固定 p_thr: {DEFAULT_P_THR}")
    print(f"k 范围: {list(K_RANGE)}")
    print()
    
    # 读取 Table A 数据
    print("加载 Table A 数据...")
    table_a_files = {}
    
    # 加载所有数据集大小的数据
    for ds in DATASET_SIZES:
        for ex in ['EX1', 'EX2', 'EX3']:
            key = f"{ex}_{ds}"
            filepath = TABLE_DIR / f"table_a_trial_level_{ex}_{ds}.csv"
            
            if filepath.exists():
                table_a_files[key] = filepath
                print(f"  ✓ 找到: {filepath.name}")
            else:
                print(f"  ⚠️  未找到: {filepath.name}")
    
    if len(table_a_files) == 0:
        print("❌ 没有找到任何 Table A 数据！")
        return
    
    # 合并所有数据
    dfs = []
    for filepath in table_a_files.values():
        df = pd.read_csv(filepath)
        dfs.append(df)
    
    table_a_df = pd.concat(dfs, ignore_index=True)
    print(f"总行数: {len(table_a_df)}")
    print()
    
    # 生成 Fast@k vs k 图
    print("="*80)
    print("生成 Fast@k vs k 美化图...")
    print("="*80)
    
    plot_all_datasets_beautiful(
        table_a_df,
        metric_name='Fast_at_k',
        ylabel='Fast@k',
        filename='fast_at_k_vs_k_beautiful.pdf',
        p_thr=DEFAULT_P_THR,
        threads_filter=16
    )
    
    print()
    
    # 生成 Speedup@k vs k 图
    print("="*80)
    print("生成 Speedup@k vs k 美化图...")
    print("="*80)
    
    plot_all_datasets_beautiful(
        table_a_df,
        metric_name='Speedup_at_k',
        ylabel='Speedup@k',
        filename='speedup_at_k_vs_k_beautiful.pdf',
        p_thr=DEFAULT_P_THR,
        threads_filter=16
    )
    
    print()
    print("="*80)
    print("✅ 美化版 k-Sensitivity 图生成完成！")
    print("="*80)


if __name__ == "__main__":
    main()
