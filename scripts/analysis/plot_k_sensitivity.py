#!/usr/bin/env python3
"""
HPC-Bench k-Sensitivity Analysis

分析固定 p_thr 下，k 值变化（1-10）对 Fast@k 和 Speedup@k 的影响
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
from itertools import combinations
from math import comb


# ==================== 配置参数 ====================

# 设置画图风格
sns.set_style("whitegrid")

# 目录配置
WORKSPACE_ROOT = Path(__file__).parent.parent.parent
TABLE_DIR = WORKSPACE_ROOT / "analysis_summaries" / "tables"
FIGURE_DIR = WORKSPACE_ROOT / "analysis_summaries" / "sensitive_analysis"

# 实验配置
ALL_EX_VERSIONS = ['EX1', 'EX2', 'EX3']
DATASET_SIZES = ['mini', 'small', 'medium', 'large', 'extra-large']

# 模型配置
ALL_MODELS = ['claude', 'gpt5.1', 'qwen']
MODEL_LABELS = {
    'claude': 'Claude',
    'gpt5.1': 'GPT-5.1',
    'qwen': 'Qwen'
}

MODEL_COLORS = {
    'claude': '#1f77b4',   # 蓝色
    'gpt5.1': '#ff7f0e',   # 橙色
    'qwen': '#2ca02c'      # 绿色
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
    
    # 枚举所有 k 组合，计算每个组合的最大 speedup
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
    
    # 筛选数据
    df = table_a_df[(table_a_df['ex'] == ex_version) & 
                    (table_a_df['dataset_size'] == dataset_size)].copy()
    
    # EX2 线程数筛选
    if ex_version == 'EX2' and threads_filter is not None:
        df = df[df['threads'] == float(threads_filter)]
    
    if len(df) == 0:
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
                speedups_correct = df_bench[df_bench['correctness'] == 1]['kernel_speedup'].values
                c_p = len(speedups_correct)
                
                # 计算 Fast@k
                fast_k = compute_fast_at_k(c_p, N, k)
                if not np.isnan(fast_k):
                    fast_k_list.append(fast_k)
                
                # 计算 Speedup@k
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
    
    return pd.DataFrame(results)


# ==================== 画图函数 ====================

def plot_metric_vs_k(
        df_results,
        metric_name='Fast_at_k',
        ylabel='Fast@k',
        title='Fast@k vs k',
        filename='fast_at_k_vs_k.pdf',
        ex_version='EX1',
        dataset_size='medium',
        p_thr=1.05):
    """
    画 Fast@k 或 Speedup@k vs k 的图
    
    Args:
        df_results: 包含 model, k, Fast_at_k, Speedup_at_k 的 DataFrame
        metric_name: 指标名称
        ylabel: Y轴标签
        title: 图标题
        filename: 输出文件名
        ex_version: EX版本（用于标题）
        dataset_size: 数据集大小（用于标题）
        p_thr: p_thr 值（用于标题）
    """
    
    # ---- Style ----
    TITLE_FS = 20
    LABEL_FS = 20
    TICK_FS = 18
    LEGEND_FS = 18
    LINE_W = 2.5
    MARK_SZ = 6
    
    if len(df_results) == 0:
        print(f"⚠️  No data for {ex_version} {dataset_size}, skipping plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 为每个模型画一条线
    for model in ALL_MODELS:
        df_model = df_results[df_results['model'] == model].sort_values('k')
        
        if len(df_model) == 0:
            continue
        
        ax.plot(df_model['k'],
               df_model[metric_name],
               marker='o',
               linewidth=LINE_W,
               markersize=MARK_SZ,
               label=MODEL_LABELS.get(model, model),
               color=MODEL_COLORS[model],
               alpha=0.9)
    
    # 标题（简化）
    ax.set_title(f'{dataset_size}\n({ex_version})', fontsize=TITLE_FS, pad=6)
    ax.set_xlabel('k', fontsize=LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis='both', labelsize=TICK_FS)
    ax.legend(loc='best', frameon=False, fontsize=LEGEND_FS)
    
    # x 轴刻度：1, 2, 3, ..., 10
    ax.set_xticks(range(1, 11))
    
    # y 轴范围
    if 'Fast' in metric_name:
        ax.set_ylim(-0.05, 1.05)
    else:
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    # 保存
    output_path = FIGURE_DIR / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {filename}")


def plot_all_ex_grid(
        table_a_df,
        metric_name='Fast_at_k',
        ylabel='Fast@k',
        title_prefix='Fast@k vs k',
        filename='fast_at_k_vs_k_all_ex.pdf',
        dataset_size='medium',
        p_thr=1.05,
        threads_filter=16):
    """
    画所有 EX 版本的网格图（EX1, EX2, EX3）
    
    Args:
        table_a_df: Table A 数据
        metric_name: 指标名称
        ylabel: Y轴标签
        title_prefix: 标题前缀
        filename: 输出文件名
        dataset_size: 数据集大小
        p_thr: p_thr 值
        threads_filter: EX2 的线程数
    """
    
    # ---- Style ----
    TITLE_FS = 20
    LABEL_FS = 20
    TICK_FS = 18
    LEGEND_FS = 16
    LINE_W = 2.5
    MARK_SZ = 6
    
    # 创建 1x3 的子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    legend_handles = {}
    
    for idx, ex in enumerate(ALL_EX_VERSIONS):
        ax = axes[idx]
        
        # EX3 暂时留空
        if ex == 'EX3':
            ax.text(0.5, 0.5, 'EX3\n(To be evaluated)', 
                   ha='center', va='center', fontsize=LABEL_FS, color='gray')
            ax.set_title(f'{dataset_size}\n({ex})', fontsize=TITLE_FS, pad=6)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        # 计算指标
        df_results = compute_metrics_by_k(
            table_a_df, ex, dataset_size, p_thr, K_RANGE, threads_filter
        )
        
        if len(df_results) == 0:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=LABEL_FS)
            ax.set_title(f'{dataset_size}\n({ex})', fontsize=TITLE_FS, pad=6)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        # 为每个模型画线
        for model in ALL_MODELS:
            df_model = df_results[df_results['model'] == model].sort_values('k')
            
            if len(df_model) == 0:
                continue
            
            line, = ax.plot(df_model['k'],
                          df_model[metric_name],
                          marker='o',
                          linewidth=LINE_W,
                          markersize=MARK_SZ,
                          label=MODEL_LABELS.get(model, model),
                          color=MODEL_COLORS[model],
                          alpha=0.9)
            
            # 存储用于全局 legend
            if model not in legend_handles:
                legend_handles[model] = line
        
        ax.set_title(f'{dataset_size}\n({ex})', fontsize=TITLE_FS, pad=6)
        ax.set_xlabel('k', fontsize=LABEL_FS)
        
        if idx == 0:
            ax.set_ylabel(ylabel, fontsize=LABEL_FS)
        
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis='both', labelsize=TICK_FS)
        ax.set_xticks(range(1, 11))
        
        if 'Fast' in metric_name:
            ax.set_ylim(-0.05, 1.05)
        else:
            ax.set_ylim(bottom=0)
    
    # 全局 legend
    handles = [legend_handles[m] for m in ALL_MODELS if m in legend_handles]
    labels = [MODEL_LABELS[m] for m in ALL_MODELS if m in legend_handles]
    
    fig.legend(handles, labels,
              loc='lower center',
              ncol=len(labels),
              fontsize=LEGEND_FS,
              frameon=False,
              bbox_to_anchor=(0.5, -0.05))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    output_path = FIGURE_DIR / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {filename}")


# ==================== 主函数 ====================

def main():
    """主函数：k-Sensitivity Analysis"""
    
    parser = argparse.ArgumentParser(description='HPC-Bench k-Sensitivity Analysis')
    
    # 基本参数
    parser.add_argument('--dataset-size', type=str, default='medium',
                       choices=DATASET_SIZES,
                       help='数据集大小 (默认: medium)')
    parser.add_argument('--p-thr', type=float, default=1.05,
                       help='固定 p_thr 值 (默认: 1.05)')
    parser.add_argument('--threads', type=int, default=16,
                       help='EX2 使用的线程数 (默认: 16)')
    
    # 模式选择
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'grid', 'both'],
                       help='画图模式: single=单个EX(EX1), grid=全部EX网格, both=都生成 (默认: single)')
    
    parser.add_argument('--ex', type=str, default='EX1',
                       choices=['EX1', 'EX2'],
                       help='单图模式下的 EX 版本 (默认: EX1)')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("HPC-Bench k-Sensitivity Analysis")
    print("="*80)
    print(f"输出目录: {FIGURE_DIR}")
    print(f"数据集大小: {args.dataset_size}")
    print(f"固定 p_thr: {args.p_thr}")
    print(f"EX2 线程数: {args.threads}")
    print(f"k 范围: {list(K_RANGE)}")
    print(f"模式: {args.mode}")
    print()
    
    # 读取 Table A
    print("Loading Table A data...")
    table_a_files = {
        'EX1': TABLE_DIR / f"table_a_trial_level_EX1_{args.dataset_size}.csv",
        'EX2': TABLE_DIR / f"table_a_trial_level_EX2_{args.dataset_size}.csv"
    }
    
    # 合并 EX1 和 EX2 的数据
    dfs = []
    for ex, filepath in table_a_files.items():
        if filepath.exists():
            df = pd.read_csv(filepath)
            dfs.append(df)
            print(f"  ✓ Loaded {filepath.name}")
        else:
            print(f"  ⚠️  Not found: {filepath.name}")
    
    if len(dfs) == 0:
        print("❌ No Table A data found!")
        return
    
    table_a_df = pd.concat(dfs, ignore_index=True)
    print(f"Total rows: {len(table_a_df)}")
    print()
    
    # ==================== 单图模式 ====================
    
    if args.mode in ['single', 'both']:
        print("="*80)
        print(f"生成单图模式 ({args.ex}, {args.dataset_size})")
        print("="*80)
        
        # 计算 Fast@k 和 Speedup@k
        df_results = compute_metrics_by_k(
            table_a_df, args.ex, args.dataset_size, args.p_thr, K_RANGE, args.threads
        )
        
        if len(df_results) > 0:
            # Fast@k vs k
            plot_metric_vs_k(
                df_results,
                metric_name='Fast_at_k',
                ylabel='Fast@k',
                title='Fast@k vs k',
                filename=f'fast_at_k_vs_k_{args.ex}_{args.dataset_size}.pdf',
                ex_version=args.ex,
                dataset_size=args.dataset_size,
                p_thr=args.p_thr
            )
            
            # Speedup@k vs k
            plot_metric_vs_k(
                df_results,
                metric_name='Speedup_at_k',
                ylabel='Speedup@k',
                title='Speedup@k vs k',
                filename=f'speedup_at_k_vs_k_{args.ex}_{args.dataset_size}.pdf',
                ex_version=args.ex,
                dataset_size=args.dataset_size,
                p_thr=args.p_thr
            )
            
            print("✓ 单图模式完成")
        else:
            print(f"⚠️  No data for {args.ex} {args.dataset_size}")
        
        print()
    
    # ==================== 网格模式 ====================
    
    if args.mode in ['grid', 'both']:
        print("="*80)
        print(f"生成网格模式 (所有 EX, {args.dataset_size})")
        print("="*80)
        
        # Fast@k vs k (all EX)
        plot_all_ex_grid(
            table_a_df,
            metric_name='Fast_at_k',
            ylabel='Fast@k',
            title_prefix='Fast@k vs k',
            filename=f'fast_at_k_vs_k_all_ex_{args.dataset_size}.pdf',
            dataset_size=args.dataset_size,
            p_thr=args.p_thr,
            threads_filter=args.threads
        )
        
        # Speedup@k vs k (all EX)
        plot_all_ex_grid(
            table_a_df,
            metric_name='Speedup_at_k',
            ylabel='Speedup@k',
            title_prefix='Speedup@k vs k',
            filename=f'speedup_at_k_vs_k_all_ex_{args.dataset_size}.pdf',
            dataset_size=args.dataset_size,
            p_thr=args.p_thr,
            threads_filter=args.threads
        )
        
        print("✓ 网格模式完成")
        print()
    
    print("="*80)
    print("✅ k-Sensitivity Analysis 完成！")
    print("="*80)


if __name__ == "__main__":
    main()
