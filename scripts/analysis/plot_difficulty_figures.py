#!/usr/bin/env python3
"""
HPC-Bench Difficulty 分层图可视化脚本

功能：
- 按 Difficulty 分层画图（d1, d2, d3, d4）
- 支持单图模式和网格模式（2×5 大图）
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path


# ==================== 配置参数 ====================

# 设置画图风格
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

# 目录配置
WORKSPACE_ROOT = Path(__file__).parent.parent.parent
TABLE_DIR = WORKSPACE_ROOT / "analysis_summaries" / "tables"
FIGURE_DIR = WORKSPACE_ROOT / "analysis_summaries" / "d_category"

# 实验配置
EX_VERSIONS = ['EX1', 'EX2']
DATASET_SIZES = ['mini', 'small', 'medium', 'large', 'extra-large']

# 模型标签
MODEL_LABELS = {
    'claude': 'Claude',
    'gpt5.1': 'GPT-5.1',
    'qwen': 'Qwen'
}

# Difficulty 颜色和标签
DIFFICULTY_COLORS = {
    'd1': '#1f77b4',  # 蓝色
    'd2': '#ff7f0e',  # 橙色
    'd3': '#2ca02c',  # 绿色
    'd4': '#d62728'   # 红色
}

DIFFICULTY_LABELS = {
    'd1': 'd1',
    'd2': 'd2',
    'd3': 'd3',
    'd4': 'd4'
}

DIFFICULTIES = ['d1', 'd2', 'd3', 'd4']


# ==================== 主要功能：按 Difficulty 分层画图（单图模式） ====================

def plot_by_difficulty_single(
        metric_name='Fast_at_k_avg',
        ylabel='Fast@3',
        title_prefix='Fast@3',
        filename_prefix='figure_fast_at_k_by_difficulty',
        model='claude',
        threads_filter=16):
    """
    为每个 (EX, dataset_size) 组合生成单独的 difficulty 分层图
    
    Args:
        metric_name: 指标列名 ('Fast_at_k_avg' 或 'Speedup_at_k_avg')
        ylabel: Y轴标签
        title_prefix: 标题前缀
        filename_prefix: 文件名前缀
        model: 选择要画的 model
        threads_filter: EX2 使用的线程数（默认 16）
    """
    
    # ---- Style (按照 ipynb 配置) ----
    SUBTITLE_FS = 20
    LABEL_FS = 20
    TICK_FS = 18
    LEGEND_FS = 18
    LINE_W = 2.5
    MARK_SZ = 6
    
    generated_count = 0
    
    for ex in EX_VERSIONS:
        for ds in DATASET_SIZES:
            # 读取 Table G（difficulty-level aggregation）
            table_g_file = TABLE_DIR / f"table_g_difficulty_k3_{ex}_{ds}.csv"
            
            if not table_g_file.exists():
                continue
            
            df_g = pd.read_csv(table_g_file)
            
            # 对于 EX2，只保留指定线程数的数据
            if ex == 'EX2':
                df_g = df_g[df_g['threads'] == float(threads_filter)]
            
            # 筛选 model
            df_model = df_g[df_g['model'] == model]
            
            if len(df_model) == 0:
                continue
            
            # 创建图
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 为每个 difficulty 画一条线
            for difficulty in DIFFICULTIES:
                df_diff = df_model[df_model['difficulty'] == difficulty].sort_values('p_thr')
                
                if len(df_diff) == 0:
                    continue
                
                ax.plot(df_diff['p_thr'],
                       df_diff[metric_name],
                       marker='o',
                       linewidth=LINE_W,
                       markersize=MARK_SZ,
                       label=DIFFICULTY_LABELS[difficulty],
                       color=DIFFICULTY_COLORS[difficulty],
                       alpha=0.9)
            
            # 简化标题：只显示 dataset_size 和 ex
            ax.set_title(f'{ds}\n({ex})', fontsize=SUBTITLE_FS, pad=6)
            ax.set_xlabel(r'$p_{\mathrm{thr}}$', fontsize=LABEL_FS)
            ax.set_ylabel(ylabel, fontsize=LABEL_FS)
            ax.grid(True, alpha=0.25)
            ax.tick_params(axis='both', labelsize=TICK_FS)
            ax.legend(loc='best', frameon=False, fontsize=LEGEND_FS)
            
            if 'Fast' in metric_name:
                ax.set_ylim(-0.05, 1.05)
            else:
                ax.set_ylim(bottom=0)
            
            plt.tight_layout()
            
            # 保存
            output_path = FIGURE_DIR / f"{filename_prefix}_{ex}_{ds}.pdf"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            generated_count += 1
    
    return generated_count


# ==================== 高级功能：Difficulty 网格图（2×5 大图） ====================

def plot_by_difficulty_grid(
        metric_name='Fast_at_k_avg',
        ylabel='Fast@3',
        title_prefix='Fast@3 by Difficulty',
        filename='figure_fast_at_k_by_difficulty_grid.pdf',
        model='claude',
        ex_versions=('EX1', 'EX2'),
        dataset_sizes=('mini', 'small', 'medium', 'large', 'extra-large'),
        ex2_threads_keep=16.0):
    """
    按 difficulty 分层画图（使用 Table G），做成 2×5 的大图：
      - 第一行: EX1 的 5 个 dataset size
      - 第二行: EX2 的 5 个 dataset size
    每个子图 4 条线 (d1, d2, d3, d4)，全局 legend 平铺底部。
    只画一个指定 model（默认 claude）。

    输入: table_g_difficulty_k3_{EX}_{dataset_size}.csv
    输出: 一个 PDF 大图
    """

    # ---- Style (bigger fonts for grid) ----
    TITLE_FS = 22
    SUBTITLE_FS = 20
    LABEL_FS = 20
    TICK_FS = 18
    LEGEND_FS = 18
    LINE_W = 2.5
    MARK_SZ = 6

    n_rows = len(ex_versions)
    n_cols = len(dataset_sizes)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 10), sharex=False, sharey=False)

    legend_handles = {}

    for row_idx, ex in enumerate(ex_versions):
        for col_idx, ds in enumerate(dataset_sizes):
            ax = axes[row_idx, col_idx]

            table_g_file = TABLE_DIR / f"table_g_difficulty_k3_{ex}_{ds}.csv"
            
            if not table_g_file.exists():
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=LABEL_FS)
                ax.set_title(f'{ds}\n({ex})', fontsize=SUBTITLE_FS, pad=6)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            df_g = pd.read_csv(table_g_file)

            # EX2: keep a representative thread count
            if ex == 'EX2' and 'threads' in df_g.columns:
                df_g = df_g[df_g['threads'] == float(ex2_threads_keep)]

            # filter model
            df_m = df_g[df_g['model'] == model]
            
            if len(df_m) == 0:
                ax.text(0.5, 0.5, 'No Model Data', ha='center', va='center', fontsize=LABEL_FS)
                ax.set_title(f'{ds}\n({ex})', fontsize=SUBTITLE_FS, pad=6)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            # plot 4 difficulty lines
            for diff in DIFFICULTIES:
                df_d = df_m[df_m['difficulty'] == diff].sort_values('p_thr')
                
                if len(df_d) == 0:
                    continue

                line, = ax.plot(
                    df_d['p_thr'],
                    df_d[metric_name],
                    marker='o',
                    linewidth=LINE_W,
                    markersize=MARK_SZ,
                    color=DIFFICULTY_COLORS[diff],
                    alpha=0.9,
                    label=DIFFICULTY_LABELS[diff]
                )

                # store for global legend
                if diff not in legend_handles:
                    legend_handles[diff] = line

            # titles / ticks / grid
            ax.set_title(f'{ds}\n({ex})', fontsize=SUBTITLE_FS, pad=6)
            ax.grid(True, alpha=0.25)
            ax.tick_params(axis='both', labelsize=TICK_FS)

            # only bottom row x-label
            if row_idx == n_rows - 1:
                ax.set_xlabel(r'$p_{\mathrm{thr}}$', fontsize=LABEL_FS)
            else:
                ax.set_xlabel('')

            # only left col y-label
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=LABEL_FS)
            else:
                ax.set_ylabel('')

            # y-limits
            if 'Fast' in metric_name:
                ax.set_ylim(-0.05, 1.05)
            else:
                ax.set_ylim(bottom=0)

    # ---- Global legend at bottom (flat) ----
    handles = [legend_handles[d] for d in DIFFICULTIES if d in legend_handles]
    labels = [DIFFICULTY_LABELS[d] for d in DIFFICULTIES if d in legend_handles]

    fig.legend(
        handles, labels,
        loc='lower center',
        ncol=len(labels),
        fontsize=LEGEND_FS,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02)
    )

    # layout: leave space for legend
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    output_path = FIGURE_DIR / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return 1


# ==================== 主函数 ====================

def main():
    """主函数：生成 Difficulty 分层图"""
    
    parser = argparse.ArgumentParser(description='生成 HPC-Bench Difficulty 分层图')
    
    # 基本参数
    parser.add_argument('--model', type=str, default='claude',
                       choices=['claude', 'gpt5.1', 'qwen'],
                       help='选择模型 (默认: claude)')
    parser.add_argument('--threads', type=int, default=16,
                       help='EX2 使用的线程数 (默认: 16)')
    
    # 指标选择
    parser.add_argument('--metric', type=str, default='Fast_at_k_avg',
                       choices=['Fast_at_k_avg', 'Speedup_at_k_avg'],
                       help='选择指标 (默认: Fast_at_k_avg)')
    
    # 画图模式
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'grid', 'both'],
                       help='画图模式: single=单独图, grid=2×5网格图, both=都生成 (默认: single)')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("HPC-Bench Difficulty 分层图生成")
    print("="*80)
    print(f"输出目录: {FIGURE_DIR}")
    print(f"模型: {args.model}")
    print(f"EX2 线程数: {args.threads}")
    print(f"指标: {args.metric}")
    print(f"模式: {args.mode}")
    print()
    
    # 准备参数
    ylabel = 'Fast@3' if args.metric == 'Fast_at_k_avg' else 'Speedup@3'
    title_prefix = 'Fast@3' if args.metric == 'Fast_at_k_avg' else 'Speedup@3'
    filename_prefix = 'figure_fast_at_k_by_difficulty' if args.metric == 'Fast_at_k_avg' else 'figure_speedup_at_k_by_difficulty'
    
    total_count = 0
    
    # 生成单独图
    if args.mode in ['single', 'both']:
        print("="*80)
        print("生成 Difficulty 分层图（单图模式）")
        print("="*80)
        
        count = plot_by_difficulty_single(
            metric_name=args.metric,
            ylabel=ylabel,
            title_prefix=title_prefix,
            filename_prefix=filename_prefix,
            model=args.model,
            threads_filter=args.threads
        )
        
        print(f"✓ 生成了 {count} 张单独图片")
        print()
        total_count += count
    
    # 生成网格图
    if args.mode in ['grid', 'both']:
        print("="*80)
        print("生成 Difficulty 分层图（2×5 网格模式）")
        print("="*80)
        
        grid_filename = f"{filename_prefix}_grid.pdf"
        
        count = plot_by_difficulty_grid(
            metric_name=args.metric,
            ylabel=ylabel,
            title_prefix=title_prefix,
            filename=grid_filename,
            model=args.model,
            ex2_threads_keep=float(args.threads)
        )
        
        print(f"✓ 生成了 1 张网格大图: {grid_filename}")
        print()
        total_count += count
    
    print("="*80)
    print(f"✅ Difficulty 图表生成完成！共生成 {total_count} 张图")
    print("="*80)


if __name__ == "__main__":
    main()
