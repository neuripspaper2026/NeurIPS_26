#!/usr/bin/env python3
"""
HPC-Bench Difficulty 分层图可视化脚本 - 美化版

功能：
- 按 Difficulty 分层画图（d1, d2, d3, d4）
- 支持单图模式和网格模式（2×5 大图）
- 美化风格：去掉 markers，弱化网格，优化图例
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


# ==================== 配置参数 ====================

# 目录配置
WORKSPACE_ROOT = Path(__file__).parent.parent.parent
TABLE_DIR = WORKSPACE_ROOT / "analysis_summaries" / "tables"
OUTPUT_DIR = WORKSPACE_ROOT / "analysis_summaries" / "beautiful_figure" / "main_paper" / "diff-level"

# 确保输出目录存在
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 实验配置
EX_VERSIONS = ['EX1', 'EX2']
DATASET_SIZES = ['mini', 'small', 'medium', 'large', 'extra-large']

# 模型标签（与 beautiful 版本一致）
MODEL_LABELS = {
    'claude': 'Claude',
    'gpt5.1': 'GPT-4o',
    'qwen': 'Qwen'
}

# Difficulty 颜色和标签
DIFFICULTY_COLORS = {
    'd1': '#2E86AB',
    'd2': '#A23B72',
    'd3': '#F18F01',
    'd4': '#d62728'
}

DIFFICULTY_LABELS = {
    'd1': 'd1',
    'd2': 'd2',
    'd3': 'd3',
    'd4': 'd4'
}

DIFFICULTIES = ['d1', 'd2', 'd3', 'd4']

# Dataset 标签
DATASET_LABELS = {
    'mini': 'Mini',
    'small': 'Small',
    'medium': 'Medium',
    'large': 'Large',
    'extra-large': 'Extra-Large'
}


# ==================== 主要功能：按 Difficulty 分层画图（单图模式） ====================

def plot_by_difficulty_single_beautiful(
        metric_name='Fast_at_k_avg',
        ylabel='Fast@3',
        filename_prefix='fast_at_k_by_difficulty',
        model='claude',
        threads_filter=16):
    """为每个 (EX, dataset_size) 生成单独的 difficulty 图（美化版）"""
    
    generated_count = 0
    
    for ex in EX_VERSIONS:
        for ds in DATASET_SIZES:
            # 读取 Table G
            table_g_file = TABLE_DIR / f"table_g_difficulty_k3_{ex}_{ds}.csv"
            
            if not table_g_file.exists():
                continue
            
            df_g = pd.read_csv(table_g_file)
            
            # EX2 筛选线程数
            if ex == 'EX2':
                df_g = df_g[df_g['threads'] == float(threads_filter)]
            
            # 筛选 model
            df_model = df_g[df_g['model'] == model]
            
            if len(df_model) == 0:
                continue
            
            # 创建图
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # 为每个 difficulty 画线
            for difficulty in DIFFICULTIES:
                df_diff = df_model[df_model['difficulty'] == difficulty].sort_values('p_thr')
                
                if len(df_diff) == 0:
                    continue
                
                ax.plot(df_diff['p_thr'],
                       df_diff[metric_name],
                       marker=None,  # 不显示 marker
                       linewidth=2.5,
                       label=DIFFICULTY_LABELS[difficulty],
                       color=DIFFICULTY_COLORS[difficulty],
                       alpha=0.85)
            
            # 网格（只保留水平网格，且更淡）
            ax.grid(True, axis='y', alpha=0.15, linestyle='-', linewidth=0.4, color='gray')
            ax.set_axisbelow(True)
            
            # 标题和标签
            ex_labels = {'EX1': 'EX1 (Serial)', 'EX2': 'EX2 (OpenMP)'}
            
            ax.set_title(f'{ex_labels[ex]} - {DATASET_LABELS[ds]}', 
                        fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel(r'Speedup threshold $p_{\mathrm{thr}}$', fontsize=13)
            ax.set_ylabel(ylabel, fontsize=13)
            
            # 图例（右下角，半透明）
            ax.legend(loc='lower right', frameon=True, fontsize=10, 
                     framealpha=0.7, edgecolor='gray')
            
            # Y轴范围
            if 'Fast' in metric_name:
                ax.set_ylim(-0.05, 1.05)
            else:
                ax.set_ylim(bottom=0)
            
            ax.tick_params(axis='both', labelsize=10)
            
            plt.tight_layout()
            
            # 保存
            output_path = OUTPUT_DIR / f"{filename_prefix}_{ex}_{ds}.pdf"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            generated_count += 1
    
    return generated_count


# ==================== 高级功能：Difficulty 网格图（2×5 大图） ====================

def plot_by_difficulty_grid_beautiful(
        metric_name='Fast_at_k_avg',
        ylabel='Fast@3',
        filename='fast_at_k_by_difficulty_grid.pdf',
        model='claude',
        ex2_threads_keep=16.0):
    """按 difficulty 分层画 2×5 网格图（美化版）"""
    
    n_rows = len(EX_VERSIONS)
    n_cols = len(DATASET_SIZES)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 8))
    
    # 调整子图间距
    plt.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.12, 
                       wspace=0.15, hspace=0.25)
    
    legend_handles = {}
    
    for row_idx, ex in enumerate(EX_VERSIONS):
        for col_idx, ds in enumerate(DATASET_SIZES):
            ax = axes[row_idx, col_idx]
            
            table_g_file = TABLE_DIR / f"table_g_difficulty_k3_{ex}_{ds}.csv"
            
            if not table_g_file.exists():
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                # 列标题
                if row_idx == 0:
                    ax.set_title(DATASET_LABELS[ds], fontsize=11, fontweight='normal', pad=15)
                continue
            
            df_g = pd.read_csv(table_g_file)
            
            # EX2 筛选
            if ex == 'EX2' and 'threads' in df_g.columns:
                df_g = df_g[df_g['threads'] == float(ex2_threads_keep)]
            
            # 筛选 model
            df_m = df_g[df_g['model'] == model]
            
            if len(df_m) == 0:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                # 列标题
                if row_idx == 0:
                    ax.set_title(DATASET_LABELS[ds], fontsize=11, fontweight='normal', pad=15)
                continue
            
            # 为每个 difficulty 画线
            for diff in DIFFICULTIES:
                df_d = df_m[df_m['difficulty'] == diff].sort_values('p_thr')
                
                if len(df_d) == 0:
                    continue
                
                line, = ax.plot(
                    df_d['p_thr'],
                    df_d[metric_name],
                    marker=None,  # 不显示 marker
                    linewidth=2,
                    color=DIFFICULTY_COLORS[diff],
                    alpha=0.85,
                    label=DIFFICULTY_LABELS[diff]
                )
                
                # 存储用于全局 legend
                if diff not in legend_handles:
                    legend_handles[diff] = line
            
            # 网格（只保留水平网格，且更淡）
            ax.grid(True, axis='y', alpha=0.15, linestyle='-', linewidth=0.4, color='gray')
            ax.set_axisbelow(True)
            
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
            if row_idx == n_rows - 1:
                ax.tick_params(axis='x', labelsize=8)
            else:
                ax.set_xticklabels([])
            
            # Y轴范围
            if 'Fast' in metric_name:
                ax.set_ylim(-0.05, 1.05)
            else:
                ax.set_ylim(bottom=0)
            
            ax.tick_params(axis='both', labelsize=8)
    
    # 添加行标签（竖着放在左侧）
    row_labels = ['EX1 (Serial)', 'EX2 (OpenMP)']
    row_positions = [0.65, 0.30]  # 对应两行的中心位置
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
    fig.text(0.52, 0.02, r'Speedup threshold $p_{\mathrm{thr}}$',
             ha='center', fontsize=13, fontweight='normal')
    fig.text(0.005, 0.5, ylabel,
             ha='center', va='center', rotation=90, fontsize=13, fontweight='normal')
    
    # Legend 放在顶部中央
    handles = [legend_handles[d] for d in DIFFICULTIES if d in legend_handles]
    labels = [DIFFICULTY_LABELS[d] for d in DIFFICULTIES if d in legend_handles]
    
    fig.legend(
        handles, labels,
        loc='upper center',
        ncol=4,
        frameon=True,
        fontsize=11,
        bbox_to_anchor=(0.5, 1.02),
        columnspacing=1.5,
        handlelength=2.5
    )
    
    # 保存
    output_path = OUTPUT_DIR / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return 1


# ==================== 主函数 ====================

def main():
    """主函数：生成美化版 Difficulty 分层图"""
    
    parser = argparse.ArgumentParser(description='生成美化版 HPC-Bench Difficulty 分层图')
    
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
    
    print("="*80)
    print("美化版 Difficulty 分层图生成")
    print("="*80)
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"模型: {args.model}")
    print(f"EX2 线程数: {args.threads}")
    print(f"指标: {args.metric}")
    print(f"模式: {args.mode}")
    print()
    
    # 准备参数
    ylabel = 'Fast@3' if args.metric == 'Fast_at_k_avg' else 'Speedup@3'
    filename_prefix = 'fast_at_k_by_difficulty' if args.metric == 'Fast_at_k_avg' else 'speedup_at_k_by_difficulty'
    
    total_count = 0
    
    # 生成单独图
    if args.mode in ['single', 'both']:
        print("="*80)
        print("生成 Difficulty 分层图（单图模式）")
        print("="*80)
        
        count = plot_by_difficulty_single_beautiful(
            metric_name=args.metric,
            ylabel=ylabel,
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
        
        count = plot_by_difficulty_grid_beautiful(
            metric_name=args.metric,
            ylabel=ylabel,
            filename=grid_filename,
            model=args.model,
            ex2_threads_keep=float(args.threads)
        )
        
        print(f"✓ 生成了 1 张网格大图: {grid_filename}")
        print()
        total_count += count
    
    print("="*80)
    print(f"✅ 美化版 Difficulty 图表生成完成！共生成 {total_count} 张图")
    print("="*80)


if __name__ == "__main__":
    main()
