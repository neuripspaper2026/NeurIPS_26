#!/usr/bin/env python3
"""
HPC-Bench Motif 分层图可视化脚本

功能：
- 按 Motif 分层画图 (支持 Difficulty 筛选)
- 为特定 Motif 画不同 Difficulty 的对比图
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
FIGURE_DIR = WORKSPACE_ROOT / "analysis_summaries" / "motif_category_by_d_level"

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
    'd1': 'Difficulty 1 (Easy)',
    'd2': 'Difficulty 2',
    'd3': 'Difficulty 3',
    'd4': 'Difficulty 4 (Hard)'
}

# 默认选择的代表性 motif
DEFAULT_MOTIFS = [
    'dense_linear_algebra',
    'stencil_computations',
    'image_and_video_processing',
    'n_body_methods',
    'statistical_computations'
]


# ==================== 主要功能：按 Motif 分层画图 ====================

def plot_by_motif(
        metric_name='Fast_at_k',
        ylabel='Fast@3',
        title_prefix='Fast@3',
        filename_prefix='figure_fast_at_k_by_motif',
        selected_motifs=None,
        difficulty_filter=None,
        model='claude',
        threads_filter=16):
    """
    按 motif 分层画图，支持 difficulty 筛选（使用 Table D）
    
    Args:
        metric_name: 指标列名 ('Fast_at_k' 或 'Speedup_at_k')
        ylabel: Y轴标签
        title_prefix: 标题前缀
        filename_prefix: 文件名前缀
        selected_motifs: 选择要画的 motif 列表（None 表示使用默认）
        difficulty_filter: 筛选特定 difficulty（None=全部, 或 'd1', 'd2', 'd3', 'd4'）
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
    
    # 如果没有指定，使用默认 motif
    if selected_motifs is None:
        selected_motifs = DEFAULT_MOTIFS
    
    # 为每个 motif 分配颜色
    motif_colors = plt.cm.tab10(range(len(selected_motifs)))
    MOTIF_COLORS = dict(zip(selected_motifs, motif_colors))
    
    generated_count = 0
    
    for ex in EX_VERSIONS:
        for ds in DATASET_SIZES:
            # 读取 Table D（包含 difficulty 信息）
            table_d_file = TABLE_DIR / f"table_d_per_benchmark_k3_{ex}_{ds}.csv"
            
            if not table_d_file.exists():
                continue
            
            df_d = pd.read_csv(table_d_file)
            
            # 对于 EX2，只保留指定线程数的数据
            if ex == 'EX2':
                df_d = df_d[df_d['threads'] == float(threads_filter)]
            
            # 筛选 model
            df_model = df_d[df_d['model'] == model]
            
            # 筛选 difficulty（如果指定）
            if difficulty_filter is not None:
                df_model = df_model[df_model['difficulty'] == difficulty_filter]
            
            if len(df_model) == 0:
                continue
            
            # 对每个 (motif, p_thr) 做平均（因为一个 motif 可能有多个 benchmark）
            df_agg = df_model.groupby(['motif', 'p_thr']).agg({
                metric_name: 'mean'
            }).reset_index()
            
            # 创建图
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 为每个选中的 motif 画一条线
            for motif in selected_motifs:
                df_motif = df_agg[df_agg['motif'] == motif].sort_values('p_thr')
                
                if len(df_motif) == 0:
                    continue
                
                # 美化 motif 名称（用于图例）
                motif_label = motif.replace('_', ' ').title()
                
                ax.plot(df_motif['p_thr'],
                       df_motif[metric_name],
                       marker='o',
                       linewidth=LINE_W,
                       markersize=MARK_SZ,
                       label=motif_label,
                       color=MOTIF_COLORS[motif],
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
            
            # 保存（文件名中包含 difficulty）
            diff_suffix = f"_{difficulty_filter}" if difficulty_filter else "_all_diff"
            output_path = FIGURE_DIR / f"{filename_prefix}{diff_suffix}_{ex}_{ds}.pdf"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            generated_count += 1
    
    return generated_count


# ==================== 高级功能：Motif × Difficulty 对比 ====================

def plot_motif_by_difficulty(
        selected_motif='dense_linear_algebra',
        metric_name='Fast_at_k',
        ylabel='Fast@3',
        title_prefix='Fast@3',
        filename_prefix='figure_fast_at_k_motif_diff_comparison',
        model='claude',
        threads_filter=16):
    """
    为一个 motif，画不同 difficulty 的对比图（4条线）
    
    Args:
        selected_motif: 要分析的 motif
        metric_name: 指标列名
        ylabel: Y轴标签
        title_prefix: 标题前缀
        filename_prefix: 文件名前缀
        model: 模型名称
        threads_filter: EX2 使用的线程数
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
            # 读取 Table D
            table_d_file = TABLE_DIR / f"table_d_per_benchmark_k3_{ex}_{ds}.csv"
            
            if not table_d_file.exists():
                continue
            
            df_d = pd.read_csv(table_d_file)
            
            # 筛选条件
            if ex == 'EX2':
                df_d = df_d[df_d['threads'] == float(threads_filter)]
            
            df_model = df_d[(df_d['model'] == model) & (df_d['motif'] == selected_motif)]
            
            if len(df_model) == 0:
                continue
            
            # 按 (difficulty, p_thr) 聚合
            df_agg = df_model.groupby(['difficulty', 'p_thr']).agg({
                metric_name: 'mean'
            }).reset_index()
            
            # 创建图
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 为每个 difficulty 画一条线
            for difficulty in ['d1', 'd2', 'd3', 'd4']:
                df_diff = df_agg[df_agg['difficulty'] == difficulty].sort_values('p_thr')
                
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
            output_path = FIGURE_DIR / f"{filename_prefix}_{selected_motif}_{ex}_{ds}.pdf"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            generated_count += 1
    
    return generated_count


# ==================== 主函数 ====================

def main():
    """主函数：生成 Motif 分层图"""
    
    parser = argparse.ArgumentParser(description='生成 HPC-Bench Motif 分层图')
    
    # 基本参数
    parser.add_argument('--model', type=str, default='claude',
                       choices=['claude', 'gpt5.1', 'qwen'],
                       help='选择模型 (默认: claude)')
    parser.add_argument('--threads', type=int, default=16,
                       help='EX2 使用的线程数 (默认: 16)')
    
    # Motif 筛选
    parser.add_argument('--motifs', type=str, nargs='+', default=None,
                       help='选择要画的 motif (默认: 5个代表性 motif)')
    
    # Difficulty 筛选
    parser.add_argument('--difficulty', type=str, default=None,
                       choices=['d1', 'd2', 'd3', 'd4'],
                       help='只画特定 difficulty (默认: 全部)')
    
    # 指标选择
    parser.add_argument('--metric', type=str, default='Fast_at_k',
                       choices=['Fast_at_k', 'Speedup_at_k'],
                       help='选择指标 (默认: Fast_at_k)')
    
    # 额外功能
    parser.add_argument('--comparison-motif', type=str, default=None,
                       help='为特定 motif 生成 Difficulty 对比图')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("HPC-Bench Motif 分层图生成")
    print("="*80)
    print(f"输出目录: {FIGURE_DIR}")
    print(f"模型: {args.model}")
    print(f"EX2 线程数: {args.threads}")
    print(f"指标: {args.metric}")
    print(f"Difficulty 筛选: {args.difficulty if args.difficulty else '全部'}")
    print(f"Motifs: {args.motifs if args.motifs else DEFAULT_MOTIFS}")
    print()
    
    # 准备参数
    ylabel = 'Fast@3' if args.metric == 'Fast_at_k' else 'Speedup@3'
    title_prefix = 'Fast@3' if args.metric == 'Fast_at_k' else 'Speedup@3'
    filename_prefix = 'figure_fast_at_k_by_motif' if args.metric == 'Fast_at_k' else 'figure_speedup_at_k_by_motif'
    
    # 生成 Motif 分层图
    print("="*80)
    print("生成 Motif 分层图")
    print("="*80)
    
    count = plot_by_motif(
        metric_name=args.metric,
        ylabel=ylabel,
        title_prefix=title_prefix,
        filename_prefix=filename_prefix,
        selected_motifs=args.motifs,
        difficulty_filter=args.difficulty,
        model=args.model,
        threads_filter=args.threads
    )
    
    print(f"✓ 生成了 {count} 张图片")
    print()
    
    # 如果指定了 comparison_motif，生成对比图
    if args.comparison_motif:
        print("="*80)
        print(f"生成 '{args.comparison_motif}' 的 Difficulty 对比图")
        print("="*80)
        
        comparison_filename = 'figure_fast_at_k_motif_diff_comparison' if args.metric == 'Fast_at_k' else 'figure_speedup_at_k_motif_diff_comparison'
        
        count2 = plot_motif_by_difficulty(
            selected_motif=args.comparison_motif,
            metric_name=args.metric,
            ylabel=ylabel,
            title_prefix=title_prefix,
            filename_prefix=comparison_filename,
            model=args.model,
            threads_filter=args.threads
        )
        
        print(f"✓ 生成了 {count2} 张图片")
        print()
    
    print("="*80)
    print("✅ Motif 图表生成完成！")
    print("="*80)


if __name__ == "__main__":
    main()
