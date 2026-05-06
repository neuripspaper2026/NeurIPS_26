#!/usr/bin/env python3
"""
HPC-Bench 分析结果可视化脚本

生成论文图表：
- 主文图：Fast@k 和 Speedup@k vs p_thr (Table E)
- 附录图：按 Difficulty 分层 (Table G)
- 附录图：按 Motif 分层 (Table F / Table D)
- 高级图：Motif × Difficulty 对比
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
MODELS = ['claude', 'gpt5.1', 'qwen']

# 颜色方案
MODEL_COLORS = {
    'claude': '#1f77b4',    # 蓝色
    'gpt5.1': '#ff7f0e',    # 橙色
    'qwen': '#2ca02c'       # 绿色
}

MODEL_LABELS = {
    'claude': 'Claude',
    'gpt5.1': 'GPT-5.1',
    'qwen': 'Qwen'
}

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


# ==================== 主文图：Fast@k / Speedup@k vs p_thr ====================

def plot_metric_vs_p_thr(metric_name='Fast_at_k_avg', 
                          ylabel='Fast@3',
                          title_prefix='Fast@3',
                          filename='figure_fast_at_k_all_datasets.pdf',
                          threads_filter=16):
    """
    画 metric vs p_thr 的图，2行5列布局 (Table E)
    
    Args:
        metric_name: 要画的指标列名 ('Fast_at_k_avg' 或 'Speedup_at_k_avg')
        ylabel: Y轴标签
        title_prefix: 图表标题前缀
        filename: 保存的文件名
        threads_filter: EX2 使用的线程数 (默认 16)
    """
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(f'{title_prefix} 随 p_thr 的变化 (k=3)', fontsize=16, y=0.995)
    
    # 遍历所有 EX × dataset_size 组合
    for row_idx, ex in enumerate(EX_VERSIONS):
        for col_idx, ds in enumerate(DATASET_SIZES):
            ax = axes[row_idx, col_idx]
            
            # 读取对应的 Table E 文件
            table_e_file = TABLE_DIR / f"table_e_paper_level_k3_{ex}_{ds}.csv"
            
            if not table_e_file.exists():
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                ax.set_title(f'{ds}\n({ex})')
                continue
            
            df_e = pd.read_csv(table_e_file)
            
            # 对于 EX2，只保留指定线程数的数据
            if ex == 'EX2':
                df_e = df_e[df_e['threads'] == float(threads_filter)]
            
            # 为每个 model 画一条线
            for model in MODELS:
                df_model = df_e[df_e['model'] == model].sort_values('p_thr')
                
                if len(df_model) == 0:
                    continue
                
                ax.plot(df_model['p_thr'], 
                       df_model[metric_name],
                       marker='o',
                       linewidth=2,
                       markersize=4,
                       label=MODEL_LABELS[model],
                       color=MODEL_COLORS[model],
                       alpha=0.8)
            
            # 设置子图标题和标签
            ax.set_title(f'{ds}\n({ex})', fontsize=11)
            ax.set_xlabel('p_thr', fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # 只在第一个子图显示图例
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc='best', framealpha=0.9)
            
            # 设置 Y 轴范围
            if metric_name == 'Fast_at_k_avg':
                ax.set_ylim(-0.05, 1.05)
            else:
                ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    # 保存图片
    output_path = FIGURE_DIR / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 已保存: {output_path}")
    
    plt.close()


# ==================== 附录图：按 Difficulty 分层 ====================

def plot_by_difficulty(metric_name='Fast_at_k_avg',
                       ylabel='Fast@3',
                       title_prefix='Fast@3',
                       filename_prefix='figure_fast_at_k_by_difficulty',
                       model='claude',
                       threads_filter=16):
    """
    按 difficulty 分层画图，使用 Table G
    每个 (EX, dataset_size) 一张图，4条线 (d1, d2, d3, d4)
    
    Args:
        metric_name: 指标列名
        ylabel: Y轴标签
        title_prefix: 标题前缀
        filename_prefix: 文件名前缀
        model: 选择的模型
        threads_filter: EX2 使用的线程数
    """
    
    for ex in EX_VERSIONS:
        for ds in DATASET_SIZES:
            # 读取 Table G
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
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # 为每个 difficulty 画一条线
            for difficulty in ['d1', 'd2', 'd3', 'd4']:
                df_diff = df_model[df_model['difficulty'] == difficulty].sort_values('p_thr')
                
                if len(df_diff) == 0:
                    continue
                
                ax.plot(df_diff['p_thr'],
                       df_diff[metric_name],
                       marker='o',
                       linewidth=2,
                       markersize=6,
                       label=DIFFICULTY_LABELS[difficulty],
                       color=DIFFICULTY_COLORS[difficulty],
                       alpha=0.8)
            
            ax.set_title(f'{title_prefix} by Difficulty ({ex}, {ds}, {MODEL_LABELS[model]})', fontsize=14)
            ax.set_xlabel('p_thr', fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', framealpha=0.9)
            
            if metric_name == 'Fast_at_k_avg':
                ax.set_ylim(-0.05, 1.05)
            else:
                ax.set_ylim(bottom=0)
            
            plt.tight_layout()
            
            # 保存
            output_path = FIGURE_DIR / f"{filename_prefix}_{ex}_{ds}.pdf"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ {output_path}")
            plt.close()


# ==================== 附录图：按 Motif 分层（支持 Difficulty 筛选）====================

def plot_by_motif_with_difficulty_filter(
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
        threads_filter: EX2 使用的线程数
    """
    
    # 如果没有指定，使用默认 motif
    if selected_motifs is None:
        selected_motifs = DEFAULT_MOTIFS
    
    # 为每个 motif 分配颜色
    motif_colors = plt.cm.tab10(range(len(selected_motifs)))
    MOTIF_COLORS = dict(zip(selected_motifs, motif_colors))
    
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
                       linewidth=2,
                       markersize=5,
                       label=motif_label,
                       color=MOTIF_COLORS[motif],
                       alpha=0.8)
            
            # 标题中包含 difficulty 信息
            diff_str = f", {difficulty_filter}" if difficulty_filter else ", All Difficulties"
            ax.set_title(f'{title_prefix} by Motif ({ex}, {ds}, {MODEL_LABELS[model]}{diff_str})', 
                        fontsize=14)
            ax.set_xlabel('p_thr', fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', framealpha=0.9, fontsize=9)
            
            if 'Fast' in metric_name:
                ax.set_ylim(-0.05, 1.05)
            else:
                ax.set_ylim(bottom=0)
            
            plt.tight_layout()
            
            # 保存（文件名中包含 difficulty）
            diff_suffix = f"_{difficulty_filter}" if difficulty_filter else "_all_diff"
            output_path = FIGURE_DIR / f"{filename_prefix}{diff_suffix}_{ex}_{ds}.pdf"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ {output_path}")
            plt.close()


# ==================== 高级图：Motif × Difficulty 对比 ====================

def plot_motif_by_difficulty_comparison(
                  metric_name='Fast_at_k',
                  ylabel='Fast@3',
                  title_prefix='Fast@3',
                  filename_prefix='figure_fast_at_k_motif_diff_comparison',
                  selected_motif='dense_linear_algebra',
                  model='claude',
                  threads_filter=16):
    """
    为一个 motif，画不同 difficulty 的对比图
    
    Args:
        metric_name: 指标列名
        ylabel: Y轴标签
        title_prefix: 标题前缀
        filename_prefix: 文件名前缀
        selected_motif: 要分析的 motif
        model: 模型名称
        threads_filter: EX2 使用的线程数
    """
    
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
                       linewidth=2,
                       markersize=6,
                       label=DIFFICULTY_LABELS[difficulty],
                       color=DIFFICULTY_COLORS[difficulty],
                       alpha=0.8)
            
            motif_label = selected_motif.replace('_', ' ').title()
            ax.set_title(f'{title_prefix} for {motif_label} by Difficulty ({ex}, {ds}, {MODEL_LABELS[model]})', 
                        fontsize=14)
            ax.set_xlabel('p_thr', fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', framealpha=0.9)
            
            if 'Fast' in metric_name:
                ax.set_ylim(-0.05, 1.05)
            else:
                ax.set_ylim(bottom=0)
            
            plt.tight_layout()
            
            # 保存
            output_path = FIGURE_DIR / f"{filename_prefix}_{selected_motif}_{ex}_{ds}.pdf"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ {output_path}")
            plt.close()


# ==================== 主函数 ====================

def main():
    """主函数：生成所有图表"""
    
    parser = argparse.ArgumentParser(description='生成 HPC-Bench 分析图表')
    parser.add_argument('--threads', type=int, default=16,
                       help='EX2 使用的线程数 (默认: 16)')
    parser.add_argument('--model', type=str, default='claude',
                       choices=['claude', 'gpt5.1', 'qwen'],
                       help='附录图使用的模型 (默认: claude)')
    parser.add_argument('--skip-main', action='store_true',
                       help='跳过主文图生成')
    parser.add_argument('--skip-difficulty', action='store_true',
                       help='跳过 Difficulty 分层图生成')
    parser.add_argument('--skip-motif', action='store_true',
                       help='跳过 Motif 分层图生成')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("HPC-Bench 图表生成")
    print("="*80)
    print(f"输出目录: {FIGURE_DIR}")
    print(f"EX2 线程数: {args.threads}")
    print(f"附录图模型: {args.model}")
    print()
    
    # 1. 生成主文图（Table E）
    if not args.skip_main:
        print("="*80)
        print("1. 生成主文图: Fast@3 和 Speedup@3 vs p_thr")
        print("="*80)
        
        plot_metric_vs_p_thr(
            metric_name='Fast_at_k_avg',
            ylabel='Fast@3',
            title_prefix='Fast@3',
            filename='figure_fast_at_k_all_datasets.pdf',
            threads_filter=args.threads
        )
        
        plot_metric_vs_p_thr(
            metric_name='Speedup_at_k_avg',
            ylabel='Speedup@3',
            title_prefix='Speedup@3',
            filename='figure_speedup_at_k_all_datasets.pdf',
            threads_filter=args.threads
        )
        print()
    
    # 2. 生成 Difficulty 分层图（Table G）
    if not args.skip_difficulty:
        print("="*80)
        print("2. 生成附录图: 按 Difficulty 分层")
        print("="*80)
        
        plot_by_difficulty(
            metric_name='Fast_at_k_avg',
            ylabel='Fast@3',
            title_prefix='Fast@3',
            filename_prefix='figure_fast_at_k_by_difficulty',
            model=args.model,
            threads_filter=args.threads
        )
        
        plot_by_difficulty(
            metric_name='Speedup_at_k_avg',
            ylabel='Speedup@3',
            title_prefix='Speedup@3',
            filename_prefix='figure_speedup_at_k_by_difficulty',
            model=args.model,
            threads_filter=args.threads
        )
        print()
    
    # 3. 生成 Motif 分层图（Table D）
    if not args.skip_motif:
        print("="*80)
        print("3. 生成附录图: 按 Motif 分层")
        print("="*80)
        
        # 为每个 difficulty 生成一组图
        for difficulty in ['d1', 'd2', 'd3', 'd4', None]:
            diff_label = difficulty if difficulty else "all"
            print(f"\n  Difficulty = {diff_label}:")
            
            plot_by_motif_with_difficulty_filter(
                metric_name='Fast_at_k',
                ylabel='Fast@3',
                title_prefix='Fast@3',
                filename_prefix='figure_fast_at_k_by_motif',
                difficulty_filter=difficulty,
                model=args.model,
                threads_filter=args.threads
            )
        print()
    
    print("="*80)
    print("✅ 所有图表生成完成！")
    print("="*80)
    print(f"\n图片保存位置: {FIGURE_DIR}/")
    print("\n生成的图片:")
    print("  - figure_fast_at_k_all_datasets.pdf (主文)")
    print("  - figure_speedup_at_k_all_datasets.pdf (主文)")
    print("  - figure_*_by_difficulty_*.pdf (附录 - Difficulty 分层)")
    print("  - figure_*_by_motif_*.pdf (附录 - Motif 分层)")


if __name__ == "__main__":
    main()
