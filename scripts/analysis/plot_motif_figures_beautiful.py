#!/usr/bin/env python3
"""
HPC-Bench Motif 分层图可视化脚本 - 美化版

功能：
- 按 Motif 分层画图 (支持 Difficulty 筛选)
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
OUTPUT_DIR = WORKSPACE_ROOT / "analysis_summaries" / "beautiful_figure" / "main_paper" / "motif-level"

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
    'd1': 'Difficulty 1',
    'd2': 'Difficulty 2',
    'd3': 'Difficulty 3',
    'd4': 'Difficulty 4'
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

def plot_by_motif_beautiful(
        metric_name='Fast_at_k',
        ylabel='Fast@3',
        filename_prefix='fast_at_k_by_motif',
        selected_motifs=None,
        difficulty_filter=None,
        model='claude',
        threads_filter=16):
    """按 motif 分层画图（美化版）"""
    
    # 如果没有指定，使用默认 motif
    if selected_motifs is None:
        selected_motifs = DEFAULT_MOTIFS
    
    # 为每个 motif 分配颜色
    motif_colors = plt.cm.tab10(range(len(selected_motifs)))
    MOTIF_COLORS = dict(zip(selected_motifs, motif_colors))
    
    generated_count = 0
    
    for ex in EX_VERSIONS:
        for ds in DATASET_SIZES:
            # 读取 Table D
            table_d_file = TABLE_DIR / f"table_d_per_benchmark_k3_{ex}_{ds}.csv"
            
            if not table_d_file.exists():
                continue
            
            df_d = pd.read_csv(table_d_file)
            
            # EX2 筛选线程数
            if ex == 'EX2':
                df_d = df_d[df_d['threads'] == float(threads_filter)]
            
            # 筛选 model
            df_model = df_d[df_d['model'] == model]
            
            # 筛选 difficulty
            if difficulty_filter is not None:
                df_model = df_model[df_model['difficulty'] == difficulty_filter]
            
            if len(df_model) == 0:
                continue
            
            # 聚合：每个 (motif, p_thr) 做平均
            df_agg = df_model.groupby(['motif', 'p_thr']).agg({
                metric_name: 'mean'
            }).reset_index()
            
            # 创建图
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # 为每个 motif 画线
            for motif in selected_motifs:
                df_motif = df_agg[df_agg['motif'] == motif].sort_values('p_thr')
                
                if len(df_motif) == 0:
                    continue
                
                # 美化 motif 名称
                motif_label = motif.replace('_', ' ').title()
                
                ax.plot(df_motif['p_thr'],
                       df_motif[metric_name],
                       marker=None,  # 不显示 marker
                       linewidth=2.5,
                       label=motif_label,
                       color=MOTIF_COLORS[motif],
                       alpha=0.85)
            
            # 网格（只保留水平网格，且更淡）
            ax.grid(True, axis='y', alpha=0.15, linestyle='-', linewidth=0.4, color='gray')
            ax.set_axisbelow(True)
            
            # 标题和标签
            ex_labels = {'EX1': 'EX1 (Serial)', 'EX2': 'EX2 (OpenMP)'}
            ds_labels = {'mini': 'Mini', 'small': 'Small', 'medium': 'Medium', 
                        'large': 'Large', 'extra-large': 'Extra-Large'}
            
            ax.set_title(f'{ex_labels[ex]} - {ds_labels[ds]}', 
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
            diff_suffix = f"_{difficulty_filter}" if difficulty_filter else "_all_diff"
            output_path = OUTPUT_DIR / f"{filename_prefix}{diff_suffix}_{ex}_{ds}.pdf"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            generated_count += 1
    
    return generated_count


# ==================== 高级功能：Motif × Difficulty 对比 ====================

def plot_motif_by_difficulty_beautiful(
        selected_motif='dense_linear_algebra',
        metric_name='Fast_at_k',
        ylabel='Fast@3',
        filename_prefix='fast_at_k_motif_diff_comparison',
        model='claude',
        threads_filter=16):
    """为一个 motif 画不同 difficulty 的对比图（美化版）"""
    
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
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # 为每个 difficulty 画线
            for difficulty in ['d1', 'd2', 'd3', 'd4']:
                df_diff = df_agg[df_agg['difficulty'] == difficulty].sort_values('p_thr')
                
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
            ds_labels = {'mini': 'Mini', 'small': 'Small', 'medium': 'Medium', 
                        'large': 'Large', 'extra-large': 'Extra-Large'}
            motif_label = selected_motif.replace('_', ' ').title()
            
            ax.set_title(f'{ex_labels[ex]} - {ds_labels[ds]}\n{motif_label}', 
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
            output_path = OUTPUT_DIR / f"{filename_prefix}_{selected_motif}_{ex}_{ds}.pdf"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            generated_count += 1
    
    return generated_count


# ==================== 主函数 ====================

def main():
    """主函数：生成美化版 Motif 分层图"""
    
    parser = argparse.ArgumentParser(description='生成美化版 HPC-Bench Motif 分层图')
    
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
    
    print("="*80)
    print("美化版 Motif 分层图生成")
    print("="*80)
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"模型: {args.model}")
    print(f"EX2 线程数: {args.threads}")
    print(f"指标: {args.metric}")
    print(f"Difficulty 筛选: {args.difficulty if args.difficulty else '全部'}")
    print(f"Motifs: {args.motifs if args.motifs else DEFAULT_MOTIFS}")
    print()
    
    # 准备参数
    ylabel = 'Fast@3' if args.metric == 'Fast_at_k' else 'Speedup@3'
    filename_prefix = 'fast_at_k_by_motif' if args.metric == 'Fast_at_k' else 'speedup_at_k_by_motif'
    
    # 生成 Motif 分层图
    print("="*80)
    print("生成 Motif 分层图")
    print("="*80)
    
    count = plot_by_motif_beautiful(
        metric_name=args.metric,
        ylabel=ylabel,
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
        
        comparison_filename = 'fast_at_k_motif_diff_comparison' if args.metric == 'Fast_at_k' else 'speedup_at_k_motif_diff_comparison'
        
        count2 = plot_motif_by_difficulty_beautiful(
            selected_motif=args.comparison_motif,
            metric_name=args.metric,
            ylabel=ylabel,
            filename_prefix=comparison_filename,
            model=args.model,
            threads_filter=args.threads
        )
        
        print(f"✓ 生成了 {count2} 张图片")
        print()
    
    print("="*80)
    print("✅ 美化版 Motif 图表生成完成！")
    print("="*80)


if __name__ == "__main__":
    main()
