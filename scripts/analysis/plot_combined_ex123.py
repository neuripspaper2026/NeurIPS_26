#!/usr/bin/env python3
"""
生成 EX1/EX2/EX3 组合的 Fast@3 和 Speedup@3 vs p_thr 图表（3行5列）

Usage:
    python scripts/analysis/plot_combined_ex123.py
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# ==================== 配置 ====================

# 设置画图风格
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 10

# 路径
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = WORKSPACE_ROOT / "analysis_summaries" / "tables"
DATA_DIR = WORKSPACE_ROOT / "analysis_summaries" / "unified_data"
FIGURE_DIR = WORKSPACE_ROOT / "analysis_summaries" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# 实验配置
EX_VERSIONS = ['EX1', 'EX2', 'EX3']
DATASET_SIZES = ['mini', 'small', 'medium', 'large', 'extra-large']
MODELS = ['claude', 'gpt5.1', 'qwen']

# 颜色和标签
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


# ==================== 数据加载 ====================

def load_ex1_ex2_data(metric_name='Fast_at_k_avg'):
    """加载 EX1 和 EX2 的数据（从现有的 table_e 文件）"""
    data = {}
    
    for ex in ['EX1', 'EX2']:
        data[ex] = {}
        for ds in DATASET_SIZES:
            table_file = TABLE_DIR / f"table_e_paper_level_k3_{ex}_{ds}.csv"
            
            if not table_file.exists():
                print(f"  [WARN] File not found: {table_file}")
                data[ex][ds] = pd.DataFrame()
                continue
            
            df = pd.read_csv(table_file)
            
            # EX2: 只保留 threads=16 的数据
            if ex == 'EX2':
                df = df[df['threads'] == 16.0]
            
            data[ex][ds] = df
    
    return data


def load_ex3_data():
    """加载 EX3 的数据"""
    data_file = DATA_DIR / "fast_speedup_at_k3_with_pthr_EX3.csv"
    
    if not data_file.exists():
        print(f"  [ERROR] EX3 data file not found: {data_file}")
        return {}
    
    df = pd.read_csv(data_file)
    
    # 按 benchmark 聚合（macro-average）
    df_agg = df.groupby(['model', 'dataset', 'p_thr']).agg({
        'fast_at_3': 'mean',
        'speedup_at_3': 'mean'
    }).reset_index()
    
    # 按 dataset 分组
    data = {}
    for ds in DATASET_SIZES:
        data[ds] = df_agg[df_agg['dataset'] == ds]
    
    return data


# ==================== 绘图函数 ====================

def plot_combined_fast_at_k(ex1_ex2_data, ex3_data, k=3):
    """
    生成 Fast@k vs p_thr 的 3行5列组合图
    
    Args:
        ex1_ex2_data: EX1/EX2 的数据字典
        ex3_data: EX3 的数据字典
        k: k 值（默认 3）
    """
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    
    for row_idx, ex in enumerate(EX_VERSIONS):
        for col_idx, ds in enumerate(DATASET_SIZES):
            ax = axes[row_idx, col_idx]
            
            if ex in ['EX1', 'EX2']:
                # EX1/EX2: 从 table_e 读取
                if ds not in ex1_ex2_data[ex] or len(ex1_ex2_data[ex][ds]) == 0:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{ds}\n({ex})')
                    continue
                
                df_e = ex1_ex2_data[ex][ds]
                
                # 为每个 model 画一条线
                for model in MODELS:
                    df_model = df_e[df_e['model'] == model].sort_values('p_thr')
                    
                    if len(df_model) == 0:
                        continue
                    
                    ax.plot(
                        df_model['p_thr'],
                        df_model['Fast_at_k_avg'],
                        marker='o',
                        linewidth=2,
                        markersize=4,
                        color=MODEL_COLORS[model],
                        alpha=0.8
                    )
            
            else:  # EX3
                if ds not in ex3_data or len(ex3_data[ds]) == 0:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{ds}\n({ex})')
                    continue
                
                df_ex3 = ex3_data[ds]
                
                # 为每个 model 画一条线
                for model in MODELS:
                    df_model = df_ex3[df_ex3['model'] == model].sort_values('p_thr')
                    
                    if len(df_model) == 0:
                        continue
                    
                    ax.plot(
                        df_model['p_thr'],
                        df_model['fast_at_3'],
                        marker='o',
                        linewidth=2,
                        markersize=4,
                        color=MODEL_COLORS[model],
                        alpha=0.8
                    )
            
            # 设置子图标题和标签
            ax.set_title(f'{ds}\n({ex})', fontsize=11)
            ax.set_xlabel('$p_{thr}$', fontsize=10)
            ax.set_ylabel(f'Fast@{k}', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.05)
    
    # 在图的底部添加统一的图例
    handles = [
        plt.Line2D([0], [0], color=MODEL_COLORS[m], linewidth=2, marker='o', 
                   markersize=6, label=MODEL_LABELS[m]) 
        for m in MODELS
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, 
               frameon=True, fontsize=11, bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    
    # 保存
    output_path = FIGURE_DIR / f"figure_fast_at_{k}_all_ex_versions.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_combined_speedup_at_k(ex1_ex2_data, ex3_data, k=3):
    """
    生成 Speedup@k vs p_thr 的 3行5列组合图
    
    Args:
        ex1_ex2_data: EX1/EX2 的数据字典
        ex3_data: EX3 的数据字典
        k: k 值（默认 3）
    """
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    
    for row_idx, ex in enumerate(EX_VERSIONS):
        for col_idx, ds in enumerate(DATASET_SIZES):
            ax = axes[row_idx, col_idx]
            
            if ex in ['EX1', 'EX2']:
                # EX1/EX2: 从 table_e 读取
                if ds not in ex1_ex2_data[ex] or len(ex1_ex2_data[ex][ds]) == 0:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{ds}\n({ex})')
                    continue
                
                df_e = ex1_ex2_data[ex][ds]
                
                # 为每个 model 画一条线
                for model in MODELS:
                    df_model = df_e[df_e['model'] == model].sort_values('p_thr')
                    
                    if len(df_model) == 0:
                        continue
                    
                    ax.plot(
                        df_model['p_thr'],
                        df_model['Speedup_at_k_avg'],
                        marker='o',
                        linewidth=2,
                        markersize=4,
                        color=MODEL_COLORS[model],
                        alpha=0.8
                    )
            
            else:  # EX3
                if ds not in ex3_data or len(ex3_data[ds]) == 0:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{ds}\n({ex})')
                    continue
                
                df_ex3 = ex3_data[ds]
                
                # 为每个 model 画一条线
                for model in MODELS:
                    df_model = df_ex3[df_ex3['model'] == model].sort_values('p_thr')
                    
                    if len(df_model) == 0:
                        continue
                    
                    ax.plot(
                        df_model['p_thr'],
                        df_model['speedup_at_3'],
                        marker='o',
                        linewidth=2,
                        markersize=4,
                        color=MODEL_COLORS[model],
                        alpha=0.8
                    )
            
            # 设置子图标题和标签
            ax.set_title(f'{ds}\n({ex})', fontsize=11)
            ax.set_xlabel('$p_{thr}$', fontsize=10)
            ax.set_ylabel(f'Speedup@{k}', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)
    
    # 在图的底部添加统一的图例
    handles = [
        plt.Line2D([0], [0], color=MODEL_COLORS[m], linewidth=2, marker='o', 
                   markersize=6, label=MODEL_LABELS[m]) 
        for m in MODELS
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, 
               frameon=True, fontsize=11, bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    
    # 保存
    output_path = FIGURE_DIR / f"figure_speedup_at_{k}_all_ex_versions.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


# ==================== 主函数 ====================

def main():
    print("\n" + "="*80)
    print("Generating Combined EX1/EX2/EX3 Figures")
    print("="*80 + "\n")
    
    # 加载数据
    print("Loading data...")
    ex1_ex2_data = load_ex1_ex2_data()
    ex3_data = load_ex3_data()
    print("✓ Data loaded\n")
    
    # 生成图表
    print("Generating Fast@3 figure (3 rows × 5 columns)...")
    plot_combined_fast_at_k(ex1_ex2_data, ex3_data, k=3)
    
    print("\nGenerating Speedup@3 figure (3 rows × 5 columns)...")
    plot_combined_speedup_at_k(ex1_ex2_data, ex3_data, k=3)
    
    print("\n" + "="*80)
    print("✓ All combined figures generated successfully!")
    print("="*80)
    print("\nOutput files:")
    print(f"  - {FIGURE_DIR}/figure_fast_at_3_all_ex_versions.pdf")
    print(f"  - {FIGURE_DIR}/figure_speedup_at_3_all_ex_versions.pdf")
    print()


if __name__ == "__main__":
    main()
