#!/usr/bin/env python3
"""EX3 Fast@k 和 Speedup@k 可视化 - 简化版"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = PROJECT_ROOT / "analysis_summaries" / "figures" / "EX3"
DATA_DIR = PROJECT_ROOT / "analysis_summaries" / "unified_data"

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11

DATASETS = ['mini', 'small', 'medium', 'large', 'extra-large']
MODELS = ['claude', 'gpt5.1', 'qwen']
COLORS = {'claude': '#1f77b4', 'gpt5.1': '#ff7f0e', 'qwen': '#2ca02c'}
LABELS = {'claude': 'Claude', 'gpt5.1': 'GPT-5.1', 'qwen': 'Qwen'}

def plot_fast_at_k(k=3):
    df = pd.read_csv(DATA_DIR / f"fast_speedup_at_k{k}_EX3.csv")
    df_agg = df.groupby(["model", "dataset"])[f"fast_at_{k}"].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(DATASETS))
    for i, m in enumerate(MODELS):
        vals = [df_agg[(df_agg.model==m)&(df_agg.dataset==d)][f"fast_at_{k}"].values[0] 
                if len(df_agg[(df_agg.model==m)&(df_agg.dataset==d)])>0 else 0 for d in DATASETS]
        ax.bar(x+i*0.25, vals, 0.25, label=LABELS[m], color=COLORS[m], alpha=0.8)
    
    ax.set_xlabel('Dataset'); ax.set_ylabel(f'Fast@{k}')
    ax.set_title(f'Fast@{k} by Dataset (EX3)')
    ax.set_xticks(x+0.25); ax.set_xticklabels(DATASETS, rotation=45)
    ax.legend(); ax.set_ylim(0,1.05); ax.grid(alpha=0.3)
    plt.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_DIR/f"fast_at_{k}_EX3.pdf", dpi=300)
    print(f"✓ Saved: fast_at_{k}_EX3.pdf")
    plt.close()

def plot_speedup_at_k(k=3):
    df = pd.read_csv(DATA_DIR / f"fast_speedup_at_k{k}_EX3.csv")
    df_agg = df.groupby(["model","dataset"])[f"speedup_at_{k}"].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(12,6))
    x = np.arange(len(DATASETS))
    for i, m in enumerate(MODELS):
        vals = [df_agg[(df_agg.model==m)&(df_agg.dataset==d)][f"speedup_at_{k}"].values[0]
                if len(df_agg[(df_agg.model==m)&(df_agg.dataset==d)])>0 else 1.0 for d in DATASETS]
        ax.bar(x+i*0.25, vals, 0.25, label=LABELS[m], color=COLORS[m], alpha=0.8)
    
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Dataset'); ax.set_ylabel(f'Speedup@{k}')
    ax.set_title(f'Speedup@{k} by Dataset (EX3)')
    ax.set_xticks(x+0.25); ax.set_xticklabels(DATASETS, rotation=45)
    ax.legend(); ax.set_ylim(bottom=0); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR/f"speedup_at_{k}_EX3.pdf", dpi=300)
    print(f"✓ Saved: speedup_at_{k}_EX3.pdf")
    plt.close()

if __name__ == "__main__":
    # 只画 k=3
    k = 3
    print(f"\nPlotting Fast@{k} and Speedup@{k} for EX3...")
    plot_fast_at_k(k)
    plot_speedup_at_k(k)
    print("\n✓ Done!")
