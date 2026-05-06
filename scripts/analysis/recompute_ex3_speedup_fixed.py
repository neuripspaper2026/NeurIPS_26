#!/usr/bin/env python3
"""
修正版 EX3 Speedup@k 计算 — 与 EX1/EX2 pipeline 对齐
====================================================

问题: 原始 EX3 pipeline 中 Speedup@k 在无候选通过 p_thr 时 fallback=1.0,
      但论文公式和 EX1/EX2 pipeline 使用 fallback=0.

修正: 使用与 EX1/EX2 完全相同的穷举组合方法 (compute_speedup_at_k_exact),
      incorrectness-gated speedup=0, 当 max < p_thr 时贡献 0.

输出文件（均带 _fixed 后缀，不覆盖原始文件）:
  - analysis_summaries/unified_data/fast_speedup_at_k3_with_pthr_EX3_fixed.csv
  - analysis_summaries/beautiful_figure/main_paper/figure_speedup_at_3_beautiful_fixed.pdf
  - analysis_summaries/beautiful_figure/main_paper/figure_fast_at_3_beautiful_fixed.pdf

Usage:
    python scripts/analysis/recompute_ex3_speedup_fixed.py
"""
from __future__ import annotations

import sys
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

# --------------- paths ---------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

UNIFIED_CSV   = PROJECT_ROOT / "analysis_summaries" / "unified_data" / "unified_df_EX3.csv"
TABLES_DIR    = PROJECT_ROOT / "analysis_summaries" / "tables"
UNIFIED_DIR   = PROJECT_ROOT / "analysis_summaries" / "unified_data"
FIGURE_DIR    = PROJECT_ROOT / "analysis_summaries" / "beautiful_figure" / "main_paper"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
UNIFIED_DIR.mkdir(parents=True, exist_ok=True)

# --------------- config ---------------
DATASET_SIZES = ["mini", "small", "medium", "large", "extra-large"]
MODELS        = ["claude", "gpt5.1", "qwen"]
P_THR_LIST    = [0.0, 1.0, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.40, 1.50, 2.0, 3.0]
K             = 3

MODEL_COLORS  = {"claude": "#2E86AB", "gpt5.1": "#A23B72", "qwen": "#F18F01"}
MODEL_LABELS  = {"claude": "Claude", "gpt5.1": "GPT-5.1", "qwen": "Qwen"}
DATASET_LABELS = {
    "mini": "Mini", "small": "Small", "medium": "Medium",
    "large": "Large", "extra-large": "Extra-Large",
}


# =====================================================================
# 核心计算函数 — 与 EX1/EX2 analysis.ipynb Cell 20 完全一致
# =====================================================================

def compute_fast_at_k(c_p: int, N: int, k: int) -> float:
    """Fast@k 闭式公式: 1 - C(N-c_p, k) / C(N, k)"""
    if N <= 0 or k <= 0 or k > N:
        return np.nan
    if c_p >= k:
        return 1.0
    if c_p == 0:
        return 0.0
    if N - c_p < k:
        return 1.0
    return 1.0 - comb(N - c_p, k) / comb(N, k)


def compute_speedup_at_k_exact(speedups_gated: list, p_thr: float, N: int, k: int) -> float:
    """
    穷举组合精确计算 Speedup@k(p).

    Speedup@k(p) = E[ max(subsample) · 1{max >= p_thr} ]

    当 max < p_thr 时贡献 0（不是 1.0）。
    """
    if len(speedups_gated) != N or N < k:
        return 0.0

    total = 0.0
    n_combos = 0
    for combo in combinations(range(N), k):
        max_s = max(speedups_gated[i] for i in combo)
        if max_s >= p_thr:
            total += max_s
        # else: 贡献 0
        n_combos += 1

    return total / n_combos if n_combos > 0 else 0.0


# =====================================================================
# Step 1: 从 unified_df_EX3.csv 重新计算 per-benchmark metrics
# =====================================================================

def recompute_ex3_metrics(df_raw: pd.DataFrame) -> pd.DataFrame:
    """对每个 (benchmark, model, dataset) 重新计算 Fast@k 和 Speedup@k"""

    rows = []
    groups = df_raw.groupby(["benchmark", "model", "dataset"])
    total_groups = len(groups)

    for idx, ((bm, model, ds), grp) in enumerate(groups):
        if model == "baseline":
            continue

        # 构建 correctness-gated speedup 列表（与 EX1/EX2 一致）
        # 不正确 → 0, 正确但无 speedup → 0
        speedups_gated = []
        for _, row in grp.iterrows():
            if row.get("is_baseline", False):
                continue
            if row.get("is_correct", False) and pd.notna(row.get("kernel_speedup")):
                speedups_gated.append(float(row["kernel_speedup"]))
            else:
                speedups_gated.append(0.0)

        N = len(speedups_gated)
        if N == 0:
            continue

        num_correct = sum(1 for s in speedups_gated if s > 0)

        for p_thr in P_THR_LIST:
            c_p = sum(1 for s in speedups_gated if s >= p_thr)
            fast = compute_fast_at_k(c_p, N, K)
            speedup = compute_speedup_at_k_exact(speedups_gated, p_thr, N, K)

            rows.append({
                "benchmark": bm,
                "model": model,
                "dataset": ds,
                "p_thr": p_thr,
                "k": K,
                "fast_at_3": fast,
                "speedup_at_3": speedup,
                "num_correct": num_correct,
                "num_total": N,
            })

        if (idx + 1) % 50 == 0:
            print(f"  进度: {idx + 1}/{total_groups} groups")

    return pd.DataFrame(rows)


# =====================================================================
# Step 2: 绘图 — 复用 plot_combined_ex123_beautiful.py 的风格
# =====================================================================

def _load_ex1_ex2_data():
    data = {}
    for ex in ["EX1", "EX2"]:
        data[ex] = {}
        for ds in DATASET_SIZES:
            f = TABLES_DIR / f"table_e_paper_level_k3_{ex}_{ds}.csv"
            if not f.exists():
                continue
            df = pd.read_csv(f)
            if ex == "EX2":
                df = df[df["threads"] == 16.0]
            data[ex][ds] = df
    return data


def _aggregate_ex3(df_fixed: pd.DataFrame):
    data = {}
    for ds in DATASET_SIZES:
        sub = df_fixed[df_fixed["dataset"] == ds]
        if len(sub) == 0:
            continue
        agg = sub.groupby(["model", "p_thr"]).agg(
            fast_at_3=("fast_at_3", "mean"),
            speedup_at_3=("speedup_at_3", "mean"),
        ).reset_index()
        data[ds] = agg
    return data


def plot_speedup_figure(ex12_data, ex3_data, filename):
    """绘制 Speedup@3 — 与 beautiful_fig.ipynb 完全一致的布局和风格"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    EX_VERSIONS = ["EX1", "EX2", "EX3"]
    fig, axes = plt.subplots(3, 5, figsize=(15, 8), dpi=150)

    plt.subplots_adjust(
        left=0.065, right=0.985,
        top=0.9, bottom=0.10,
        hspace=0.15, wspace=0.15,
    )

    for row_idx, ex in enumerate(EX_VERSIONS):
        for col_idx, ds in enumerate(DATASET_SIZES):
            ax = axes[row_idx, col_idx]

            if ex in ["EX1", "EX2"]:
                if ds in ex12_data[ex]:
                    df_e = ex12_data[ex][ds]
                    for model in MODELS:
                        df_model = df_e[df_e["model"] == model]
                        if len(df_model) > 0:
                            ax.plot(
                                df_model["p_thr"],
                                df_model["Speedup_at_k_avg"],
                                color=MODEL_COLORS[model],
                                linewidth=2,
                                marker=None,
                                markersize=0,
                                alpha=0.85,
                            )
            elif ex == "EX3":
                if ds in ex3_data:
                    df_ex3 = ex3_data[ds]
                    for model in MODELS:
                        df_model = df_ex3[df_ex3["model"] == model].sort_values("p_thr")
                        if len(df_model) > 0:
                            ax.plot(
                                df_model["p_thr"],
                                df_model["speedup_at_3"],
                                color=MODEL_COLORS[model],
                                linewidth=2,
                                marker=None,
                                markersize=0,
                                alpha=0.85,
                            )

            # 与 notebook 一致：水平网格
            ax.grid(True, axis="y", alpha=0.15, linestyle="-", linewidth=0.4, color="gray")
            ax.set_axisbelow(True)

            # 与 notebook 一致：不设 ylim，完全自动缩放
            if col_idx == 0:
                ax.set_ylabel("", fontsize=10)
            else:
                ax.set_yticklabels([])
            if row_idx == 2:
                ax.set_xlabel("", fontsize=10)
            else:
                ax.set_xticklabels([])
            if row_idx == 0:
                ax.set_title(DATASET_LABELS[ds], fontsize=11, fontweight="normal", pad=15)
            ax.tick_params(axis="both", labelsize=8)

    # 行标签
    row_labels = ["EX1 (Serial)", "EX2 (OpenMP)", "EX3 (CUDA)"]
    row_positions = [0.77, 0.50, 0.23]
    for i, label in enumerate(row_labels):
        fig.text(0.02, row_positions[i], label, fontsize=12, fontweight="bold",
                 ha="center", va="center", rotation=90)

    # 轴标签
    fig.text(0.53, 0.03, r"Speedup threshold $p_{\mathrm{thr}}$",
             ha="center", fontsize=13, fontweight="normal")
    fig.text(0.005, 0.5, r"Speedup@3",
             ha="center", va="center", rotation=90, fontsize=13, fontweight="normal")

    # Legend
    handles = [
        plt.Line2D([0], [0], color=MODEL_COLORS[m], linewidth=2.5,
                   marker="o", markersize=5, label=MODEL_LABELS[m])
        for m in MODELS
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=True,
               fontsize=11, bbox_to_anchor=(0.5, 1.02),
               columnspacing=1.5, handlelength=2.5)

    out = FIGURE_DIR / filename
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  保存: {out}")


def plot_fast_figure(ex12_data, ex3_data, filename):
    """绘制 Fast@3 — 与 beautiful_fig.ipynb 完全一致的布局和风格"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    EX_VERSIONS = ["EX1", "EX2", "EX3"]
    fig, axes = plt.subplots(3, 5, figsize=(15, 8), dpi=150)

    plt.subplots_adjust(
        left=0.055, right=0.985,
        top=0.92, bottom=0.08,
        hspace=0.15, wspace=0.15,
    )

    for row_idx, ex in enumerate(EX_VERSIONS):
        for col_idx, ds in enumerate(DATASET_SIZES):
            ax = axes[row_idx, col_idx]

            if ex in ["EX1", "EX2"]:
                if ds in ex12_data[ex]:
                    df_e = ex12_data[ex][ds]
                    for model in MODELS:
                        df_model = df_e[df_e["model"] == model]
                        if len(df_model) > 0:
                            ax.plot(
                                df_model["p_thr"],
                                df_model["Fast_at_k_avg"],
                                color=MODEL_COLORS[model],
                                linewidth=2,
                                marker=None,
                                markersize=0,
                                alpha=0.85,
                            )
            elif ex == "EX3":
                if ds in ex3_data:
                    df_ex3 = ex3_data[ds]
                    for model in MODELS:
                        df_model = df_ex3[df_ex3["model"] == model].sort_values("p_thr")
                        if len(df_model) > 0:
                            ax.plot(
                                df_model["p_thr"],
                                df_model["fast_at_3"],
                                color=MODEL_COLORS[model],
                                linewidth=2,
                                marker=None,
                                markersize=0,
                                alpha=0.85,
                            )

            ax.grid(True, axis="y", alpha=0.15, linestyle="-", linewidth=0.4, color="gray")
            ax.set_axisbelow(True)
            ax.set_ylim(-0.05, 1.05)

            if col_idx == 0:
                ax.set_ylabel("", fontsize=10)
            else:
                ax.set_yticklabels([])
            if row_idx == 2:
                ax.set_xlabel("", fontsize=10)
            else:
                ax.set_xticklabels([])
            if row_idx == 0:
                ax.set_title(DATASET_LABELS[ds], fontsize=11, fontweight="normal", pad=15)
            ax.tick_params(axis="both", labelsize=8)

    row_labels = ["EX1 (Serial)", "EX2 (OpenMP)", "EX3 (CUDA)"]
    row_positions = [0.77, 0.50, 0.23]
    for i, label in enumerate(row_labels):
        fig.text(0.02, row_positions[i], label, fontsize=12, fontweight="bold",
                 ha="center", va="center", rotation=90)

    fig.text(0.53, 0.03, r"Speedup threshold $p_{\mathrm{thr}}$",
             ha="center", fontsize=13, fontweight="normal")
    fig.text(0.005, 0.5, r"Fast@3",
             ha="center", va="center", rotation=90, fontsize=13, fontweight="normal")

    handles = [
        plt.Line2D([0], [0], color=MODEL_COLORS[m], linewidth=2.5,
                   marker="o", markersize=5, label=MODEL_LABELS[m])
        for m in MODELS
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=True,
               fontsize=11, bbox_to_anchor=(0.5, 1.02),
               columnspacing=1.5, handlelength=2.5)

    out = FIGURE_DIR / filename
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  保存: {out}")


# =====================================================================
# Step 3: 对比报告
# =====================================================================

def print_comparison(df_fixed: pd.DataFrame, df_orig_path: Path):
    if not df_orig_path.exists():
        print("  [WARN] 原始 with_pthr CSV 不存在，跳过对比")
        return

    df_orig = pd.read_csv(df_orig_path)
    # 只比较非 baseline
    df_orig = df_orig[df_orig["model"] != "baseline"]

    print("\n" + "=" * 70)
    print("原始 vs 修正 Speedup@3 对比 (macro-average)")
    print("=" * 70)

    for ds in DATASET_SIZES:
        print(f"\n--- {ds} ---")
        print(f"  {'model':8s}  {'p_thr':>6s}  {'原始':>8s}  {'修正':>8s}  {'差值':>8s}")
        for model in MODELS:
            for pthr in [0.0, 1.0, 1.05, 1.5, 3.0]:
                orig_sub = df_orig[(df_orig["dataset"] == ds) &
                                   (df_orig["model"] == model) &
                                   (df_orig["p_thr"] == pthr)]
                fix_sub = df_fixed[(df_fixed["dataset"] == ds) &
                                    (df_fixed["model"] == model) &
                                    (df_fixed["p_thr"] == pthr)]
                orig_val = orig_sub["speedup_at_3"].mean() if len(orig_sub) > 0 else float("nan")
                fix_val  = fix_sub["speedup_at_3"].mean() if len(fix_sub) > 0 else float("nan")
                diff = orig_val - fix_val
                print(f"  {model:8s}  {pthr:6.2f}  {orig_val:8.4f}  {fix_val:8.4f}  {diff:+8.4f}")


# =====================================================================
# main
# =====================================================================

def main():
    print("=" * 70)
    print("EX3 Speedup@k 修正 (fallback 1.0 → 0, 与 EX1/EX2 对齐)")
    print("=" * 70)

    # 1) 加载原始数据
    print(f"\n[1/4] 加载 {UNIFIED_CSV.name} ...")
    df_raw = pd.read_csv(UNIFIED_CSV)
    print(f"  共 {len(df_raw)} 行, {df_raw['benchmark'].nunique()} 个 benchmark")

    # 2) 重算
    print(f"\n[2/4] 用穷举组合重算 Fast@{K} / Speedup@{K} (fallback=0) ...")
    df_fixed = recompute_ex3_metrics(df_raw)
    out_csv = UNIFIED_DIR / "fast_speedup_at_k3_with_pthr_EX3_fixed.csv"
    df_fixed.to_csv(out_csv, index=False)
    print(f"  保存: {out_csv}  ({len(df_fixed)} 行)")

    # 3) 画图
    print(f"\n[3/4] 生成修正版 Figure ...")
    ex12 = _load_ex1_ex2_data()
    ex3_agg = _aggregate_ex3(df_fixed)

    plot_speedup_figure(ex12, ex3_agg, "figure_speedup_at_3_beautiful_fixed.pdf")
    plot_fast_figure(ex12, ex3_agg, "figure_fast_at_3_beautiful_fixed.pdf")

    # 4) 对比
    print(f"\n[4/4] 原始 vs 修正 对比 ...")
    orig_csv = UNIFIED_DIR / "fast_speedup_at_k3_with_pthr_EX3.csv"
    print_comparison(df_fixed, orig_csv)

    print("\n" + "=" * 70)
    print("完成！输出文件:")
    print(f"  CSV:  {out_csv}")
    print(f"  图表: {FIGURE_DIR}/figure_speedup_at_3_beautiful_fixed.pdf")
    print(f"  图表: {FIGURE_DIR}/figure_fast_at_3_beautiful_fixed.pdf")
    print("=" * 70)


if __name__ == "__main__":
    main()
