#!/usr/bin/env python3
"""
Variant of recompute_ex3_speedup_fixed.py that reads the April rerun
unified dataframe and emits the new (rerun) Fast@3 / Speedup@3 figures
side-by-side with EX1/EX2 panels.

Outputs:
  - analysis_summaries/unified_data_rerun/fast_speedup_at_k3_with_pthr_EX3_rerun.csv
  - analysis_summaries/beautiful_figure/main_paper/figure_speedup_at_3_beautiful_rerun.pdf
  - analysis_summaries/beautiful_figure/main_paper/figure_fast_at_3_beautiful_rerun.pdf
"""
from __future__ import annotations

import sys
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Read EX3 from the new rerun unified dir; EX1/EX2 inputs are unchanged
UNIFIED_CSV   = PROJECT_ROOT / "analysis_summaries" / "unified_data_rerun" / "unified_df_EX3.csv"
TABLES_DIR    = PROJECT_ROOT / "analysis_summaries" / "tables"
UNIFIED_DIR   = PROJECT_ROOT / "analysis_summaries" / "unified_data_rerun"
FIGURE_DIR    = PROJECT_ROOT / "analysis_summaries" / "beautiful_figure" / "main_paper"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
UNIFIED_DIR.mkdir(parents=True, exist_ok=True)

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


def compute_fast_at_k(c_p: int, N: int, k: int) -> float:
    if N <= 0 or k <= 0 or k > N:
        return np.nan
    if c_p >= k:
        return 1.0
    if c_p == 0:
        return 0.0
    if N - c_p < k:
        return 1.0
    return 1.0 - comb(N - c_p, k) / comb(N, k)


def compute_speedup_at_k_exact(speedups_gated, p_thr, N, k):
    if len(speedups_gated) != N or N < k:
        return 0.0
    total = 0.0
    n_combos = 0
    for combo in combinations(range(N), k):
        max_s = max(speedups_gated[i] for i in combo)
        if max_s >= p_thr:
            total += max_s
        n_combos += 1
    return total / n_combos if n_combos > 0 else 0.0


def recompute_ex3_metrics(df_raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = df_raw.groupby(["benchmark", "model", "dataset"])
    for (bm, model, ds), grp in groups:
        if model == "baseline":
            continue
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
    return pd.DataFrame(rows)


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


def _aggregate_ex3(df_fixed):
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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    EX_VERSIONS = ["EX1", "EX2", "EX3"]
    fig, axes = plt.subplots(3, 5, figsize=(15, 8), dpi=150)
    plt.subplots_adjust(left=0.065, right=0.985, top=0.9, bottom=0.10, hspace=0.15, wspace=0.15)

    for row_idx, ex in enumerate(EX_VERSIONS):
        for col_idx, ds in enumerate(DATASET_SIZES):
            ax = axes[row_idx, col_idx]

            if ex in ["EX1", "EX2"]:
                if ds in ex12_data[ex]:
                    df_e = ex12_data[ex][ds]
                    for model in MODELS:
                        df_model = df_e[df_e["model"] == model]
                        if len(df_model) > 0:
                            ax.plot(df_model["p_thr"], df_model["Speedup_at_k_avg"],
                                    color=MODEL_COLORS[model], linewidth=2, alpha=0.85)
            elif ex == "EX3":
                if ds in ex3_data:
                    df_ex3 = ex3_data[ds]
                    for model in MODELS:
                        df_model = df_ex3[df_ex3["model"] == model].sort_values("p_thr")
                        if len(df_model) > 0:
                            ax.plot(df_model["p_thr"], df_model["speedup_at_3"],
                                    color=MODEL_COLORS[model], linewidth=2, alpha=0.85)

            ax.grid(True, axis="y", alpha=0.15, linestyle="-", linewidth=0.4, color="gray")
            ax.set_axisbelow(True)
            if col_idx != 0: ax.set_yticklabels([])
            if row_idx != 2: ax.set_xticklabels([])
            if row_idx == 0:
                ax.set_title(DATASET_LABELS[ds], fontsize=11, pad=15)
            ax.tick_params(axis="both", labelsize=8)

    row_labels = ["EX1 (Serial)", "EX2 (OpenMP)", "EX3 (CUDA)"]
    row_positions = [0.77, 0.50, 0.23]
    for i, label in enumerate(row_labels):
        fig.text(0.02, row_positions[i], label, fontsize=12, fontweight="bold",
                 ha="center", va="center", rotation=90)

    fig.text(0.53, 0.03, r"Speedup threshold $p_{\mathrm{thr}}$", ha="center", fontsize=13)
    fig.text(0.005, 0.5, r"Speedup@3", ha="center", va="center", rotation=90, fontsize=13)

    handles = [plt.Line2D([0], [0], color=MODEL_COLORS[m], linewidth=2.5, marker="o",
                           markersize=5, label=MODEL_LABELS[m]) for m in MODELS]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=True, fontsize=11,
               bbox_to_anchor=(0.5, 1.02), columnspacing=1.5, handlelength=2.5)

    out = FIGURE_DIR / filename
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  saved: {out}")


def plot_fast_figure(ex12_data, ex3_data, filename):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    EX_VERSIONS = ["EX1", "EX2", "EX3"]
    fig, axes = plt.subplots(3, 5, figsize=(15, 8), dpi=150)
    plt.subplots_adjust(left=0.055, right=0.985, top=0.92, bottom=0.08, hspace=0.15, wspace=0.15)

    for row_idx, ex in enumerate(EX_VERSIONS):
        for col_idx, ds in enumerate(DATASET_SIZES):
            ax = axes[row_idx, col_idx]

            if ex in ["EX1", "EX2"]:
                if ds in ex12_data[ex]:
                    df_e = ex12_data[ex][ds]
                    for model in MODELS:
                        df_model = df_e[df_e["model"] == model]
                        if len(df_model) > 0:
                            ax.plot(df_model["p_thr"], df_model["Fast_at_k_avg"],
                                    color=MODEL_COLORS[model], linewidth=2, alpha=0.85)
            elif ex == "EX3":
                if ds in ex3_data:
                    df_ex3 = ex3_data[ds]
                    for model in MODELS:
                        df_model = df_ex3[df_ex3["model"] == model].sort_values("p_thr")
                        if len(df_model) > 0:
                            ax.plot(df_model["p_thr"], df_model["fast_at_3"],
                                    color=MODEL_COLORS[model], linewidth=2, alpha=0.85)

            ax.grid(True, axis="y", alpha=0.15, linestyle="-", linewidth=0.4, color="gray")
            ax.set_axisbelow(True)
            ax.set_ylim(-0.05, 1.05)
            if col_idx != 0: ax.set_yticklabels([])
            if row_idx != 2: ax.set_xticklabels([])
            if row_idx == 0:
                ax.set_title(DATASET_LABELS[ds], fontsize=11, pad=15)
            ax.tick_params(axis="both", labelsize=8)

    row_labels = ["EX1 (Serial)", "EX2 (OpenMP)", "EX3 (CUDA)"]
    row_positions = [0.77, 0.50, 0.23]
    for i, label in enumerate(row_labels):
        fig.text(0.02, row_positions[i], label, fontsize=12, fontweight="bold",
                 ha="center", va="center", rotation=90)

    fig.text(0.53, 0.03, r"Speedup threshold $p_{\mathrm{thr}}$", ha="center", fontsize=13)
    fig.text(0.005, 0.5, r"Fast@3", ha="center", va="center", rotation=90, fontsize=13)

    handles = [plt.Line2D([0], [0], color=MODEL_COLORS[m], linewidth=2.5, marker="o",
                           markersize=5, label=MODEL_LABELS[m]) for m in MODELS]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=True, fontsize=11,
               bbox_to_anchor=(0.5, 1.02), columnspacing=1.5, handlelength=2.5)

    out = FIGURE_DIR / filename
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  saved: {out}")


def main():
    print(f"[1/3] Loading {UNIFIED_CSV.name} ...")
    df_raw = pd.read_csv(UNIFIED_CSV)
    print(f"  {len(df_raw)} rows, {df_raw['benchmark'].nunique()} benchmarks")

    print(f"[2/3] Recomputing Fast@{K}/Speedup@{K} (fallback=0)...")
    df_fixed = recompute_ex3_metrics(df_raw)
    out_csv = UNIFIED_DIR / "fast_speedup_at_k3_with_pthr_EX3_rerun.csv"
    df_fixed.to_csv(out_csv, index=False)
    print(f"  saved: {out_csv} ({len(df_fixed)} rows)")

    print(f"[3/3] Plotting figures...")
    ex12 = _load_ex1_ex2_data()
    ex3_agg = _aggregate_ex3(df_fixed)
    plot_speedup_figure(ex12, ex3_agg, "figure_speedup_at_3_beautiful_rerun.pdf")
    plot_fast_figure(ex12, ex3_agg, "figure_fast_at_3_beautiful_rerun.pdf")
    print("Done.")


if __name__ == "__main__":
    main()
