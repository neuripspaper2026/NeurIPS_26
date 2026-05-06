#!/usr/bin/env python3
"""
Build the new combined main-paper figure (replaces Figs. 2 and 3):

  Layout: 2 rows x 3 cols
    rows = [Fast@3, Speedup@3]
    cols = [EX1 (Serial), EX2 (OpenMP, 16 threads), EX3 (CUDA)]
    each panel: x = p_thr, y = metric, lines = models (Claude/GPT/Qwen)
    metric = macro-average across all 5 dataset sizes (mini..extra-large).

Also re-renders the original 3x5 per-dataset-size detail figures with
the GPT-5.1 -> GPT label change, for the appendix.

Inputs (same as the per-dataset versions):
  - EX1, EX2: final_data/unified_results.csv
  - EX3:     analysis_summaries/unified_data_real/unified_df_EX3.csv
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

EX12_CSV  = PROJECT_ROOT / "final_data" / "unified_results.csv"
EX3_CSV   = PROJECT_ROOT / "analysis_summaries" / "unified_data_real" / "unified_df_EX3.csv"
TABLES_DIR = PROJECT_ROOT / "analysis_summaries" / "tables"
FIG_DIR    = PROJECT_ROOT / "analysis_summaries" / "beautiful_figure" / "main_paper"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATASET_SIZES = ["mini", "small", "medium", "large", "extra-large"]
MODELS        = ["claude", "gpt5.1", "qwen"]
P_THR_LIST    = [0.0, 1.0, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.40, 1.50, 2.0, 3.0]
K, N_EXPECTED = 3, 10
EX2_THREADS   = 16

# Same colors as the existing main figures
MODEL_COLORS  = {"claude": "#2E86AB", "gpt5.1": "#A23B72", "qwen": "#F18F01"}
# Per user: Claude / GPT / Qwen (GPT-5.1 -> GPT, others unchanged)
MODEL_LABELS  = {"claude": "Claude", "gpt5.1": "GPT", "qwen": "Qwen"}
DATASET_LABELS = {
    "mini": "Mini", "small": "Small", "medium": "Medium",
    "large": "Large", "extra-large": "Extra-Large",
}
EX_LABELS = {"EX1": "EX1 (Serial)", "EX2": "EX2 (OpenMP)", "EX3": "EX3 (CUDA)"}


def fast_k(c_p, N, k):
    if c_p >= k: return 1.0
    if c_p == 0: return 0.0
    if N - c_p < k: return 1.0
    return 1.0 - comb(N - c_p, k) / comb(N, k)


def speedup_k(speeds, p, N, k):
    if N < k: return 0.0
    tot, n = 0.0, 0
    for combo in combinations(range(N), k):
        m = max(speeds[i] for i in combo)
        if m >= p: tot += m
        n += 1
    return tot / n if n else 0.0


def per_benchmark_metrics(speeds):
    out = {}
    for p in P_THR_LIST:
        c_p = sum(1 for x in speeds if x >= p)
        out[p] = (fast_k(c_p, N_EXPECTED, K), speedup_k(speeds, p, N_EXPECTED, K))
    return out


# ---- gated-vector builders --------------------------------------------------
def build_ex12(df_all):
    df = df_all.copy()
    df = df[(df["ex"] == "EX1") | ((df["ex"] == "EX2") & (df["threads"] == EX2_THREADS))]
    df["is_correct_kernel"] = (
        (df["compile_status"] == "success")
        & (df["correctness"] == 1.0)
        & df["kernel_speedup"].notna()
    )
    df["gated"] = np.where(df["is_correct_kernel"], df["kernel_speedup"], 0.0)
    gated = {}
    for (ex, ds, model, bench), grp in df.groupby(["ex", "dataset_size", "model", "benchmark"]):
        if model == "baseline":
            continue
        s = grp["gated"].tolist()
        if len(s) > N_EXPECTED: s = s[:N_EXPECTED]
        elif len(s) < N_EXPECTED: s = s + [0.0] * (N_EXPECTED - len(s))
        gated[(ex, ds, model, bench)] = s
    return gated


def build_ex3(df_e3):
    df = df_e3.copy()
    df = df[df["model"] != "baseline"]
    df["is_correct_kernel"] = df["is_correct"] & df["kernel_speedup"].notna()
    df["gated"] = np.where(df["is_correct_kernel"], df["kernel_speedup"], 0.0)
    gated = {}
    for (ds, model, bench), grp in df.groupby(["dataset", "model", "benchmark"]):
        s = grp["gated"].tolist()
        if len(s) > N_EXPECTED: s = s[:N_EXPECTED]
        elif len(s) < N_EXPECTED: s = s + [0.0] * (N_EXPECTED - len(s))
        gated[("EX3", ds, model, bench)] = s
    return gated


# ---- Aggregation: macro-average across benchmarks within (ex, ds, model) ---
def aggregate_per_dataset(gated):
    rows = []
    keyed = {}
    for (ex, ds, model, bench), speeds in gated.items():
        keyed.setdefault((ex, ds, model), []).append(speeds)
    for (ex, ds, model), entries in keyed.items():
        per_p_f = {p: [] for p in P_THR_LIST}
        per_p_s = {p: [] for p in P_THR_LIST}
        for sp in entries:
            mp = per_benchmark_metrics(sp)
            for p in P_THR_LIST:
                f, s = mp[p]
                if not np.isnan(f): per_p_f[p].append(f)
                per_p_s[p].append(s)
        for p in P_THR_LIST:
            rows.append({
                "ex": ex, "dataset": ds, "model": model, "p_thr": p,
                "fast_at_3":   np.mean(per_p_f[p]) if per_p_f[p] else np.nan,
                "speedup_at_3":np.mean(per_p_s[p]) if per_p_s[p] else np.nan,
                "n_benchmarks": len(entries),
            })
    return pd.DataFrame(rows)


def aggregate_dataset_avg(df_per_ds):
    """Average per-dataset metrics across the 5 dataset sizes."""
    return (df_per_ds
            .groupby(["ex", "model", "p_thr"])
            .agg(fast_at_3=("fast_at_3", "mean"),
                 speedup_at_3=("speedup_at_3", "mean"))
            .reset_index())


# ---- Plotters --------------------------------------------------------------
def plot_combined_2x3(df_avg, out_name):
    """2 rows x 3 cols main figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    EX_VERSIONS = ["EX1", "EX2", "EX3"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), dpi=150)
    plt.subplots_adjust(left=0.085, right=0.985, top=0.88, bottom=0.12,
                        hspace=0.20, wspace=0.18)

    rows_spec = [
        ("fast_at_3",    "Fast@3",    (-0.05, 1.05)),
        ("speedup_at_3", "Speedup@3", None),
    ]

    for r_i, (col, ylab, ylim) in enumerate(rows_spec):
        for c_i, ex in enumerate(EX_VERSIONS):
            ax = axes[r_i, c_i]
            d_ex = df_avg[df_avg["ex"] == ex]
            for m in MODELS:
                d_m = d_ex[d_ex["model"] == m].sort_values("p_thr")
                if len(d_m):
                    ax.plot(d_m["p_thr"], d_m[col],
                            color=MODEL_COLORS[m], linewidth=2.2, alpha=0.9)
            ax.grid(True, axis="y", alpha=0.18, linestyle="-",
                    linewidth=0.4, color="gray")
            ax.set_axisbelow(True)
            if ylim is not None:
                ax.set_ylim(*ylim)
            if r_i == 0:
                ax.set_title(EX_LABELS[ex], fontsize=12, pad=8)
            if r_i == 1:
                ax.set_xlabel(r"Speedup threshold $p_{\mathrm{thr}}$",
                              fontsize=11)
            if c_i == 0:
                ax.set_ylabel(ylab, fontsize=12)
            ax.tick_params(axis="both", labelsize=9)

    handles = [plt.Line2D([0], [0], color=MODEL_COLORS[m], linewidth=2.5,
                          label=MODEL_LABELS[m])
               for m in MODELS]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=True,
               fontsize=11, bbox_to_anchor=(0.5, 1.01),
               columnspacing=2.0, handlelength=2.5)

    out = FIG_DIR / out_name
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  saved: {out}")


def plot_per_dataset_3x5(df_per_ds, ycol, out_name, ylab, ylim=None):
    """3 rows (EX) x 5 cols (dataset) — appendix detail figure, with GPT label."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    EX_VERSIONS = ["EX1", "EX2", "EX3"]
    fig, axes = plt.subplots(3, 5, figsize=(15, 8), dpi=150)
    plt.subplots_adjust(left=0.065, right=0.985,
                        top=0.92 if ylim else 0.90,
                        bottom=0.08 if ylim else 0.10,
                        hspace=0.15, wspace=0.15)

    for r_i, ex in enumerate(EX_VERSIONS):
        for c_i, ds in enumerate(DATASET_SIZES):
            ax = axes[r_i, c_i]
            sub = df_per_ds[(df_per_ds["ex"] == ex) & (df_per_ds["dataset"] == ds)]
            for m in MODELS:
                d_m = sub[sub["model"] == m].sort_values("p_thr")
                if len(d_m):
                    ax.plot(d_m["p_thr"], d_m[ycol],
                            color=MODEL_COLORS[m], linewidth=2, alpha=0.85)
            ax.grid(True, axis="y", alpha=0.15, linestyle="-",
                    linewidth=0.4, color="gray")
            ax.set_axisbelow(True)
            if ylim is not None: ax.set_ylim(*ylim)
            if c_i != 0: ax.set_yticklabels([])
            if r_i != 2: ax.set_xticklabels([])
            if r_i == 0:
                ax.set_title(DATASET_LABELS[ds], fontsize=11, pad=15)
            ax.tick_params(axis="both", labelsize=8)

    for i, lab in enumerate(["EX1 (Serial)", "EX2 (OpenMP)", "EX3 (CUDA)"]):
        fig.text(0.02, [0.77, 0.50, 0.23][i], lab, fontsize=12, fontweight="bold",
                 ha="center", va="center", rotation=90)

    fig.text(0.53, 0.03, r"Speedup threshold $p_{\mathrm{thr}}$",
             ha="center", fontsize=13)
    fig.text(0.005, 0.5, ylab, ha="center", va="center", rotation=90, fontsize=13)

    handles = [plt.Line2D([0], [0], color=MODEL_COLORS[m], linewidth=2.5,
                          label=MODEL_LABELS[m])
               for m in MODELS]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=True, fontsize=11,
               bbox_to_anchor=(0.5, 1.02), columnspacing=1.5, handlelength=2.5)

    out = FIG_DIR / out_name
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  saved: {out}")


def main():
    print("[1/4] Loading EX1/EX2 unified ...")
    df12 = pd.read_csv(EX12_CSV)
    print(f"  rows: {len(df12)}")

    print("[2/4] Loading EX3 unified (real correctness) ...")
    df3 = pd.read_csv(EX3_CSV)
    print(f"  rows: {len(df3)}")

    print("[3/4] Building gated kernel_speedup vectors and per-dataset metrics ...")
    gated = build_ex12(df12)
    gated.update(build_ex3(df3))
    print(f"  total (ex,ds,model,bench) cells: {len(gated)}")
    df_per_ds = aggregate_per_dataset(gated)

    df_avg = aggregate_dataset_avg(df_per_ds)
    out_csv = PROJECT_ROOT / "analysis_summaries" / "unified_data_real" / "fast_speedup_at_k3_kernel_avg_over_datasets.csv"
    df_avg.to_csv(out_csv, index=False)
    print(f"  saved aggregated CSV: {out_csv}")

    print("[4/4] Plotting ...")
    # Combined main figure (2x3, dataset-averaged)
    plot_combined_2x3(df_avg, "figure_fast_speedup_at_3_combined.pdf")
    # Detail figures for appendix (3x5, per-dataset, GPT label updated)
    plot_per_dataset_3x5(df_per_ds, "fast_at_3",
                         "figure_fast_at_3_per_dataset.pdf",
                         r"Fast@3", ylim=(-0.05, 1.05))
    plot_per_dataset_3x5(df_per_ds, "speedup_at_3",
                         "figure_speedup_at_3_per_dataset.pdf",
                         r"Speedup@3")


if __name__ == "__main__":
    main()
