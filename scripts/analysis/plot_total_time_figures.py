#!/usr/bin/env python3
"""
Generate appendix Fast@3 / Speedup@3 figures based on TOTAL (end-to-end)
runtime instead of kernel runtime, for EX1, EX2, and EX3.

Same metric definition and plotting layout as the main paper figures
(figure_*_at_3_beautiful_v2.pdf), but computed on the `total_speedup`
column instead of `kernel_speedup`. Correctness gating + fallback=0
are preserved so curves remain monotonically non-increasing in p_thr.

Inputs:
  - EX1, EX2: final_data/unified_results.csv  (full Cartesian, with
    correctness/compile_status flags and per-trial total_mean_s)
  - EX3:     analysis_summaries/unified_data_real/unified_df_EX3.csv
    (April timing rerun + real correctness, mini-broadcast)

Outputs:
  - analysis_summaries/beautiful_figure/main_paper/figure_fast_at_3_total.pdf
  - analysis_summaries/beautiful_figure/main_paper/figure_speedup_at_3_total.pdf
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

EX12_CSV = PROJECT_ROOT / "final_data" / "unified_results.csv"
EX3_CSV  = PROJECT_ROOT / "analysis_summaries" / "unified_data_real" / "unified_df_EX3.csv"
FIG_DIR  = PROJECT_ROOT / "analysis_summaries" / "beautiful_figure" / "main_paper"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATASET_SIZES = ["mini", "small", "medium", "large", "extra-large"]
MODELS        = ["claude", "gpt5.1", "qwen"]
P_THR_LIST    = [0.0, 1.0, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.40, 1.50, 2.0, 3.0]
K, N_EXPECTED = 3, 10
EX2_THREADS   = 16

MODEL_COLORS  = {"claude": "#2E86AB", "gpt5.1": "#A23B72", "qwen": "#F18F01"}
MODEL_LABELS  = {"claude": "Claude", "gpt5.1": "GPT", "qwen": "Qwen"}
DATASET_LABELS = {
    "mini": "Mini", "small": "Small", "medium": "Medium",
    "large": "Large", "extra-large": "Extra-Large",
}


def compute_fast_at_k(c_p, N, k):
    if N <= 0 or k <= 0 or k > N: return np.nan
    if c_p >= k: return 1.0
    if c_p == 0: return 0.0
    if N - c_p < k: return 1.0
    return 1.0 - comb(N - c_p, k) / comb(N, k)


def compute_speedup_at_k_exact(speedups_gated, p_thr, N, k):
    if len(speedups_gated) != N or N < k: return 0.0
    total = 0.0
    n_combos = 0
    for combo in combinations(range(N), k):
        max_s = max(speedups_gated[i] for i in combo)
        if max_s >= p_thr:
            total += max_s
        n_combos += 1
    return total / n_combos if n_combos > 0 else 0.0


def per_benchmark_metrics(speedups_gated):
    """Compute (Fast@K, Speedup@K) at each p_thr for one benchmark's gated list."""
    N = len(speedups_gated)
    out = {}
    for p in P_THR_LIST:
        c_p = sum(1 for s in speedups_gated if s >= p)
        out[p] = (
            compute_fast_at_k(c_p, N, K),
            compute_speedup_at_k_exact(speedups_gated, p, N, K),
        )
    return out


# =============================================================================
# Build a uniform per-(ex, dataset, model, benchmark) gated TOTAL speedup vector
# =============================================================================

def build_ex12_total(df_all):
    """For EX1/EX2 (final_data/unified_results.csv) build:
       gated[(ex, dataset, model, benchmark)] = list of N=10 total_speedup,
       with 0.0 substituted for incorrect / not-compiled / missing trials.
    """
    df = df_all.copy()
    # For EX2, restrict to 16-thread results to match main paper's setting
    df = df[(df["ex"] == "EX1") | ((df["ex"] == "EX2") & (df["threads"] == EX2_THREADS))]

    # Correctness: compile_status==success AND correctness==1
    df["is_correct_total"] = (
        (df["compile_status"] == "success")
        & (df["correctness"] == 1.0)
        & df["total_speedup"].notna()
    )
    # Replace incorrect rows with 0
    df["gated_total"] = np.where(df["is_correct_total"], df["total_speedup"], 0.0)

    gated = {}
    for (ex, ds, model, bench), grp in df.groupby(["ex", "dataset_size", "model", "benchmark"]):
        if model == "baseline":
            continue
        # 10 trials per benchmark/model/dataset (full Cartesian); take all rows
        speeds = grp["gated_total"].tolist()
        # If N != 10 for some reason, pad/trim
        if len(speeds) > N_EXPECTED:
            speeds = speeds[:N_EXPECTED]
        elif len(speeds) < N_EXPECTED:
            speeds = speeds + [0.0] * (N_EXPECTED - len(speeds))
        gated[(ex, ds, model, bench)] = speeds
    return gated


def build_ex3_total(df_e3):
    df = df_e3.copy()
    df = df[df["model"] != "baseline"]
    df["is_correct_total"] = df["is_correct"] & df["total_speedup"].notna()
    df["gated_total"] = np.where(df["is_correct_total"], df["total_speedup"], 0.0)
    gated = {}
    for (ds, model, bench), grp in df.groupby(["dataset", "model", "benchmark"]):
        speeds = grp["gated_total"].tolist()
        if len(speeds) > N_EXPECTED:
            speeds = speeds[:N_EXPECTED]
        elif len(speeds) < N_EXPECTED:
            speeds = speeds + [0.0] * (N_EXPECTED - len(speeds))
        gated[("EX3", ds, model, bench)] = speeds
    return gated


# =============================================================================
# Aggregate per (ex, dataset, model, p_thr)
# =============================================================================
def aggregate(gated):
    rows = []
    # group by ex, ds, model
    keyed = {}
    for (ex, ds, model, bench), speeds in gated.items():
        keyed.setdefault((ex, ds, model), []).append((bench, speeds))
    for (ex, ds, model), entries in keyed.items():
        # per-benchmark Fast / Speedup at each p_thr, then macro-avg
        per_p_fast = {p: [] for p in P_THR_LIST}
        per_p_speed = {p: [] for p in P_THR_LIST}
        for bench, speeds in entries:
            metrics = per_benchmark_metrics(speeds)
            for p in P_THR_LIST:
                f, s = metrics[p]
                if not np.isnan(f):
                    per_p_fast[p].append(f)
                per_p_speed[p].append(s)
        for p in P_THR_LIST:
            rows.append({
                "ex": ex,
                "dataset": ds,
                "model": model,
                "p_thr": p,
                "fast_at_3": np.mean(per_p_fast[p]) if per_p_fast[p] else np.nan,
                "speedup_at_3": np.mean(per_p_speed[p]) if per_p_speed[p] else np.nan,
                "n_benchmarks": len(entries),
            })
    return pd.DataFrame(rows)


# =============================================================================
# Plot
# =============================================================================
def plot(df_agg, ycol, fig_name, ylabel, ylim=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 5, figsize=(15, 8), dpi=150)
    plt.subplots_adjust(left=0.065, right=0.985, top=0.92, bottom=0.10, hspace=0.15, wspace=0.15)

    for r_i, ex in enumerate(["EX1", "EX2", "EX3"]):
        for c_i, ds in enumerate(DATASET_SIZES):
            ax = axes[r_i, c_i]
            sub_ds = df_agg[(df_agg["ex"] == ex) & (df_agg["dataset"] == ds)]
            for m in MODELS:
                d = sub_ds[sub_ds["model"] == m].sort_values("p_thr")
                if len(d):
                    ax.plot(d["p_thr"], d[ycol],
                            color=MODEL_COLORS[m], linewidth=2, alpha=0.85)
            ax.grid(True, axis="y", alpha=0.15, linestyle="-", linewidth=0.4, color="gray")
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

    fig.text(0.53, 0.03, r"Speedup threshold $p_{\mathrm{thr}}$", ha="center", fontsize=13)
    fig.text(0.005, 0.5, ylabel, ha="center", va="center", rotation=90, fontsize=13)

    handles = [plt.Line2D([0], [0], color=MODEL_COLORS[m], linewidth=2.5,
                          label=MODEL_LABELS[m]) for m in MODELS]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=True, fontsize=11,
               bbox_to_anchor=(0.5, 1.02), columnspacing=1.5, handlelength=2.5)

    out = FIG_DIR / fig_name
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

    print("[3/4] Building per-benchmark gated TOTAL speedup vectors ...")
    g = build_ex12_total(df12)
    g.update(build_ex3_total(df3))
    print(f"  total (ex,ds,model,bench) cells: {len(g)}")

    df_agg = aggregate(g)
    out_csv = PROJECT_ROOT / "analysis_summaries" / "unified_data_real" / "fast_speedup_at_k3_TOTAL_all_ex.csv"
    df_agg.to_csv(out_csv, index=False)
    print(f"  saved aggregated CSV: {out_csv}")

    print("[4/4] Plotting ...")
    plot(df_agg, "speedup_at_3", "figure_speedup_at_3_total.pdf", r"Speedup@3 (total time)")
    plot(df_agg, "fast_at_3",    "figure_fast_at_3_total.pdf",    r"Fast@3 (total time)", ylim=(-0.05, 1.05))


if __name__ == "__main__":
    main()
