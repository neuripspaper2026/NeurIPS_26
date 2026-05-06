#!/usr/bin/env python3
"""
Re-render the EX1 medium k-sensitivity figure (originally Figure 3 in the
paper) in two new styles:

  (a) Fast@k vs k  -> heatmap (3 models x 10 k values)
  (b) Speedup@k vs k -> filled step-area for the same 3 model series

Same data source as plot_k_sensitivity_medium_only.py: per-trial table
analysis_summaries/tables/table_a_trial_level_EX1_medium.csv,
filtered to EX1 medium, p_thr=1.05, k in 1..10. Macro-averaged across
benchmarks per (model, k).

Outputs:
  - analysis_summaries/beautiful_figure/main_paper/sensitivity_analysis/
      fast_at_k_vs_k_EX1_medium_dotplot.pdf
  - analysis_summaries/beautiful_figure/main_paper/sensitivity_analysis/
      speedup_at_k_vs_k_EX1_medium_stepfill.pdf
"""
from __future__ import annotations

from pathlib import Path
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = PROJECT_ROOT / "analysis_summaries" / "tables"
FIG_DIR   = PROJECT_ROOT / "analysis_summaries" / "beautiful_figure" / "main_paper" / "sensitivity_analysis"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATASET = "medium"
P_THR   = 1.05
N_TOTAL = 10
K_RANGE = list(range(1, N_TOTAL + 1))
MODELS  = ["claude", "gpt5.1", "qwen"]
MODEL_COLORS = {"claude": "#2E86AB", "gpt5.1": "#A23B72", "qwen": "#F18F01"}
# Per user direction: GPT-5.1 -> GPT, Claude/Qwen unchanged
MODEL_LABELS = {"claude": "Claude", "gpt5.1": "GPT", "qwen": "Qwen"}


def fast_k(c_p, N, k):
    if c_p < 0 or N <= 0 or k <= 0 or k > N: return np.nan
    if c_p >= k: return 1.0
    if c_p == 0: return 0.0
    num = comb(N - c_p, k); den = comb(N, k)
    return np.nan if den == 0 else 1.0 - num / den


def speedup_k_exact(speedups_correct, p_thr, N, k):
    if k > N or k <= 0: return np.nan
    speedups = list(speedups_correct) + [0.0] * (N - len(speedups_correct))
    tot, cnt = 0.0, 0
    for combo in combinations(speedups, k):
        m = max(combo)
        if m >= p_thr: tot += m
        cnt += 1
    return tot / cnt if cnt else 0.0


def compute_metrics(df, p_thr, k_range):
    rows = []
    for model in MODELS:
        d = df[df["model"] == model]
        if len(d) == 0: continue
        for k in k_range:
            fasts, speeds = [], []
            for bench in d["benchmark"].unique():
                db = d[d["benchmark"] == bench].head(N_TOTAL)
                actual_n = len(db)
                correct_speedups = db[db["correctness"] == 1.0]["kernel_speedup"].values
                c_p = len(correct_speedups)
                f = fast_k(c_p, actual_n, k)
                if not np.isnan(f): fasts.append(f)
                s = speedup_k_exact(correct_speedups, p_thr, actual_n, k)
                if not np.isnan(s): speeds.append(s)
            rows.append({
                "model": model, "k": k,
                "Fast_at_k":    float(np.mean(fasts))   if fasts   else np.nan,
                "Speedup_at_k": float(np.mean(speeds)) if speeds else np.nan,
            })
    return pd.DataFrame(rows)


def plot_dotplot(df, out_path):
    """Dot plot of Fast@k vs k for 3 models, with direct labels at the right
    of each series (no legend), matching the user's reference style."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=150)

    # Plot each model as scatter dots
    last_points = {}
    for m in MODELS:
        sub = df[df["model"] == m].sort_values("k")
        xs = sub["k"].values.astype(float)
        ys = sub["Fast_at_k"].values
        ax.scatter(xs, ys, s=42, color=MODEL_COLORS[m], alpha=0.95,
                   edgecolor="white", linewidth=0.6, zorder=3)
        last_points[m] = (xs[-1], ys[-1])

    # Direct labels on the right of the rightmost point
    # Order labels by y-value to avoid overlap; nudge upward/downward if close
    sorted_by_y = sorted(MODELS, key=lambda m: last_points[m][1])
    label_y_offsets = {m: 0.0 for m in MODELS}
    # If two adjacent series overlap on y at k=10, push them apart
    for i in range(1, len(sorted_by_y)):
        m_lo, m_hi = sorted_by_y[i-1], sorted_by_y[i]
        gap = last_points[m_hi][1] - last_points[m_lo][1]
        if gap < 0.012:
            label_y_offsets[m_hi] += (0.012 - gap)

    x_max = max(K_RANGE)
    label_x = x_max + 0.55
    for m in MODELS:
        x, y = last_points[m]
        y_lab = y + label_y_offsets[m]
        # Tiny leader from the rightmost dot to the label
        ax.plot([x + 0.10, label_x - 0.08], [y, y_lab],
                color=MODEL_COLORS[m], linewidth=0.8, alpha=0.5, zorder=2)
        ax.text(label_x, y_lab, MODEL_LABELS[m],
                color=MODEL_COLORS[m], fontsize=15, fontweight="bold",
                va="center", ha="left")

    ax.set_xlim(K_RANGE[0] - 0.4, x_max + 1.8)
    # Auto y-limits with a small margin so labels are visible
    all_ys = df["Fast_at_k"].dropna().values
    y_lo, y_hi = float(np.min(all_ys)), float(np.max(all_ys))
    pad = (y_hi - y_lo) * 0.18 if (y_hi - y_lo) > 0 else 0.05
    ax.set_ylim(y_lo - pad, y_hi + pad)

    ax.set_xticks(K_RANGE)
    ax.set_xticklabels(K_RANGE, fontsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_xlabel(r"Sampling budget $k$", fontsize=15)
    ax.set_ylabel(r"$\mathrm{Fast}@k$", fontsize=15)

    # Vertical dashed gridlines at each k, light style
    ax.grid(True, axis="x", linestyle=":", linewidth=0.6, alpha=0.45, color="gray")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.30, color="gray")
    ax.set_axisbelow(True)

    # Remove top / right spines for a cleaner look
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  saved: {out_path}")


def plot_stepfill(df, out_path):
    """Speedup@k vs k; one filled step-area per model.
    Series are drawn in ascending order (smaller area first, larger last)
    so all series remain visible (overlapping, alpha-blended)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=150)

    # Get per-model series
    series = {}
    for m in MODELS:
        sub = df[df["model"] == m].sort_values("k")
        series[m] = sub["Speedup_at_k"].values

    # Sort by max value ascending so larger areas are drawn on top
    order = sorted(MODELS, key=lambda m: np.max(series[m]))
    ks = np.array(K_RANGE, dtype=float)

    for m in order:
        y = series[m]
        ax.fill_between(ks, 0, y, step="pre",
                        color=MODEL_COLORS[m], alpha=0.40, linewidth=0)
        ax.step(ks, y, where="pre",
                color=MODEL_COLORS[m], linewidth=2.2, label=MODEL_LABELS[m])

    ax.set_xlim(K_RANGE[0], K_RANGE[-1])
    ax.set_ylim(0, max(np.max(s) for s in series.values()) * 1.08)
    ax.set_xticks(K_RANGE)
    ax.set_xticklabels(K_RANGE, fontsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_xlabel(r"Sampling budget $k$", fontsize=15)
    ax.set_ylabel(r"$\mathrm{Speedup}@k$", fontsize=15)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)

    # Legend in original Claude/GPT/Qwen order, regardless of plotting order
    handles_in_order = []
    for m in MODELS:
        handles_in_order.append(plt.Rectangle((0, 0), 1, 1,
                                              fc=MODEL_COLORS[m], ec=MODEL_COLORS[m],
                                              alpha=0.55, label=MODEL_LABELS[m]))
    ax.legend(handles=handles_in_order, loc="lower right", frameon=True, fontsize=14)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  saved: {out_path}")


def main():
    src = TABLE_DIR / f"table_a_trial_level_EX1_{DATASET}.csv"
    print(f"[1/3] Loading {src.name} ...")
    df = pd.read_csv(src)
    df = df[(df["ex"] == "EX1") & (df["dataset_size"] == DATASET)]
    print(f"  rows: {len(df)}")

    print(f"[2/3] Computing Fast@k / Speedup@k for k=1..10, p_thr={P_THR} ...")
    res = compute_metrics(df, P_THR, K_RANGE)
    out_csv = FIG_DIR / "k_sensitivity_EX1_medium_p_thr_1.05.csv"
    res.to_csv(out_csv, index=False)
    print(f"  saved: {out_csv}")
    print(res.pivot(index="model", columns="k",
                    values=["Fast_at_k", "Speedup_at_k"]).round(3))

    print("[3/3] Plotting ...")
    plot_dotplot(res, FIG_DIR / "fast_at_k_vs_k_EX1_medium_dotplot.pdf")
    plot_stepfill(res, FIG_DIR / "speedup_at_k_vs_k_EX1_medium_stepfill.pdf")


if __name__ == "__main__":
    main()
