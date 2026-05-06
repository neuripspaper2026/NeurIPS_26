#!/usr/bin/env python3
"""
Generate appendix sampling-budget sensitivity figures for EX2 and EX3
on the medium dataset at p_thr=1.05, mirroring the main paper Fig 3:

  Fast@k vs k     -> dot plot with direct labels
  Speedup@k vs k  -> filled step-area

Inputs:
  - EX2: final_data/unified_results.csv (filter EX2, threads=16, dataset=medium)
  - EX3: analysis_summaries/unified_data_real/unified_df_EX3.csv
         (April rerun + real correctness, broadcast)

Outputs (figure/appendix_figure/):
  - fast_at_k_vs_k_EX2_medium_dotplot.pdf
  - speedup_at_k_vs_k_EX2_medium_stepfill.pdf
  - fast_at_k_vs_k_EX3_medium_dotplot.pdf
  - speedup_at_k_vs_k_EX3_medium_stepfill.pdf
"""
from __future__ import annotations

from pathlib import Path
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EX12_CSV = PROJECT_ROOT / "final_data" / "unified_results.csv"
EX3_CSV  = PROJECT_ROOT / "analysis_summaries" / "unified_data_real" / "unified_df_EX3.csv"
OUT_DIR  = PROJECT_ROOT / "NeurIPS_26" / "figure" / "appendix_figure"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET   = "medium"
P_THR     = 1.05
N_TOTAL   = 10
K_RANGE   = list(range(1, N_TOTAL + 1))
EX2_THR   = 16

MODELS         = ["claude", "gpt5.1", "qwen"]
MODEL_COLORS   = {"claude": "#2E86AB", "gpt5.1": "#A23B72", "qwen": "#F18F01"}
MODEL_LABELS   = {"claude": "Claude", "gpt5.1": "GPT", "qwen": "Qwen"}


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


def per_model_metrics(per_bench_correct_speedups, k_range, p_thr):
    """For each model and k, macro-average Fast@k and Speedup@k across benchmarks."""
    rows = []
    for model in MODELS:
        per_bench = per_bench_correct_speedups.get(model, {})
        if not per_bench:
            continue
        for k in k_range:
            fasts, speeds = [], []
            for bench, correct_sps in per_bench.items():
                actual_n = N_TOTAL  # all entries reflect 10 trials (correct subset)
                c_p = sum(1 for s in correct_sps if s >= p_thr)
                f = fast_k(c_p, actual_n, k)
                if not np.isnan(f): fasts.append(f)
                s = speedup_k_exact(correct_sps, p_thr, actual_n, k)
                if not np.isnan(s): speeds.append(s)
            rows.append({
                'model': model, 'k': k,
                'Fast_at_k': float(np.mean(fasts)) if fasts else np.nan,
                'Speedup_at_k': float(np.mean(speeds)) if speeds else np.nan,
            })
    return pd.DataFrame(rows)


# ---------------- data loaders ----------------
def load_ex2():
    df = pd.read_csv(EX12_CSV)
    df = df[(df['ex'] == 'EX2') & (df['threads'] == float(EX2_THR))
            & (df['dataset_size'] == DATASET) & (df['model'] != 'baseline')]
    out = {}
    for (model, bench), grp in df.groupby(['model', 'benchmark']):
        correct = grp[(grp['compile_status'] == 'success')
                      & (grp['correctness'] == 1.0)
                      & grp['kernel_speedup'].notna()]
        sps = correct['kernel_speedup'].tolist()
        out.setdefault(model, {})[bench] = sps
    return out


def load_ex3():
    df = pd.read_csv(EX3_CSV)
    df = df[(df['dataset'] == DATASET) & (df['model'] != 'baseline')]
    out = {}
    for (model, bench), grp in df.groupby(['model', 'benchmark']):
        correct = grp[(grp['is_correct'] == True) & grp['kernel_speedup'].notna()]
        sps = correct['kernel_speedup'].tolist()
        out.setdefault(model, {})[bench] = sps
    return out


# ---------------- plotters (same style as main paper Fig 3) ----------------
def plot_dotplot(df, out_path, title_suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=150)

    last_points = {}
    for m in MODELS:
        sub = df[df["model"] == m].sort_values("k")
        if not len(sub): continue
        xs = sub["k"].values.astype(float)
        ys = sub["Fast_at_k"].values
        ax.scatter(xs, ys, s=42, color=MODEL_COLORS[m], alpha=0.95,
                   edgecolor="white", linewidth=0.6, zorder=3)
        last_points[m] = (xs[-1], ys[-1])

    sorted_by_y = sorted(last_points, key=lambda m: last_points[m][1])
    label_y_offsets = {m: 0.0 for m in last_points}
    for i in range(1, len(sorted_by_y)):
        m_lo, m_hi = sorted_by_y[i-1], sorted_by_y[i]
        gap = last_points[m_hi][1] - last_points[m_lo][1]
        if gap < 0.012:
            label_y_offsets[m_hi] += (0.012 - gap)

    x_max = max(K_RANGE)
    label_x = x_max + 0.55
    for m, (x, y) in last_points.items():
        y_lab = y + label_y_offsets[m]
        ax.plot([x + 0.10, label_x - 0.08], [y, y_lab],
                color=MODEL_COLORS[m], linewidth=0.8, alpha=0.5, zorder=2)
        ax.text(label_x, y_lab, MODEL_LABELS[m],
                color=MODEL_COLORS[m], fontsize=15, fontweight="bold",
                va="center", ha="left")

    ax.set_xlim(K_RANGE[0] - 0.4, x_max + 1.8)
    all_ys = df["Fast_at_k"].dropna().values
    if len(all_ys):
        y_lo, y_hi = float(np.min(all_ys)), float(np.max(all_ys))
        pad = (y_hi - y_lo) * 0.18 if (y_hi - y_lo) > 0 else 0.05
        ax.set_ylim(max(0.0, y_lo - pad), min(1.05, y_hi + pad))

    ax.set_xticks(K_RANGE)
    ax.set_xticklabels(K_RANGE, fontsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_xlabel(r"Sampling budget $k$", fontsize=15)
    ax.set_ylabel(r"$\mathrm{Fast}@k$", fontsize=15)
    ax.grid(True, axis="x", linestyle=":", linewidth=0.6, alpha=0.45, color="gray")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.30, color="gray")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  saved: {out_path}")


def plot_stepfill(df, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=150)

    series = {}
    for m in MODELS:
        sub = df[df["model"] == m].sort_values("k")
        series[m] = sub["Speedup_at_k"].values

    order = sorted(series, key=lambda m: np.max(series[m]) if len(series[m]) else 0)
    ks = np.array(K_RANGE, dtype=float)

    for m in order:
        y = series[m]
        if not len(y): continue
        ax.fill_between(ks, 0, y, step="pre",
                        color=MODEL_COLORS[m], alpha=0.40, linewidth=0)
        ax.step(ks, y, where="pre",
                color=MODEL_COLORS[m], linewidth=2.2, label=MODEL_LABELS[m])

    ax.set_xlim(K_RANGE[0], K_RANGE[-1])
    max_y = max((np.max(s) for s in series.values() if len(s)), default=1.0)
    ax.set_ylim(0, max_y * 1.08)
    ax.set_xticks(K_RANGE)
    ax.set_xticklabels(K_RANGE, fontsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_xlabel(r"Sampling budget $k$", fontsize=15)
    ax.set_ylabel(r"$\mathrm{Speedup}@k$", fontsize=15)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)

    handles_in_order = []
    for m in MODELS:
        if m not in series or not len(series[m]): continue
        handles_in_order.append(plt.Rectangle(
            (0, 0), 1, 1, fc=MODEL_COLORS[m], ec=MODEL_COLORS[m],
            alpha=0.55, label=MODEL_LABELS[m]))
    ax.legend(handles=handles_in_order, loc="lower right",
              frameon=True, fontsize=14)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  saved: {out_path}")


def main():
    print("[1/4] Loading EX2 ...")
    ex2_data = load_ex2()
    res_ex2 = per_model_metrics(ex2_data, K_RANGE, P_THR)
    print("[2/4] Loading EX3 ...")
    ex3_data = load_ex3()
    res_ex3 = per_model_metrics(ex3_data, K_RANGE, P_THR)

    print("[3/4] Plotting EX2 ...")
    plot_dotplot(res_ex2, OUT_DIR / "fast_at_k_vs_k_EX2_medium_dotplot.pdf")
    plot_stepfill(res_ex2, OUT_DIR / "speedup_at_k_vs_k_EX2_medium_stepfill.pdf")
    print("[4/4] Plotting EX3 ...")
    plot_dotplot(res_ex3, OUT_DIR / "fast_at_k_vs_k_EX3_medium_dotplot.pdf")
    plot_stepfill(res_ex3, OUT_DIR / "speedup_at_k_vs_k_EX3_medium_stepfill.pdf")


if __name__ == "__main__":
    main()
