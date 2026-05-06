#!/usr/bin/env python3
"""
EX2 thread-count sensitivity for the appendix.

For each thread count n in the data, compute Fast@3 / Speedup@3 at
p_thr=1.05 on the medium dataset, macro-averaged across benchmarks
per (model, threads). Plot as:

  Fast@3   -> dot plot with direct labels (mirroring main paper)
  Speedup@3 -> filled step area (mirroring main paper)

Data threads: {1, 2, 8, 16, 32} (final_data/unified_results.csv).
"""
from __future__ import annotations

from pathlib import Path
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EX12_CSV = PROJECT_ROOT / "final_data" / "unified_results.csv"
OUT_DIR  = PROJECT_ROOT / "NeurIPS_26" / "figure" / "appendix_figure"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET = "medium"
P_THR   = 1.05
N_TOTAL = 10
K       = 3

MODELS         = ["claude", "gpt5.1", "qwen"]
MODEL_COLORS   = {"claude": "#2E86AB", "gpt5.1": "#A23B72", "qwen": "#F18F01"}
MODEL_LABELS   = {"claude": "Claude", "gpt5.1": "GPT", "qwen": "Qwen"}


def fast_k(c_p, N, k):
    if c_p < 0 or N <= 0 or k <= 0 or k > N: return np.nan
    if c_p >= k: return 1.0
    if c_p == 0: return 0.0
    num = comb(N - c_p, k); den = comb(N, k)
    return np.nan if den == 0 else 1.0 - num / den


def speedup_k_exact(speeds_correct, p_thr, N, k):
    if k > N or k <= 0: return np.nan
    speeds = list(speeds_correct) + [0.0] * (N - len(speeds_correct))
    tot, cnt = 0.0, 0
    for combo in combinations(speeds, k):
        m = max(combo)
        if m >= p_thr: tot += m
        cnt += 1
    return tot / cnt if cnt else 0.0


def compute(df):
    rows = []
    threads = sorted(df['threads'].dropna().unique())
    for t in threads:
        for m in MODELS:
            sub_m = df[(df['threads'] == t) & (df['model'] == m)]
            fasts, speeds = [], []
            for bench, grp in sub_m.groupby('benchmark'):
                correct = grp[(grp['compile_status'] == 'success')
                              & (grp['correctness'] == 1.0)
                              & grp['kernel_speedup'].notna()]
                cs = correct['kernel_speedup'].tolist()
                c_p = sum(1 for s in cs if s >= P_THR)
                f = fast_k(c_p, N_TOTAL, K)
                if not np.isnan(f): fasts.append(f)
                s = speedup_k_exact(cs, P_THR, N_TOTAL, K)
                if not np.isnan(s): speeds.append(s)
            rows.append({
                'threads': int(t), 'model': m,
                'Fast_at_3': float(np.mean(fasts)) if fasts else np.nan,
                'Speedup_at_3': float(np.mean(speeds)) if speeds else np.nan,
            })
    return pd.DataFrame(rows)


def plot_dotplot(df, threads_order, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=150)
    last_points = {}
    for m in MODELS:
        sub = df[df["model"] == m].sort_values("threads")
        if not len(sub): continue
        xs = np.arange(len(threads_order)).astype(float)
        ys = sub.set_index('threads').reindex(threads_order)['Fast_at_3'].values
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

    x_max = len(threads_order) - 1
    label_x = x_max + 0.55
    for m, (x, y) in last_points.items():
        y_lab = y + label_y_offsets[m]
        ax.plot([x + 0.10, label_x - 0.08], [y, y_lab],
                color=MODEL_COLORS[m], linewidth=0.8, alpha=0.5, zorder=2)
        ax.text(label_x, y_lab, MODEL_LABELS[m],
                color=MODEL_COLORS[m], fontsize=15, fontweight="bold",
                va="center", ha="left")

    ax.set_xlim(-0.4, x_max + 1.8)
    all_ys = df["Fast_at_3"].dropna().values
    if len(all_ys):
        y_lo, y_hi = float(np.min(all_ys)), float(np.max(all_ys))
        pad = (y_hi - y_lo) * 0.18 if (y_hi - y_lo) > 0 else 0.05
        ax.set_ylim(max(0.0, y_lo - pad), min(1.05, y_hi + pad))

    ax.set_xticks(range(len(threads_order)))
    ax.set_xticklabels([str(t) for t in threads_order], fontsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_xlabel(r"OpenMP thread count $n$", fontsize=15)
    ax.set_ylabel(r"$\mathrm{Fast}@3$", fontsize=15)
    ax.grid(True, axis="x", linestyle=":", linewidth=0.6, alpha=0.45, color="gray")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.30, color="gray")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  saved: {out_path}")


def plot_stepfill(df, threads_order, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=150)
    series = {}
    for m in MODELS:
        sub = df[df['model'] == m].sort_values('threads').set_index('threads')
        ys = sub.reindex(threads_order)['Speedup_at_3'].values
        series[m] = ys

    order = sorted(series, key=lambda m: np.nanmax(series[m]) if len(series[m]) else 0)
    xs = np.arange(len(threads_order)).astype(float)

    for m in order:
        y = series[m]
        if not len(y): continue
        ax.fill_between(xs, 0, y, step="pre",
                        color=MODEL_COLORS[m], alpha=0.40, linewidth=0)
        ax.step(xs, y, where="pre",
                color=MODEL_COLORS[m], linewidth=2.2, label=MODEL_LABELS[m])

    ax.set_xlim(0, len(threads_order) - 1)
    max_y = np.nanmax([np.nanmax(s) for s in series.values()])
    ax.set_ylim(0, max_y * 1.08)
    ax.set_xticks(range(len(threads_order)))
    ax.set_xticklabels([str(t) for t in threads_order], fontsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_xlabel(r"OpenMP thread count $n$", fontsize=15)
    ax.set_ylabel(r"$\mathrm{Speedup}@3$", fontsize=15)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)

    handles_in_order = []
    for m in MODELS:
        if m not in series or not len(series[m]): continue
        handles_in_order.append(plt.Rectangle(
            (0, 0), 1, 1, fc=MODEL_COLORS[m], ec=MODEL_COLORS[m],
            alpha=0.55, label=MODEL_LABELS[m]))
    ax.legend(handles=handles_in_order, loc="upper left",
              frameon=True, fontsize=14)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  saved: {out_path}")


def main():
    df = pd.read_csv(EX12_CSV)
    df = df[(df['ex'] == 'EX2') & (df['dataset_size'] == DATASET)
            & (df['model'] != 'baseline')]
    res = compute(df)
    threads_order = sorted(res['threads'].unique().tolist())
    print(res.pivot(index='model', columns='threads',
                    values=['Fast_at_3', 'Speedup_at_3']).round(3))
    plot_dotplot(res, threads_order, OUT_DIR / "fast_at_3_vs_threads_EX2_medium_dotplot.pdf")
    plot_stepfill(res, threads_order, OUT_DIR / "speedup_at_3_vs_threads_EX2_medium_stepfill.pdf")


if __name__ == "__main__":
    main()
