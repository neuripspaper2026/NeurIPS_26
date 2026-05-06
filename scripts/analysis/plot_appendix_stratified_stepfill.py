#!/usr/bin/env python3
"""
Generate motif- and difficulty-stratified Fast@3 / Speedup@3 figures
for the appendix, in filled step-area style (matching the main paper
Speedup@3 panel rather than the older line-plot version).

For each (EX in {EX1, EX2-16t}, metric in {Fast@3, Speedup@3}, p_thr range),
we compute per-(group, p_thr) macro-averaged values across the 3 models
and across all benchmarks belonging to the group, then draw one filled
step area per group on a single panel.

Inputs:
  - final_data/unified_results.csv  (EX1, EX2 trial-level data)
  - scripts/analysis/difficulty_scores.csv

Output (under figure/appendix_figure/):
  - fig_stratified_motif_fast_at_3_EX1_medium.pdf
  - fig_stratified_motif_speedup_at_3_EX1_medium.pdf
  - fig_stratified_motif_fast_at_3_EX2_medium.pdf
  - fig_stratified_motif_speedup_at_3_EX2_medium.pdf
  - fig_stratified_difficulty_fast_at_3_EX1_medium.pdf
  - fig_stratified_difficulty_speedup_at_3_EX1_medium.pdf
  - fig_stratified_difficulty_fast_at_3_EX2_medium.pdf
  - fig_stratified_difficulty_speedup_at_3_EX2_medium.pdf

Notes:
  - Model legend uses GPT (not GPT-5.1 / GPT-4o), per user direction.
  - This script aggregates ACROSS the 3 models (not per-model), to keep
    each figure focused on motif/difficulty stratification.
"""
from __future__ import annotations

from pathlib import Path
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EX12_CSV  = PROJECT_ROOT / "final_data" / "unified_results.csv"
DIFF_CSV  = PROJECT_ROOT / "scripts" / "analysis" / "difficulty_scores.csv"
OUT_DIR   = PROJECT_ROOT / "NeurIPS_26" / "figure" / "appendix_figure"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET = "medium"
P_THR_LIST = [0.0, 1.0, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.40, 1.50, 2.0, 3.0]
N_TOTAL = 10
K = 3
EX2_THREADS = 16

MOTIF_ABBR = {
    'dense_linear_algebra':'DLA', 'sparse_linear_algebra':'SLA',
    'stencil_computations':'SC', 'graph_algorithms':'GA',
    'dynamic_programming':'DP', 'n_body_methods':'NBM',
    'image_and_video_processing':'IVP', 'clustering_and_ml':'CML',
    'statistical_computations':'STC', 'spectral_methods':'SM',
    'computational_fluid_dynamics':'CFD', 'monte_carlo_methods':'MCM',
    'cryptography_and_encoding':'CE', 'sorting_and_searching':'SS',
    'medical_and_scientific':'MSC', 'utility':'UB',
}

# Motif colors: tab10 + extras
MOTIF_COLORS_FULL = {
    'DLA': '#1f77b4', 'SC': '#ff7f0e', 'NBM': '#d62728', 'SLA': '#9467bd',
    'GA': '#8c564b', 'DP': '#e377c2', 'CML': '#7f7f7f', 'IVP': '#17becf',
    'MCM': '#bcbd22', 'STC': '#2ca02c', 'CFD': '#393b79', 'CE': '#637939',
    'SS': '#7b4173', 'SM': '#d6616b', 'MSC': '#a55194', 'UB': '#5254a3',
}

DIFFICULTY_COLORS = {
    'd1': '#2E86AB', 'd2': '#A23B72', 'd3': '#F18F01', 'd4': '#C73E1D',
}


# ---------------- core metric helpers ----------------
def fast_k(c_p, N, k):
    if c_p < 0 or N <= 0 or k <= 0 or k > N: return np.nan
    if c_p >= k: return 1.0
    if c_p == 0: return 0.0
    num = comb(N - c_p, k); den = comb(N, k)
    return np.nan if den == 0 else 1.0 - num / den

def speedup_k_exact(speeds_gated, p, N, k):
    if N < k: return 0.0
    tot = 0.0; n = 0
    for combo in combinations(range(N), k):
        m = max(speeds_gated[i] for i in combo)
        if m >= p: tot += m
        n += 1
    return tot / n if n else 0.0


# ---------------- aggregation ----------------
def gated_per_bench(df, ex, threads=None):
    d = df[df['ex'] == ex].copy()
    if threads is not None: d = d[d['threads'] == threads]
    d = d[d['dataset_size'] == DATASET]
    d['gated'] = np.where(
        (d['compile_status'] == 'success') & (d['correctness'] == 1.0)
        & d['kernel_speedup'].notna(),
        d['kernel_speedup'], 0.0,
    )
    out = {}
    for (model, bench), grp in d.groupby(['model', 'benchmark']):
        if model == 'baseline': continue
        v = grp['gated'].tolist()
        if len(v) > N_TOTAL: v = v[:N_TOTAL]
        elif len(v) < N_TOTAL: v += [0.0] * (N_TOTAL - len(v))
        out[(model, bench)] = v
    return out


def per_group_metrics(gated, group_of_bench):
    """For each group, compute per-(p_thr) Fast@3 and Speedup@3 macro-averaged
    across (model, benchmark) within that group."""
    metrics = {}
    for (model, bench), v in gated.items():
        g = group_of_bench.get(bench)
        if g is None: continue
        for p in P_THR_LIST:
            c_p = sum(1 for x in v if x >= p)
            f = fast_k(c_p, N_TOTAL, K)
            s = speedup_k_exact(v, p, N_TOTAL, K)
            metrics.setdefault((g, p), {'fast': [], 'speed': []})
            if not np.isnan(f): metrics[(g, p)]['fast'].append(f)
            metrics[(g, p)]['speed'].append(s)
    rows = []
    for (g, p), d in metrics.items():
        rows.append({
            'group': g, 'p_thr': p,
            'fast_at_3': np.mean(d['fast']) if d['fast'] else np.nan,
            'speedup_at_3': np.mean(d['speed']) if d['speed'] else np.nan,
            'n_entries': len(d['speed']),
        })
    return pd.DataFrame(rows)


def plot_stepfill(df_metrics, group_order, group_colors, ycol, ylabel,
                  out_path, ylim=None):
    """Filled step-area plot, one area per group, drawn smallest-on-top."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # rank groups by total area (so smaller drawn last on top)
    auc = {}
    for g in group_order:
        sub = df_metrics[df_metrics['group'] == g].sort_values('p_thr')
        if len(sub):
            auc[g] = float(sub[ycol].sum())
        else:
            auc[g] = -1.0
    sorted_for_draw = sorted([g for g in group_order if auc[g] >= 0],
                             key=lambda g: -auc[g])  # large first → small on top

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    for g in sorted_for_draw:
        sub = df_metrics[df_metrics['group'] == g].sort_values('p_thr')
        if not len(sub): continue
        xs = sub['p_thr'].values.astype(float)
        ys = sub[ycol].values.astype(float)
        c = group_colors.get(g, '#888888')
        ax.fill_between(xs, 0, ys, step='pre',
                        color=c, alpha=0.30, linewidth=0)
        ax.step(xs, ys, where='pre',
                color=c, linewidth=2.0, alpha=0.95, label=g)

    ax.set_xlabel(r"Speedup threshold $p_{\mathrm{thr}}$", fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.tick_params(axis='both', labelsize=13)
    ax.grid(True, axis='y', alpha=0.25, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    if ylim is not None:
        ax.set_ylim(*ylim)

    # Legend in original group_order (lexicographic / canonical), not draw order
    handles_in_order = []
    for g in group_order:
        if g not in auc or auc[g] < 0: continue
        c = group_colors.get(g, '#888888')
        handles_in_order.append(plt.Rectangle(
            (0, 0), 1, 1, fc=c, ec=c, alpha=0.55, label=g))
    if handles_in_order:
        ncol = 2 if len(handles_in_order) > 6 else 1
        ax.legend(handles=handles_in_order, loc='upper right',
                  frameon=True, fontsize=14, ncol=ncol)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  saved: {out_path}")


def pick_representative_motifs(df_motif_metrics, k=7):
    """Pick a representative subset of motifs spanning the speedup range.
    Sort by Speedup@3 at p_thr=1.0 descending, take top k. This keeps the
    plot legible while showing the high/medium/low ends."""
    sub = df_motif_metrics[df_motif_metrics['p_thr'] == 1.0].copy()
    sub = sub.sort_values('speedup_at_3', ascending=False)
    return sub['group'].tolist()[:k]


def main():
    print("[1/4] Loading data ...")
    df12 = pd.read_csv(EX12_CSV)
    diff = pd.read_csv(DIFF_CSV)[['benchmark', 'bucket']]
    motif_lookup = df12[['benchmark', 'motif']].drop_duplicates()
    motif_lookup = dict(zip(motif_lookup['benchmark'],
                             motif_lookup['motif'].map(MOTIF_ABBR).fillna('?')))
    diff_lookup = dict(zip(diff['benchmark'], diff['bucket']))

    for ex_label, ex, threads in [('EX1', 'EX1', None),
                                   ('EX2', 'EX2', float(EX2_THREADS))]:
        print(f"[2/4] Computing gated vectors for {ex_label} ...")
        gated = gated_per_bench(df12, ex, threads=threads)

        # Motif stratification (filter to top-7 motifs by Speedup@3 at p_thr=1.0)
        df_motif = per_group_metrics(gated, motif_lookup)
        top_motifs = pick_representative_motifs(df_motif, k=7)
        df_motif_top = df_motif[df_motif['group'].isin(top_motifs)].copy()
        # Difficulty stratification (4 buckets, all)
        df_diff = per_group_metrics(gated, diff_lookup)

        for (df_g, group_order, group_colors, prefix) in [
            (df_motif_top, top_motifs, MOTIF_COLORS_FULL, 'motif'),
            (df_diff,  ['d1', 'd2', 'd3', 'd4'], DIFFICULTY_COLORS, 'difficulty'),
        ]:
            for ycol, ylab, lim, suff in [
                ('fast_at_3', r"$\mathrm{Fast}@3$", (-0.02, 1.02), 'fast'),
                ('speedup_at_3', r"$\mathrm{Speedup}@3$", None, 'speedup'),
            ]:
                fname = f"fig_stratified_{prefix}_{suff}_at_3_{ex_label}_{DATASET}.pdf"
                plot_stepfill(df_g, group_order, group_colors, ycol, ylab,
                              OUT_DIR / fname, ylim=lim)


if __name__ == '__main__':
    main()
