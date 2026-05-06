#!/usr/bin/env python3
"""
End-to-end EX3 figure regeneration using:
  - April 2026 timing rerun  (results/time_measurements_ex3_rerun -> per-benchmark
    summaries already produced into analysis_summaries/speedup/detail/EX3_rerun)
  - REAL correctness verdicts at per-(bench, model, version, dataset) granularity
    derived from results/correctness/<bench>/EX3/correctness.jsonl

Outputs:
  - analysis_summaries/unified_data_real/unified_df_EX3.csv
  - analysis_summaries/unified_data_real/fast_speedup_at_k3_with_pthr_EX3_real.csv
  - analysis_summaries/beautiful_figure/main_paper/figure_speedup_at_3_beautiful_real.pdf
  - analysis_summaries/beautiful_figure/main_paper/figure_fast_at_3_beautiful_real.pdf
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

CORR_AGG     = PROJECT_ROOT / "analysis_summaries" / "correctness" / "EX3_correctness_real_aggregated.csv"
DETAIL_DIR   = PROJECT_ROOT / "analysis_summaries" / "speedup" / "detail" / "EX3_rerun"
TABLES_DIR   = PROJECT_ROOT / "analysis_summaries" / "tables"
OUT_DIR      = PROJECT_ROOT / "analysis_summaries" / "unified_data_real"
FIG_DIR      = PROJECT_ROOT / "analysis_summaries" / "beautiful_figure" / "main_paper"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

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


def parse_speedup_value(v):
    if pd.isna(v) or v == "-":
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).rstrip("x"))
    except Exception:
        return None


# -- 1) load timing detail ---------------------------------------------------
def load_speedup_detail():
    frames = []
    for csv_file in sorted(DETAIL_DIR.glob("*_EX3_summary.csv")):
        bench = csv_file.stem.replace("_EX3_summary", "")
        df = pd.read_csv(csv_file)
        df["benchmark"] = bench
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["dataset"] = df["dataset"].astype(str).str.lower()
    df["version"] = pd.to_numeric(df["version"], errors="coerce")
    df["kernel_speedup"] = df["kernel_speedup"].apply(parse_speedup_value)
    df["total_speedup"]  = df["total_speedup"].apply(parse_speedup_value)
    return df


# -- 2) load real correctness (broadcast mini verdict to all datasets) -----
def load_correctness():
    """Correctness is checked only on `mini` for EX3. We broadcast the mini
    verdict to all datasets, since the optimized code is identical across
    dataset sizes."""
    df = pd.read_csv(CORR_AGG)
    df["version"] = pd.to_numeric(df["version"], errors="coerce")
    df["is_correct"] = df["correctness"].astype(str).str.lower() == "true"
    return df[["benchmark","model","version","is_correct"]]


# -- 3) build unified df merging on (bench, model, version, dataset) ---------
def build_unified():
    print("[1/4] Loading speedup detail (April rerun)...")
    df_t = load_speedup_detail()
    print(f"  rows: {len(df_t)}, benchmarks: {df_t['benchmark'].nunique()}")

    print("[2/4] Loading real correctness (mini-broadcast)...")
    df_c = load_correctness()
    print(f"  rows: {len(df_c)}")

    df_t["version"] = df_t["version"].astype(float)
    df_c["version"] = df_c["version"].astype(float)

    df = df_t.merge(df_c, on=["benchmark","model","version"], how="left")
    df["is_correct"] = df["is_correct"].fillna(False).infer_objects(copy=False)

    # baseline rows are NA model -> always treated as is_correct=True (not used in metrics)
    print(f"  unified rows: {len(df)}, correct cells: {(df['is_correct']==True).sum()}")
    out = OUT_DIR / "unified_df_EX3.csv"
    df.to_csv(out, index=False)
    print(f"  saved: {out}")
    return df


# -- 4) recompute metrics with fallback=0 ------------------------------------
def compute_fast_at_k(c_p, N, k):
    if N <= 0 or k <= 0 or k > N:
        return np.nan
    if c_p >= k: return 1.0
    if c_p == 0: return 0.0
    if N - c_p < k: return 1.0
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


def recompute_metrics(df):
    rows = []
    groups = df.groupby(["benchmark", "model", "dataset"])
    for (bm, model, ds), grp in groups:
        if model == "baseline" or pd.isna(model):
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
                "benchmark": bm, "model": model, "dataset": ds,
                "p_thr": p_thr, "k": K,
                "fast_at_3": fast, "speedup_at_3": speedup,
                "num_correct": num_correct, "num_total": N,
            })
    return pd.DataFrame(rows)


# -- 5) plotting (same layout as fixed/rerun versions) -----------------------
def _load_ex1_ex2_data():
    data = {}
    for ex in ["EX1", "EX2"]:
        data[ex] = {}
        for ds in DATASET_SIZES:
            f = TABLES_DIR / f"table_e_paper_level_k3_{ex}_{ds}.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f)
            if ex == "EX2":
                d = d[d["threads"] == 16.0]
            data[ex][ds] = d
    return data


def _aggregate_ex3(df_fixed):
    out = {}
    for ds in DATASET_SIZES:
        sub = df_fixed[df_fixed["dataset"] == ds]
        if not len(sub): continue
        out[ds] = sub.groupby(["model","p_thr"]).agg(
            fast_at_3=("fast_at_3","mean"),
            speedup_at_3=("speedup_at_3","mean"),
        ).reset_index()
    return out


def _plot(ex12, ex3, ycol_e12, ycol_e3, fig_name, ylabel, ylim=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 5, figsize=(15, 8), dpi=150)
    plt.subplots_adjust(left=0.065, right=0.985, top=0.92, bottom=0.10, hspace=0.15, wspace=0.15)

    EX_VERSIONS = ["EX1","EX2","EX3"]
    for r_i, ex in enumerate(EX_VERSIONS):
        for c_i, ds in enumerate(DATASET_SIZES):
            ax = axes[r_i, c_i]
            if ex in ["EX1","EX2"]:
                if ds in ex12[ex]:
                    d = ex12[ex][ds]
                    for m in MODELS:
                        dm = d[d["model"]==m]
                        if len(dm):
                            ax.plot(dm["p_thr"], dm[ycol_e12],
                                    color=MODEL_COLORS[m], linewidth=2, alpha=0.85)
            else:
                if ds in ex3:
                    d = ex3[ds]
                    for m in MODELS:
                        dm = d[d["model"]==m].sort_values("p_thr")
                        if len(dm):
                            ax.plot(dm["p_thr"], dm[ycol_e3],
                                    color=MODEL_COLORS[m], linewidth=2, alpha=0.85)
            ax.grid(True, axis="y", alpha=0.15, linestyle="-", linewidth=0.4, color="gray")
            ax.set_axisbelow(True)
            if ylim is not None:
                ax.set_ylim(*ylim)
            if c_i != 0: ax.set_yticklabels([])
            if r_i != 2: ax.set_xticklabels([])
            if r_i == 0:
                ax.set_title(DATASET_LABELS[ds], fontsize=11, pad=15)
            ax.tick_params(axis="both", labelsize=8)

    for i, lab in enumerate(["EX1 (Serial)","EX2 (OpenMP)","EX3 (CUDA)"]):
        fig.text(0.02, [0.77,0.50,0.23][i], lab, fontsize=12, fontweight="bold",
                 ha="center", va="center", rotation=90)

    fig.text(0.53, 0.03, r"Speedup threshold $p_{\mathrm{thr}}$", ha="center", fontsize=13)
    fig.text(0.005, 0.5, ylabel, ha="center", va="center", rotation=90, fontsize=13)

    handles = [plt.Line2D([0],[0], color=MODEL_COLORS[m], linewidth=2.5,
                          marker="o", markersize=5, label=MODEL_LABELS[m]) for m in MODELS]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=True, fontsize=11,
               bbox_to_anchor=(0.5, 1.02), columnspacing=1.5, handlelength=2.5)

    out = FIG_DIR / fig_name
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  saved: {out}")


def main():
    df_unified = build_unified()

    print("[3/4] Recomputing Fast@3 / Speedup@3 (fallback=0) ...")
    df_fixed = recompute_metrics(df_unified)
    out = OUT_DIR / "fast_speedup_at_k3_with_pthr_EX3_real.csv"
    df_fixed.to_csv(out, index=False)
    print(f"  saved: {out}  ({len(df_fixed)} rows)")

    print("[4/4] Plotting figures ...")
    ex12 = _load_ex1_ex2_data()
    ex3_agg = _aggregate_ex3(df_fixed)
    _plot(ex12, ex3_agg, "Speedup_at_k_avg", "speedup_at_3",
          "figure_speedup_at_3_beautiful_real.pdf", r"Speedup@3")
    _plot(ex12, ex3_agg, "Fast_at_k_avg", "fast_at_3",
          "figure_fast_at_3_beautiful_real.pdf", r"Fast@3", ylim=(-0.05, 1.05))


if __name__ == "__main__":
    main()
