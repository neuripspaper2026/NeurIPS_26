#!/usr/bin/env python3
"""
Diagnostic: per-benchmark best-of-N kernel speedup distribution for
EX1 (Serial) vs EX2 (OpenMP, 16t) on the medium dataset.

Used to verify the claim that EX2's distribution is bimodal (a large
mass near zero from sub-millisecond OpenMP-overhead-dominated kernels,
plus a long high tail from a small set of highly parallelizable kernels)
while EX1 is more unimodal (most benchmarks land in the 0.95-1.5x band).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EX12_CSV = PROJECT_ROOT / "final_data" / "unified_results.csv"
FIG_DIR  = PROJECT_ROOT / "analysis_summaries" / "beautiful_figure" / "main_paper"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_COLORS = {"claude": "#2E86AB", "gpt5.1": "#A23B72", "qwen": "#F18F01"}
MODEL_LABELS = {"claude": "Claude", "gpt5.1": "GPT", "qwen": "Qwen"}
MODELS = ["claude", "gpt5.1", "qwen"]


def best_per_bench(df, ex, threads=None, dataset="medium"):
    d = df[df["ex"] == ex]
    if threads is not None: d = d[d["threads"] == threads]
    d = d[d["dataset_size"] == dataset]
    out = {m: [] for m in MODELS}
    for (model, bench), grp in d.groupby(["model", "benchmark"]):
        if model == "baseline": continue
        v = [(r["kernel_speedup"]
              if (r["compile_status"] == "success"
                  and r["correctness"] == 1.0
                  and pd.notna(r["kernel_speedup"]))
              else 0.0)
             for _, r in grp.iterrows()]
        if model in out:
            out[model].append(max(v) if v else 0.0)
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df12 = pd.read_csv(EX12_CSV)
    ex1 = best_per_bench(df12, "EX1")
    ex2 = best_per_bench(df12, "EX2", threads=16.0)

    # Bin edges chosen to expose the bimodality:
    # tight near 1.0 (no-gain band), wider for the high tail
    edges = [0.0, 0.1, 0.5, 0.95, 1.05, 1.5, 2.0, 5.0, 10.0, 20.0, 60.0]
    centers = [(edges[i] + edges[i+1])/2 for i in range(len(edges)-1)]
    width  = [edges[i+1] - edges[i] for i in range(len(edges)-1)]

    fig, axes = plt.subplots(2, 1, figsize=(9, 6.2), dpi=150, sharex=True)
    plt.subplots_adjust(left=0.10, right=0.985, top=0.92, bottom=0.10, hspace=0.30)

    for ax, (ex_name, ex_data, title) in zip(
        axes,
        [("EX1", ex1, "EX1 (Serial CPU): per-benchmark best-of-N kernel speedup distribution"),
         ("EX2", ex2, "EX2 (OpenMP, 16 threads): per-benchmark best-of-N kernel speedup distribution")]
    ):
        n_models = len(MODELS)
        bar_w = 0.27
        for j, m in enumerate(MODELS):
            vals = np.array(ex_data[m])
            counts, _ = np.histogram(vals, bins=edges)
            offsets = (j - 1) * bar_w
            xs = np.arange(len(edges) - 1) + offsets
            ax.bar(xs, counts, width=bar_w, color=MODEL_COLORS[m],
                   edgecolor='white', linewidth=0.5,
                   label=MODEL_LABELS[m], alpha=0.92)

        ax.set_xticks(np.arange(len(edges) - 1))
        ax.set_xticklabels([f"{edges[i]:g}-{edges[i+1]:g}" for i in range(len(edges)-1)],
                           rotation=30, ha='right', fontsize=9)
        ax.set_ylabel("# benchmarks", fontsize=11)
        ax.set_title(title, fontsize=11, pad=6, loc='left')
        ax.grid(True, axis="y", alpha=0.18, linestyle="-",
                linewidth=0.4, color="gray")
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=9)

        # Shade the "no-gain band" lightly
        no_gain_idx = edges.index(0.95)
        ax.axvspan(no_gain_idx - 0.5, no_gain_idx + 0.5,
                   color="gray", alpha=0.10, zorder=0)

    axes[1].set_xlabel(r"Best-of-N kernel speedup bucket ($\times$ over baseline)", fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, frameon=True,
               fontsize=10, bbox_to_anchor=(0.985, 0.99))

    out = FIG_DIR / "figure_best_of_n_distribution_ex1_vs_ex2.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
