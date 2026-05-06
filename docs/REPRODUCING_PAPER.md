# Reproducing the Paper

This document maps each figure and table in the paper to the exact command(s) needed to regenerate it from the artifacts shipped in this repository.

## Prerequisites

* `tools/setup_env.sh` has been sourced.
* `tools/setup_input_data.sh` has populated `EX{1,2,3}/<bench>/input_data/` (use `--with-large` for full reproduction).
* Python dependencies installed via `pip install -r requirements.txt`.
* GCC ≥ 14 and (for EX3) CUDA ≥ 12.4 available.

The pre-computed results in `results/` reflect the exact runs used in the paper. Running the analysis scripts below against `results/` regenerates every figure/table without re-running the (expensive) measurement step.

## Quick path: regenerate every figure from existing data

```bash
source tools/setup_env.sh
bash scripts/analysis/run_ex3_analysis.sh
bash scripts/analysis/generate_all_figures.sh
```

Output figures land in `analysis_summaries/beautiful_figure/`, `analysis_summaries/figures/`, and `analysis_summaries/tables/`.

## Per-figure commands

> The script names below are stable; if a script needs additional CLI flags, run it with `-h` or `--help`.

### Main paper

| Figure / Table | Script |
|---|---|
| Fast@k vs k (sensitivity) | `scripts/analysis/plot_k_sensitivity.py` |
| Combined Fast@3 / Speedup@3 across EX1–EX3 | `scripts/analysis/plot_combined_main_figure.py` |
| Best-of-n distribution (EX1 vs EX2) | `scripts/analysis/plot_best_of_n_distribution.py` |
| Speedup@k vs k step-fill | `scripts/analysis/plot_appendix_stratified_stepfill.py` |
| Per-difficulty breakdown | `scripts/analysis/plot_difficulty_figures.py` |
| Per-motif breakdown | `scripts/analysis/plot_motif_figures.py` |
| EX3 unified table | `scripts/analysis/generate_table_a_ex3.py` |

### Appendix

| Figure / Table | Script |
|---|---|
| K-sensitivity, EX2/EX3 | `scripts/analysis/plot_appendix_k_sensitivity_ex2_ex3.py` |
| Thread-count sensitivity (EX2) | `scripts/analysis/plot_appendix_thread_sensitivity.py` |
| Total runtime breakdown | `scripts/analysis/plot_total_time_figures.py` |
| Error category summary | `scripts/analysis/generate_combined_error_summary.py` |

## Re-running raw measurements

If you want to re-run the full measurement pipeline (slow; takes hours on an A100 / 32-core EPYC):

```bash
# 1) LLM generation (requires API keys)
for ex in EX1 EX2 EX3; do
  for family in anthropic openai together; do
    tools/run_auto_run.sh "$ex" all "$family" "$model" "$abbrev"
  done
done

# 2) Build all variants
for ex in EX1 EX2 EX3; do
  python scripts/arg.py benchmark build --root . --ex-versions "$ex" --benchmarks all
done

# 3) Reference outputs
for ex in EX1 EX2 EX3; do
  python scripts/arg.py benchmark correctness-run --root . --ex-versions "$ex" \
    --benchmarks all --models baseline,claude,gpt5.1,qwen --datasets mini,small,medium,large
done

# 4) Correctness verification
for ex in EX1 EX2 EX3; do
  tools/run_correctness.sh "$ex"
done

# 5) Time measurement
for ex in EX1 EX2 EX3; do
  for bench in $(python scripts/arg.py benchmark list --benchmarks all --ex-versions "$ex" | tail -n +2); do
    tools/run_time_measurement.sh "$ex" "$bench"
  done
done

# 6) Re-run the analysis pipeline as in the "Quick path" above.
```

## Aggregated CSV inputs

The analysis scripts read from these intermediate CSVs (all regenerable from `results/`):

* `analysis_summaries/unified_data/*.csv` — per-paper unified time/correctness joins
* `analysis_summaries/speedup/detail/*.csv` — per-benchmark speedup tables
* `analysis_summaries/correctness/*.csv` — per-benchmark correctness summaries
* `final_data/unified_results.csv` — the master merge

To regenerate these from scratch:

```bash
python scripts/analysis/build_unified_dataframe.py
python scripts/analysis/build_unified_dataframe_ex3.py
```
