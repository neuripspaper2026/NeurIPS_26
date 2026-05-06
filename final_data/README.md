# Unified HPC-Bench Evaluation Results

This directory contains the unified DataFrame aggregating all evaluation results from HPC-Bench experiments.

## Files

- `unified_results.csv`: Main unified DataFrame with all evaluation data

## Data Schema

### Tidy/Long-Form Structure

Each row represents: **one benchmark + one EX version + one model + one trial + one dataset size + (optional) thread count**

### Columns (26 total)

#### Core Identifiers (8 columns)
- `benchmark`: Benchmark name (e.g., "2mm", "nw", "backprop")
- `ex`: EX version indicating parallelization strategy ("EX1", "EX2")
- `model`: LLM model name ("claude", "gpt5.1", "qwen")
- `trial_id`: Trial version number (1-10)
- `dataset_size`: Input dataset size ("mini", "small", "medium", "large", "extra-large")
- `compiler`: Compiler used ("gcc", "nvcc")
- `threads`: Thread count for EX2+ (1, 2, 8, 16, 32; NaN for EX1)
- `suite`: Benchmark suite source ("polybench", "rodinia", "codee", "machsuite", "parboil", "mibench", "parsec")

#### Metadata Attributes (2 columns)
- `difficulty`: Difficulty level ("d1", "d2", "d3", "d4")
- `motif`: Computational motif ("dense_linear_algebra", "graph_algorithms", etc.)

#### Compilation Status (1 column)
- `compile_status`: Compilation result ("success" or "NC" for Not Compiled)
  - "success": Compilation succeeded, runtime data available
  - "NC": Compilation failed, no runtime data (correctness=NaN)

#### Correctness Metrics (2 columns)
- `correctness`: 1=pass, 0=fail, NaN=not compiled
- `correctness_tested`: Whether explicitly tested (True) or broadcasted (False)

#### Baseline Runtime (2 columns)
- `baseline_kernel_mean_s`: Baseline kernel runtime in seconds
- `baseline_total_mean_s`: Baseline total runtime in seconds

#### Optimized Runtime (4 columns)
- `kernel_mean_s`: Optimized kernel runtime in seconds (NaN if compile failed)
- `total_mean_s`: Optimized total runtime in seconds (NaN if compile failed)
- `kernel_speedup`: baseline_kernel_mean_s / kernel_mean_s
- `total_speedup`: baseline_total_mean_s / total_mean_s

#### Experimental Metadata (7 columns)
- `warmup_runs`: Number of warmup runs
- `measure_runs`: Number of measurement runs
- `hint_key`: Runtime hint key used
- `success_rate`: Success rate from time measurement
- `checker`: Checker used for correctness validation
- `correctness_message`: Detailed correctness message
- `executable_path`: Path to compiled executable

## Data Statistics

- **Total rows**: 73,800
- **Benchmarks**: 92
- **EX versions**: 2 (EX1, EX2)
- **Models**: 3 (claude, gpt5.1, qwen)
- **Dataset sizes**: 5 (mini, small, medium, large, extra-large)
- **Trial IDs**: 10 per model (1-10)
- **Compilation success rate**: 77.86%
- **Compilation failures (NC)**: 22.14%
- **Overall pass rate (of compiled)**: 74.89%
- **Median kernel speedup**: 1.002x

## Data Coverage

### Cartesian Product Structure
- The DataFrame contains the **full Cartesian product** of all possible combinations:
  - 92 benchmarks × 3 models × 10 trials × 5 dataset sizes = 13,800 rows (EX1)
  - 92 benchmarks × 3 models × 10 trials × 5 dataset sizes × 5 thread counts = 69,000 rows (EX2, but only 80 benchmarks have data = 60,000 rows)
  - Total: 73,800 rows

### Compilation Status
- **Success**: 57,462 rows (77.86%)
  - Have runtime data (`kernel_mean_s`, `total_mean_s`)
  - May or may not have correctness data
- **NC (Not Compiled)**: 16,338 rows (22.14%)
  - No runtime data
  - `correctness` = NaN
  - Only metadata and baseline values are present

### Compilation Success by Model
- **Claude**: 83.62% (20,570 / 24,600)
- **GPT5.1**: 75.98% (18,692 / 24,600)
- **Qwen**: 73.98% (18,200 / 24,600)

### Compilation Success by EX
- **EX1**: 88.40% (12,199 / 13,800)
- **EX2**: 75.44% (45,263 / 60,000)

### Correctness Data (for successfully compiled trials)
- **Rows with correctness data**: ~45% of compiled trials
- **Correctness broadcast mechanism**: When a benchmark passes for `mini` dataset, the result is broadcasted to all other dataset sizes
- `correctness_tested` flag distinguishes explicitly tested (True) vs broadcasted (False) results

## Usage Examples

### Load the data
```python
import pandas as pd
df = pd.read_csv('final_data/unified_results.csv')
```

### Calculate pass rate by model and EX (excluding NC)
```python
compiled_df = df[df['compile_status'] == 'success']
pass_rate = compiled_df.groupby(['model', 'ex'])['correctness'].mean()
print(pass_rate)
```

### Calculate compilation success rate by model
```python
compile_rate = df.groupby('model')['compile_status'].apply(lambda x: (x == 'success').mean())
print(compile_rate)
```

### Fast@k analysis (correct optimizations only)
```python
correct_df = df[(df['correctness'] == 1) & (df['compile_status'] == 'success')]
speedup_stats = correct_df.groupby(['model', 'ex'])['kernel_speedup'].agg(['mean', 'median', 'count'])
print(speedup_stats)
```

### Stratified analysis by difficulty (excluding NC)
```python
compiled_df = df[df['compile_status'] == 'success']
by_difficulty = compiled_df.groupby(['difficulty', 'model'])['correctness'].mean().unstack()
print(by_difficulty)
```

### Trial-level variance
```python
variance = df.groupby(['benchmark', 'model', 'ex'])['kernel_speedup'].std()
print(variance.describe())
```

## Data Quality Notes

1. **Full Cartesian Product**: The DataFrame now contains **all possible combinations** of benchmark, model, trial, dataset, and threads (for EX2). This means:
   - Rows with `compile_status='NC'` indicate compilation failures
   - No missing rows - every expected combination is present
   - Easy to identify NC cases: just filter by `compile_status == 'NC'`

2. **NC (Not Compiled) rows**: These rows have:
   - `compile_status = 'NC'`
   - `correctness = NaN` (cannot test correctness if not compiled)
   - `kernel_mean_s`, `total_mean_s`, `kernel_speedup`, `total_speedup` = NaN
   - Baseline and metadata fields are still populated

3. **Missing baselines**: Some NC rows may have NaN baseline values if:
   - The baseline itself failed to compile
   - The specific dataset size was not measured for the baseline

4. **Extreme speedup values**: Some rows show speedups > 1000x due to:
   - Data quality issues in source CSV files (e.g., incorrect baseline measurements)
   - These outliers should be filtered when calculating aggregate statistics
   - Recommended: Filter speedups to reasonable range (e.g., 0.01x to 100x) for analysis

## Generation

Generated by: `scripts/analysis/build_unified_dataframe.py`

Command:
```bash
python scripts/analysis/build_unified_dataframe.py --root ${REPO_ROOT} --output final_data/unified_results.csv --ex-versions EX1 EX2
```

## Citation

If you use this data, please cite the HPC-Bench paper:

```bibtex
@inproceedings{hpcbench2026,
  title={HPC-Bench: A Comprehensive Benchmark for High-Performance Computing Optimization},
  author={...},
  booktitle={ICLR},
  year={2026}
}
```

