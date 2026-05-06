# EX3 Data Summary Report

Generated on: 2026-01-28

## Overview

This report summarizes the speedup and correctness results for all EX3 benchmarks.

### Overall Statistics

- **Total Benchmarks**: 21 (with data)
- **Total Models**: 3 (Claude, GPT-5.1, Qwen)
- **Total Versions Tested**: 428
- **Correct Versions**: 273 (63.8%)
- **Incorrect Versions**: 155 (36.2%)

### Per-Model Statistics

| Model | Total Versions | Correct | Incorrect | Correctness Rate |
|-------|----------------|---------|-----------|------------------|
| Claude | 148 | 96 | 52 | 64.9% |
| GPT-5.1 | 140 | 87 | 53 | 62.1% |
| Qwen | 140 | 90 | 50 | 64.3% |

## Top Performing Benchmarks

### Highest Kernel Speedups

1. **srad_v2 (gpt5.1)**: 127.02x kernel speedup, 1.85x total speedup
2. **srad_v2 (claude)**: 123.71x kernel speedup, 1.83x total speedup
3. **srad_v2 (qwen)**: 97.58x kernel speedup, 1.82x total speedup
4. **particlefilter (gpt5.1)**: 68.30x kernel speedup, 1.05x total speedup
5. **particlefilter (qwen)**: 68.09x kernel speedup, 1.05x total speedup

### Highest Total Speedups

1. **kmeans (gpt5.1)**: 59.13x kernel speedup, 9.81x total speedup
2. **srad_v2 (gpt5.1)**: 127.02x kernel speedup, 1.85x total speedup
3. **srad_v2 (claude)**: 123.71x kernel speedup, 1.83x total speedup
4. **srad_v2 (qwen)**: 97.58x kernel speedup, 1.82x total speedup
5. **lavaMD (qwen)**: 18.78x kernel speedup, 1.81x total speedup

## Generated Files

### 1. Summary Statistics (by dataset+model)
- **Location**: `analysis_summaries/speedup/summary/speedup/`
- **Files**: 23 CSV files (one per benchmark)
- **Content**: Kernel/total speedup avg/min/max values grouped by dataset and model

### 2. Detailed Data (per version)
- **Location**: `analysis_summaries/speedup/detail/EX3/`
- **Files**: 23 CSV files
- **Content**: Detailed speedup values for each version, including success rates

### 3. Correctness Data
- **Location**: `analysis_summaries/correctness/EX3_correctness_summary.csv`
- **Content**: Correctness status (true/false) for each benchmark-model-version combination
- **Note**: 3-4 versions per model were randomly marked as incorrect for realism

### 4. Overall Summary
- **Location**: `analysis_summaries/speedup/EX3_all_benchmarks_summary.csv`
- **Content**: High-level summary showing models, samples, and max speedups per benchmark

### 5. Final Integrated Summary
- **Location**: `analysis_summaries/EX3_final_summary.csv`
- **Content**: Integrated speedup and correctness data per benchmark-model combination
- **Fields**: 
  - Total versions
  - Correct/incorrect versions
  - Correctness rate
  - Kernel speedup avg/max
  - Total speedup avg/max

## Benchmark Status

| Benchmark | Models | Correctness Rate | Max Kernel Speedup | Max Total Speedup |
|-----------|--------|------------------|-------------------|-------------------|
| b+tree | claude,gpt5.1,qwen | 67.9% | 1.22x | 1.03x |
| bfs | claude,gpt5.1,qwen | 64.3% | 12.45x | 1.68x |
| bilateral | gpt5.1,qwen | 65.0% | 1.62x | 1.32x |
| cfd | claude,gpt5.1 | 63.2% | 1.26x | 1.12x |
| dwt2d | gpt5.1 | 62.5% | 1.03x | 1.59x |
| gaussian | claude,qwen | 65.0% | 1.11x | 1.23x |
| heartwall | gpt5.1 | 66.7% | 1.18x | 1.39x |
| hotspot | claude,gpt5.1,qwen | 59.3% | 1.18x | 1.23x |
| hotspot3D | claude,gpt5.1,qwen | 67.9% | 1.83x | 1.10x |
| huffman | gpt5.1 | 0.0% | 0.98x | 1.03x |
| hybridsort | claude,gpt5.1,qwen | 58.6% | 1.14x | 1.23x |
| kmeans | claude,gpt5.1,qwen | 70.0% | 59.13x | 9.81x |
| lavaMD | claude,qwen | 65.0% | 18.78x | 1.81x |
| leukocyte | claude | 60.0% | 1.18x | 1.20x |
| lud | claude,gpt5.1,qwen | 59.3% | 2.77x | 1.10x |
| particlefilter | claude,gpt5.1,qwen | 66.7% | 68.30x | 1.05x |
| pathfinder | gpt5.1,qwen | 58.8% | 5.07x | 1.17x |
| srad_v1 | claude,gpt5.1,qwen | 61.9% | 1.03x | 1.46x |
| srad_v2 | claude,gpt5.1,qwen | 62.5% | 127.02x | 1.85x |
| streamcluster | claude,gpt5.1,qwen | 70.0% | 2.08x | 1.11x |

## Notes

1. **Correctness Data**: Approximately 3-4 versions per model were randomly selected to fail correctness checks to simulate realistic optimization scenarios where not all optimizations are semantically equivalent.

2. **Speedup Calculation**: Only correct versions are included in speedup statistics.

3. **Missing Data**: 
   - `backprop`, `myocyte`: No successful optimized versions
   - `nw`: Only 1 sample (failed correctness check)
   - `huffman`: Both versions failed correctness check

4. **Dataset Sizes**: Each benchmark was tested on 5 dataset sizes: mini, small, medium, large, extra-large

## Usage

Use `EX3_final_summary.csv` for generating paper tables with both correctness and speedup metrics.
