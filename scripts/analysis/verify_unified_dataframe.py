#!/usr/bin/env python3
"""
Quick verification and example usage of the unified HPC-Bench DataFrame.

This script demonstrates common analysis patterns on the unified results.
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Load the unified DataFrame
PROJECT_ROOT = Path(__file__).resolve().parents[2]
df = pd.read_csv(PROJECT_ROOT / "final_data" / "unified_results.csv")

print("="*80)
print("HPC-BENCH UNIFIED DATAFRAME - VERIFICATION & EXAMPLES")
print("="*80)

print("\n1. BASIC STATISTICS")
print("-"*80)
print(f"Total records: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

print("\n2. PASS RATE BY MODEL AND EX")
print("-"*80)
pass_rate = df.groupby(['model', 'ex'])['correctness'].agg(['mean', 'count'])
pass_rate.columns = ['pass_rate', 'total_tests']
pass_rate['pass_rate'] = pass_rate['pass_rate'].apply(lambda x: f"{x:.2%}")
print(pass_rate)

print("\n3. PASS RATE BY DIFFICULTY LEVEL")
print("-"*80)
difficulty_pass = df.groupby(['difficulty', 'ex'])['correctness'].mean().unstack()
difficulty_pass = difficulty_pass.applymap(lambda x: f"{x:.2%}")
print(difficulty_pass)

print("\n4. SPEEDUP STATISTICS (correct optimizations only, filtering outliers)")
print("-"*80)
correct_df = df[(df['correctness'] == 1) & 
                (df['kernel_speedup'].notna()) & 
                (df['kernel_speedup'] > 0.01) & 
                (df['kernel_speedup'] < 100)]
speedup_stats = correct_df.groupby(['model', 'ex'])['kernel_speedup'].agg([
    ('count', 'count'),
    ('mean', 'mean'),
    ('median', 'median'),
    ('p25', lambda x: x.quantile(0.25)),
    ('p75', lambda x: x.quantile(0.75)),
])
speedup_stats = speedup_stats.round(3)
print(speedup_stats)

print("\n5. TOP 5 BENCHMARKS BY MEDIAN SPEEDUP (EX1)")
print("-"*80)
ex1_correct = df[(df['ex'] == 'EX1') & 
                 (df['correctness'] == 1) & 
                 (df['kernel_speedup'].notna()) &
                 (df['kernel_speedup'] > 0.01) & 
                 (df['kernel_speedup'] < 100)]
top_benchmarks = ex1_correct.groupby('benchmark')['kernel_speedup'].agg([
    'median', 'count'
]).sort_values('median', ascending=False).head(5)
top_benchmarks['median'] = top_benchmarks['median'].apply(lambda x: f"{x:.3f}x")
print(top_benchmarks)

print("\n6. COMPILATION SUCCESS RATE (has runtime data)")
print("-"*80)
compile_success = df.groupby(['model', 'ex']).apply(
    lambda g: (g['kernel_mean_s'].notna().sum() / len(g))
).unstack()
compile_success = compile_success.applymap(lambda x: f"{x:.2%}")
print(compile_success)

print("\n7. THREAD SCALING ANALYSIS (EX2 only)")
print("-"*80)
ex2_with_speedup = df[(df['ex'] == 'EX2') & 
                      (df['correctness'] == 1) & 
                      (df['kernel_speedup'].notna()) &
                      (df['kernel_speedup'] > 0.01) & 
                      (df['kernel_speedup'] < 100)]
if len(ex2_with_speedup) > 0:
    thread_scaling = ex2_with_speedup.groupby('threads')['kernel_speedup'].agg([
        'count', 'mean', 'median'
    ]).round(3)
    print(thread_scaling)
else:
    print("No valid EX2 speedup data available")

print("\n8. MOTIF-BASED ANALYSIS")
print("-"*80)
motif_stats = df.groupby('motif')['correctness'].agg(['mean', 'count'])
motif_stats = motif_stats.sort_values('mean', ascending=False).head(10)
motif_stats.columns = ['pass_rate', 'total_tests']
motif_stats['pass_rate'] = motif_stats['pass_rate'].apply(lambda x: f"{x:.2%}")
print(motif_stats)

print("\n" + "="*80)
print("✓ All analysis examples completed successfully!")
print("="*80)

