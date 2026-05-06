#!/usr/bin/env python3
"""
生成 EX1/EX2/EX3 的错误统计表格

统计每个 EX 版本在 mini dataset 上的：
- NC (Not Compiled)
- Runtime Error
- Incorrect (编译运行成功但输出错误)

Usage:
    python scripts/analysis/generate_error_summary_table.py
"""
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ==================== 配置 ====================

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = WORKSPACE_ROOT / "results" / "time_measurements"
OUTPUT_DIR = WORKSPACE_ROOT / "analysis_summaries" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EX_VERSIONS = ['EX1', 'EX2', 'EX3']
MODELS = ['claude', 'gpt5.1', 'qwen']
DATASET = 'mini'  # 只统计 mini dataset

# ==================== 数据加载和统计 ====================

def load_time_measurements(benchmark_dir: Path, ex_version: str):
    """加载一个 benchmark 的时间测量数据"""
    jsonl_file = benchmark_dir / f"{ex_version}_time_measurements.jsonl"
    
    if not jsonl_file.exists():
        return []
    
    # 从目录名获取 benchmark 名称
    benchmark_name = benchmark_dir.name
    
    records = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                # 添加 benchmark 字段以便后续匹配
                record['benchmark'] = benchmark_name
                records.append(record)
            except:
                continue
    
    return records


def categorize_error(record):
    """
    将记录分类为：NC, Runtime Error, Incorrect, Correct
    
    根据 status 字段：
    - compile_error -> NC
    - runtime_error -> Runtime Error
    - timing_parse_error -> Runtime Error (无法解析时间说明运行有问题)
    - ok -> 检查 correctness
    """
    status = record.get('status', '')
    
    if status == 'compile_error':
        return 'NC'
    elif status in ['runtime_error', 'timing_parse_error']:
        return 'Runtime Error'
    elif status == 'ok':
        # 需要检查 correctness（但这里时间测量没有 correctness 信息）
        # 我们假设 ok 状态的是运行成功的，incorrect 需要从 correctness 数据获取
        return 'Success'
    else:
        return 'Unknown'


def count_errors_for_ex(ex_version: str, dataset: str = 'mini', correctness_map=None):
    """统计一个 EX 版本在指定 dataset 上的错误"""
    
    # 发现所有 benchmarks
    benchmarks = []
    for bm_dir in sorted(RESULTS_ROOT.iterdir()):
        if bm_dir.is_dir():
            benchmarks.append(bm_dir.name)
    
    # 统计每个模型的错误
    model_stats = defaultdict(lambda: defaultdict(int))
    
    for benchmark in benchmarks:
        benchmark_dir = RESULTS_ROOT / benchmark
        records = load_time_measurements(benchmark_dir, ex_version)
        
        for record in records:
            # 只统计 summary 类型的记录
            if record.get('record_type') != 'summary':
                continue
            
            # 只统计指定的 dataset
            if record.get('dataset', '').lower() != dataset.lower():
                continue
            
            model = record.get('model', '')
            if model not in MODELS and model != 'baseline':
                continue
            
            # 跳过 baseline
            if model == 'baseline':
                continue
            
            # 分类错误
            error_type = categorize_error(record)
            
            # 如果运行成功，检查 correctness
            if error_type == 'Success' and correctness_map:
                version = record.get('version')
                key = (benchmark, model, version)
                
                # 查询 correctness
                if key in correctness_map:
                    if correctness_map[key] == False:  # incorrect
                        error_type = 'Incorrect'
                    else:  # correct
                        error_type = 'Correct'
                else:
                    # 没有 correctness 信息，保持为 Success
                    error_type = 'Correct'
            
            model_stats[model][error_type] += 1
    
    return model_stats


def load_correctness_data(ex_version: str):
    """加载 correctness 数据，返回 (benchmark, model, version) -> is_correct 的映射"""
    if ex_version == 'EX3':
        corr_file = WORKSPACE_ROOT / "analysis_summaries" / "correctness" / "EX3_correctness_summary.csv"
    else:
        # EX1/EX2 的 correctness 在不同位置
        corr_file = WORKSPACE_ROOT / "analysis_summaries" / "correctness" / f"{ex_version}_correctness_summary.csv"
    
    if not corr_file.exists():
        print(f"  [WARN] Correctness file not found: {corr_file}")
        return None
    
    df = pd.read_csv(corr_file)
    
    # 创建 (benchmark, model, version) -> is_correct 的映射
    correctness_map = {}
    
    for _, row in df.iterrows():
        benchmark = row.get('benchmark', '')
        model = row.get('model', '')
        version = row.get('version')
        correctness = row.get('correctness', 'false')
        
        # 处理不同的 correctness 表示方式
        if isinstance(correctness, str):
            is_correct = correctness.lower() == 'true'
        else:
            is_correct = bool(correctness)
        
        key = (benchmark, model, version)
        correctness_map[key] = is_correct
    
    return correctness_map


# ==================== 主函数 ====================

def main():
    print("\n" + "="*80)
    print("Generating Error Summary Table for EX1/EX2/EX3 (mini dataset)")
    print("="*80 + "\n")
    
    # 收集所有数据
    all_stats = []
    
    for ex in EX_VERSIONS:
        print(f"Processing {ex}...")
        
        # 加载 correctness 映射
        correctness_map = load_correctness_data(ex)
        
        # 从时间测量获取所有统计（包括 correctness 关联）
        model_stats = count_errors_for_ex(ex, dataset=DATASET, correctness_map=correctness_map)
        
        # 整合数据
        for model in MODELS:
            nc_count = model_stats[model].get('NC', 0)
            runtime_error_count = model_stats[model].get('Runtime Error', 0)
            incorrect_count = model_stats[model].get('Incorrect', 0)
            correct_count = model_stats[model].get('Correct', 0)
            
            all_stats.append({
                'EX Version': ex,
                'Model': model,
                'Dataset': DATASET,
                'NC': nc_count,
                'Runtime Error': runtime_error_count,
                'Incorrect': incorrect_count,
                'Correct': correct_count,
                'Total': nc_count + runtime_error_count + incorrect_count + correct_count
            })
        
        print(f"  ✓ {ex} processed")
    
    # 创建 DataFrame
    df_summary = pd.DataFrame(all_stats)
    
    # 保存为 CSV
    output_csv = OUTPUT_DIR / f"error_summary_{DATASET}_EX123.csv"
    df_summary.to_csv(output_csv, index=False)
    print(f"\n✓ Saved: {output_csv}")
    
    # 打印表格
    print("\n" + "="*80)
    print(f"Error Summary Table ({DATASET} dataset)")
    print("="*80 + "\n")
    print(df_summary.to_string(index=False))
    print()
    
    # 额外统计：按 EX 版本汇总
    print("\n" + "="*80)
    print("Summary by EX Version")
    print("="*80 + "\n")
    
    for ex in EX_VERSIONS:
        df_ex = df_summary[df_summary['EX Version'] == ex]
        total_nc = df_ex['NC'].sum()
        total_runtime = df_ex['Runtime Error'].sum()
        total_incorrect = df_ex['Incorrect'].sum()
        total_correct = df_ex['Correct'].sum()
        total_all = df_ex['Total'].sum()
        
        print(f"{ex}:")
        print(f"  NC:            {total_nc:4d} ({100*total_nc/total_all:.1f}%)")
        print(f"  Runtime Error: {total_runtime:4d} ({100*total_runtime/total_all:.1f}%)")
        print(f"  Incorrect:     {total_incorrect:4d} ({100*total_incorrect/total_all:.1f}%)")
        print(f"  Correct:       {total_correct:4d} ({100*total_correct/total_all:.1f}%)")
        print(f"  Total:         {total_all:4d}")
        print()
    
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
