#!/usr/bin/env python3
"""
EX3 错误统计表生成器

参考 EX1/EX2 的格式，生成 EX3 的错误统计表：
1. Compilation Failure (CF): 编译失败
2. Runtime Failure (RF): 运行时失败（编译成功但运行崩溃）
3. Correctness Error (CE): 正确性错误（运行成功但输出不正确）
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict


# ==================== 配置参数 ====================

WORKSPACE_ROOT = Path(__file__).parent.parent.parent
TIME_MEASUREMENTS_ROOT = WORKSPACE_ROOT / "results" / "time_measurements"
CORRECTNESS_FILE = WORKSPACE_ROOT / "analysis_summaries" / "correctness" / "EX3_correctness_summary.csv"
OUTPUT_DIR = WORKSPACE_ROOT / "analysis_summaries" / "error_tables"
EX3_ROOT = WORKSPACE_ROOT / "EX3"
BENCHMARK_ARGS_FILE = WORKSPACE_ROOT / "configs" / "EX3_configs" / "benchmark_args_ex3.json"

MODELS = ['claude', 'gpt5.1', 'qwen']
DATASET_SIZES = ['mini', 'small', 'medium', 'large', 'extra-large']
N_TRIALS = 10  # 每个配置的试验次数


# ==================== 数据加载函数 ====================

def load_ex3_benchmarks():
    """加载 EX3 的 benchmark 列表"""
    with open(BENCHMARK_ARGS_FILE, 'r') as f:
        data = json.load(f)
    # 排除 "comment" 这样的非 benchmark 项
    benchmarks = [bm for bm in data.keys() if bm != 'comment']
    return benchmarks


def load_correctness_data():
    """加载 EX3 correctness 数据"""
    if not CORRECTNESS_FILE.exists():
        print(f"[WARN] Correctness file not found: {CORRECTNESS_FILE}")
        return {}
    
    df = pd.read_csv(CORRECTNESS_FILE)
    
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


def load_time_measurements_data(dataset_size):
    """
    加载指定 dataset_size 的所有 benchmark 的 time_measurements 数据
    
    Returns:
        dict: {(benchmark, model, version): {status, ...}}
    """
    data = {}
    
    for bm_dir in sorted(TIME_MEASUREMENTS_ROOT.iterdir()):
        if not bm_dir.is_dir():
            continue
        
        benchmark = bm_dir.name
        time_file = bm_dir / "EX3_time_measurements.jsonl"
        
        if not time_file.exists():
            continue
        
        with open(time_file, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    
                    # 只处理 summary 类型的记录
                    if record.get('record_type') != 'summary':
                        continue
                    
                    # 只处理指定 dataset_size
                    if record.get('dataset', '').lower() != dataset_size.lower():
                        continue
                    
                    model = record.get('model', '')
                    version = record.get('version')
                    
                    # 跳过 baseline
                    if model == 'baseline' or version is None:
                        continue
                    
                    # 跳过不在指定模型列表中的
                    if model not in MODELS:
                        continue
                    
                    key = (benchmark, model, version)
                    
                    # 判断运行状态
                    successful_runs = record.get('successful_runs', 0)
                    status = record.get('status', 'unknown')
                    
                    data[key] = {
                        'status': status,
                        'successful_runs': successful_runs,
                        'benchmark': benchmark,
                        'model': model,
                        'version': version
                    }
                    
                except Exception as e:
                    continue
    
    return data


def get_compiled_versions(benchmark):
    """
    获取每个模型编译成功的版本列表
    
    Returns:
        dict: {model: set(versions)}
    """
    compiled_versions = {model: set() for model in MODELS}
    
    bm_dir = EX3_ROOT / benchmark / "EX3_optimized_codes"
    
    if not bm_dir.exists():
        return compiled_versions
    
    # 检查哪些可执行文件存在
    for model in MODELS:
        for version in range(1, N_TRIALS + 1):
            exe_name = f"{benchmark}_{model}_v{version}"
            exe_file = bm_dir / exe_name
            
            if exe_file.exists() and exe_file.is_file():
                # 检查是否可执行
                if exe_file.stat().st_mode & 0o111:
                    compiled_versions[model].add(version)
    
    return compiled_versions


def analyze_errors_for_dataset(dataset_size, correctness_map):
    """
    统计指定 dataset_size 的错误
    
    Returns:
        DataFrame with columns: benchmark, model, total_trials, CF, RF, CE, correct
    """
    
    # 加载 time_measurements 数据
    time_data = load_time_measurements_data(dataset_size)
    
    # 只统计 EX3 的 benchmark
    benchmarks = load_ex3_benchmarks()
    
    # 统计每个 (benchmark, model) 的错误
    results_dict = {}
    
    for benchmark in benchmarks:
        # 获取编译成功的版本
        compiled_versions = get_compiled_versions(benchmark)
        
        for model in MODELS:
            key = (benchmark, model)
            counts = {'CF': 0, 'RF': 0, 'CE': 0, 'correct': 0}
            
            # 统计编译失败（CF）：应该有10个版本，实际编译成功的少于10个
            compiled_set = compiled_versions[model]
            all_versions = set(range(1, N_TRIALS + 1))
            failed_versions = all_versions - compiled_set
            counts['CF'] = len(failed_versions)
            
            # 统计编译成功的版本的运行情况
            for version in compiled_set:
                time_key = (benchmark, model, version)
                
                if time_key in time_data:
                    time_info = time_data[time_key]
                    successful_runs = time_info['successful_runs']
                    
                    if successful_runs == 0:
                        # 运行时失败（RF）
                        counts['RF'] += 1
                    else:
                        # 运行成功，检查 correctness
                        corr_key = (benchmark, model, version)
                        is_correct = correctness_map.get(corr_key, False)
                        
                        if is_correct:
                            counts['correct'] += 1
                        else:
                            counts['CE'] += 1
                else:
                    # 编译成功但没有 time_measurements 记录
                    # 可能是还没运行，暂时不统计
                    pass
            
            # 只保存有数据的 benchmark-model 组合
            total = counts['CF'] + counts['RF'] + counts['CE'] + counts['correct']
            if total > 0:
                results_dict[key] = {
                    'benchmark': benchmark,
                    'model': model,
                    'total_trials': total,
                    'CF': counts['CF'],
                    'RF': counts['RF'],
                    'CE': counts['CE'],
                    'correct': counts['correct']
                }
    
    # 转换为 DataFrame
    results = list(results_dict.values())
    return pd.DataFrame(results)


def generate_summary_table(error_df):
    """
    生成汇总表：按 benchmark 汇总所有模型的错误
    """
    if len(error_df) == 0:
        return pd.DataFrame()
    
    # 按 benchmark 聚合
    summary = error_df.groupby('benchmark').agg({
        'total_trials': 'sum',
        'CF': 'sum',
        'RF': 'sum',
        'CE': 'sum',
        'correct': 'sum'
    }).reset_index()
    
    # 计算百分比
    summary['CF_pct'] = (summary['CF'] / summary['total_trials'] * 100).round(2)
    summary['RF_pct'] = (summary['RF'] / summary['total_trials'] * 100).round(2)
    summary['CE_pct'] = (summary['CE'] / summary['total_trials'] * 100).round(2)
    summary['correct_pct'] = (summary['correct'] / summary['total_trials'] * 100).round(2)
    
    return summary


def generate_model_breakdown_table(error_df):
    """
    生成模型分解表：每个 benchmark 在不同模型下的错误
    """
    if len(error_df) == 0:
        return pd.DataFrame()
    
    # 为每个模型创建一个子表
    model_dfs = []
    
    for model in MODELS:
        df_model = error_df[error_df['model'] == model][['benchmark', 'CF', 'RF', 'CE', 'correct']].copy()
        
        # 重命名列
        df_model.columns = ['benchmark', f'{model}_CF', f'{model}_RF', f'{model}_CE', f'{model}_correct']
        
        model_dfs.append(df_model)
    
    # 合并所有模型
    result = model_dfs[0]
    for df in model_dfs[1:]:
        result = result.merge(df, on='benchmark', how='outer')
    
    return result


# ==================== 主函数 ====================

def main():
    """主函数：生成 EX3 错误统计表"""
    
    print("="*80)
    print("EX3 错误统计表生成器")
    print("="*80)
    print()
    
    # 确保输出目录存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {OUTPUT_DIR}")
    print()
    
    # 加载 correctness 数据
    print("加载 correctness 数据...")
    correctness_map = load_correctness_data()
    print(f"  ✓ 加载了 {len(correctness_map)} 条 correctness 记录")
    print()
    
    # 处理每个 dataset_size
    print("="*80)
    print("处理 EX3")
    print("="*80)
    print()
    
    for ds in DATASET_SIZES:
        print(f"  处理 EX3 - {ds}...")
        
        # 分析错误
        error_df = analyze_errors_for_dataset(ds, correctness_map)
        
        if len(error_df) == 0:
            print(f"    ⚠️  无数据")
            continue
        
        print(f"    ✓ 分析了 {len(error_df)} 条记录")
        
        # ===== 1. 详细表（每个 benchmark × model） =====
        detail_file = OUTPUT_DIR / f"error_detail_EX3_{ds}.csv"
        error_df['ex'] = 'EX3'  # 添加 ex 列以便与 EX1/EX2 合并
        error_df.to_csv(detail_file, index=False)
        print(f"    ✓ 详细表: {detail_file.name}")
        
        # ===== 2. 汇总表（按 benchmark 汇总） =====
        summary_df = generate_summary_table(error_df)
        summary_df['ex'] = 'EX3'  # 添加 ex 列
        summary_file = OUTPUT_DIR / f"error_summary_EX3_{ds}.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"    ✓ 汇总表: {summary_file.name}")
        
        # ===== 3. 模型分解表（宽格式） =====
        model_breakdown_df = generate_model_breakdown_table(error_df)
        breakdown_file = OUTPUT_DIR / f"error_model_breakdown_EX3_{ds}.csv"
        model_breakdown_df.to_csv(breakdown_file, index=False)
        print(f"    ✓ 模型分解表: {breakdown_file.name}")
        
        print()
    
    print("="*80)
    print("✅ EX3 错误统计表生成完成！")
    print("="*80)
    print()
    print("生成的文件类型：")
    print("  1. error_detail_EX3_*.csv         - 详细表（benchmark × model）")
    print("  2. error_summary_EX3_*.csv        - 汇总表（按 benchmark）")
    print("  3. error_model_breakdown_EX3_*    - 模型分解表（宽格式）")
    print()
    print("错误类型说明：")
    print("  - CF (Compilation Failure):  编译失败")
    print("  - RF (Runtime Failure):      运行时失败（编译成功但运行崩溃）")
    print("  - CE (Correctness Error):    正确性错误（运行成功但输出不正确）")
    print("  - correct:                   完全正确")
    print()
    print("注意：EX3 的 CF 计数可能为 0，因为编译失败的版本不会生成 time_measurements 记录")
    print()


if __name__ == "__main__":
    main()
