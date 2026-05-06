#!/usr/bin/env python3
"""
生成 EX1/EX2/EX3 的整体汇总表
只有3行：每个 EX 版本的总体统计
"""

import pandas as pd
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).parent.parent.parent
ERROR_TABLES_DIR = WORKSPACE_ROOT / "analysis_summaries" / "error_tables"
OUTPUT_DIR = WORKSPACE_ROOT / "analysis_summaries" / "tables"

DATASET_SIZES = ['mini', 'small', 'medium', 'large', 'extra-large']
EX_VERSIONS = ['EX1', 'EX2', 'EX3']


def generate_overall_summary(dataset_size):
    """
    生成指定 dataset_size 的 EX1/EX2/EX3 整体汇总表
    """
    
    # 读取详细的 summary 数据
    summary_file = ERROR_TABLES_DIR / f"error_summary_all_ex_{dataset_size}.csv"
    
    if not summary_file.exists():
        print(f"  ⚠️  未找到: {summary_file.name}")
        return None
    
    df = pd.read_csv(summary_file)
    
    # 按 EX 版本聚合
    results = []
    
    for ex in EX_VERSIONS:
        df_ex = df[df['ex'] == ex]
        
        if len(df_ex) == 0:
            continue
        
        # 计算总计
        total_trials = df_ex['total_trials'].sum()
        total_cf = df_ex['CF'].sum()
        total_rf = df_ex['RF'].sum()
        total_ce = df_ex['CE'].sum()
        total_correct = df_ex['correct'].sum()
        
        # 计算百分比
        cf_pct = (total_cf / total_trials * 100) if total_trials > 0 else 0
        rf_pct = (total_rf / total_trials * 100) if total_trials > 0 else 0
        ce_pct = (total_ce / total_trials * 100) if total_trials > 0 else 0
        correct_pct = (total_correct / total_trials * 100) if total_trials > 0 else 0
        
        results.append({
            'EX': ex,
            'Benchmarks': len(df_ex),
            'Total Trials': total_trials,
            'CF': total_cf,
            'CF (%)': f"{cf_pct:.1f}%",
            'RF': total_rf,
            'RF (%)': f"{rf_pct:.1f}%",
            'CE': total_ce,
            'CE (%)': f"{ce_pct:.1f}%",
            'Correct': total_correct,
            'Correct (%)': f"{correct_pct:.1f}%"
        })
    
    return pd.DataFrame(results)


def main():
    """主函数"""
    
    print("="*80)
    print("生成 EX1/EX2/EX3 整体汇总表")
    print("="*80)
    print()
    
    # 确保输出目录存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for ds in DATASET_SIZES:
        print(f"处理 {ds}...")
        
        df_summary = generate_overall_summary(ds)
        
        if df_summary is None or len(df_summary) == 0:
            print(f"  ⚠️  无数据")
            continue
        
        # 保存为 CSV
        output_file = OUTPUT_DIR / f"ex_overall_summary_{ds}.csv"
        df_summary.to_csv(output_file, index=False)
        print(f"  ✓ 保存: {output_file.name}")
        
        # 打印预览
        print(f"\n  预览 ({ds}):")
        print(df_summary.to_string(index=False))
        print()
    
    print("="*80)
    print("✅ 完成！")
    print("="*80)
    print()
    print("生成的文件：")
    print("  analysis_summaries/tables/ex_overall_summary_{dataset_size}.csv")
    print()


if __name__ == "__main__":
    main()
