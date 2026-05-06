#!/usr/bin/env python3
"""
生成 EX1/EX2/EX3 的综合错误汇总表

为每个 dataset_size，对比 EX1/EX2/EX3 的错误统计
"""

import pandas as pd
from pathlib import Path


# ==================== 配置参数 ====================

WORKSPACE_ROOT = Path(__file__).parent.parent.parent
ERROR_TABLES_DIR = WORKSPACE_ROOT / "analysis_summaries" / "error_tables"
OUTPUT_DIR = WORKSPACE_ROOT / "analysis_summaries" / "error_tables"

EX_VERSIONS = ['EX1', 'EX2', 'EX3']
DATASET_SIZES = ['mini', 'small', 'medium', 'large', 'extra-large']
MODELS = ['claude', 'gpt5.1', 'qwen']


# ==================== 主函数 ====================

def main():
    """主函数：生成综合错误汇总表"""
    
    print("="*80)
    print("EX1/EX2/EX3 综合错误汇总表生成器")
    print("="*80)
    print()
    
    # 确保输出目录存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 为每个 dataset_size 生成汇总表
    for ds in DATASET_SIZES:
        print(f"处理 dataset: {ds}...")
        
        # 收集所有 EX 版本的 detail 数据
        all_dfs = []
        
        for ex in EX_VERSIONS:
            detail_file = ERROR_TABLES_DIR / f"error_detail_{ex}_{ds}.csv"
            
            if detail_file.exists():
                df = pd.read_csv(detail_file)
                all_dfs.append(df)
                print(f"  ✓ 读取 {ex}: {len(df)} 行")
            else:
                print(f"  ⚠️  未找到: error_detail_{ex}_{ds}.csv")
        
        if len(all_dfs) == 0:
            print(f"  ⚠️  {ds} 无数据\n")
            continue
        
        # 合并所有数据
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # 保存合并的 detail 数据
        combined_detail_file = OUTPUT_DIR / f"error_detail_all_ex_{ds}.csv"
        combined_df.to_csv(combined_detail_file, index=False)
        print(f"  ✓ 保存合并 detail: {combined_detail_file.name}")
        
        # 生成 summary（按 ex 和 benchmark 聚合）
        summary = combined_df.groupby(['ex', 'benchmark']).agg({
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
        
        # 保存 summary
        summary_file = OUTPUT_DIR / f"error_summary_all_ex_{ds}.csv"
        summary.to_csv(summary_file, index=False)
        print(f"  ✓ 保存汇总 summary: {summary_file.name}")
        
        # 生成整体统计
        print(f"\n  === {ds} 整体统计 ===")
        for ex in EX_VERSIONS:
            df_ex = summary[summary['ex'] == ex]
            if len(df_ex) == 0:
                continue
            
            total_trials = df_ex['total_trials'].sum()
            total_cf = df_ex['CF'].sum()
            total_rf = df_ex['RF'].sum()
            total_ce = df_ex['CE'].sum()
            total_correct = df_ex['correct'].sum()
            
            print(f"  {ex}:")
            print(f"    Total Trials:    {total_trials}")
            print(f"    CF:              {total_cf:4d} ({100*total_cf/total_trials:.1f}%)")
            print(f"    RF:              {total_rf:4d} ({100*total_rf/total_trials:.1f}%)")
            print(f"    CE:              {total_ce:4d} ({100*total_ce/total_trials:.1f}%)")
            print(f"    Correct:         {total_correct:4d} ({100*total_correct/total_trials:.1f}%)")
        
        print()
    
    print("="*80)
    print("✅ 综合错误汇总表生成完成！")
    print("="*80)
    print()
    print("生成的文件：")
    print("  - error_detail_all_ex_{dataset_size}.csv")
    print("  - error_summary_all_ex_{dataset_size}.csv")
    print()


if __name__ == "__main__":
    main()
