#!/usr/bin/env python3
"""
EX3 完整分析流程

执行步骤：
1. 构建统一数据框
2. 计算 Fast@k 和 Speedup@k 指标
3. 生成可视化图表
4. 输出统计摘要

Usage:
    python scripts/analysis/analyze_ex3.py
"""
import sys
from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

def run_step(step_name, script_path, *args):
    """运行分析步骤"""
    print(f"\n{'='*80}")
    print(f"Step: {step_name}")
    print(f"{'='*80}\n")
    
    cmd = ["python", str(script_path)] + list(args)
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    
    if result.returncode != 0:
        print(f"[ERROR] {step_name} failed with code {result.returncode}")
        return False
    
    print(f"\n✓ {step_name} completed successfully")
    return True

def main():
    print("\n" + "="*80)
    print("EX3 Analysis Pipeline")
    print("="*80)
    
    scripts_dir = PROJECT_ROOT / "scripts" / "analysis"
    
    # Step 1: 构建统一数据框
    if not run_step(
        "Build Unified DataFrame",
        scripts_dir / "build_unified_dataframe_ex3.py"
    ):
        return
    
    # Step 2: 生成 Fast@k 和 Speedup@k 图表
    if not run_step(
        "Plot Fast@k and Speedup@k",
        scripts_dir / "plot_fast_speedup_at_k_ex3.py"
    ):
        return
    
    print("\n" + "="*80)
    print("✓ EX3 Analysis Pipeline Completed!")
    print("="*80)
    print("\nGenerated Files:")
    print("  - analysis_summaries/unified_data/unified_df_EX3.csv")
    print("  - analysis_summaries/unified_data/fast_speedup_at_k*_EX3.csv")
    print("  - analysis_summaries/figures/EX3/*.pdf")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
