#!/usr/bin/env bash
#
# EX3 分析流程执行脚本
#
# 使用方法:
#   chmod +x scripts/analysis/run_ex3_analysis.sh
#   ./scripts/analysis/run_ex3_analysis.sh
#

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "========================================"
echo "EX3 Analysis Pipeline"
echo "========================================"
echo ""

# Step 1: 构建统一数据框
echo "[1/2] Building unified dataframe..."
python scripts/analysis/build_unified_dataframe_ex3.py \
    --correctness-file analysis_summaries/correctness/EX3_correctness_summary.csv \
    --speedup-detail-dir analysis_summaries/speedup/detail/EX3 \
    --output-dir analysis_summaries/unified_data

echo ""

# Step 2: 生成图表
echo "[2/2] Generating plots..."
python scripts/analysis/plot_fast_speedup_at_k_ex3.py

echo ""
echo "========================================"
echo "✓ EX3 Analysis Completed!"
echo "========================================"
echo ""
echo "Generated files:"
echo "  Data: analysis_summaries/unified_data/"
echo "  Figures: analysis_summaries/figures/EX3/"
echo ""
