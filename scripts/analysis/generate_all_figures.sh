#!/bin/bash

echo "========================================="
echo "  HPC-Bench 完整图表生成"
echo "  Fast@3 + Speedup@3"
echo "========================================="

# 配置
MODEL="claude"
THREADS=16

# ==================== Fast@3 图表 ====================

echo ""
echo "========================================="
echo "  Part 1: Fast@3 图表"
echo "========================================="

# 1.1 Motif 分层图 - Fast@3（所有 difficulty 平均）
echo ""
echo "[1/6] Generating Motif figures (Fast@3, all difficulties)..."
python scripts/analysis/plot_motif_figures.py \
    --model $MODEL \
    --threads $THREADS \
    --metric Fast_at_k

# 1.2 Motif 分层图 - Fast@3（按 difficulty 分层）
echo ""
echo "[2/6] Generating Motif figures (Fast@3, by difficulty)..."
for diff in d1 d2 d3 d4; do
    echo "  - Difficulty: $diff"
    python scripts/analysis/plot_motif_figures.py \
        --model $MODEL \
        --threads $THREADS \
        --difficulty $diff \
        --metric Fast_at_k
done

# 1.3 Difficulty 分层图 - Fast@3
echo ""
echo "[3/6] Generating Difficulty figures (Fast@3)..."
python scripts/analysis/plot_difficulty_figures.py \
    --model $MODEL \
    --threads $THREADS \
    --metric Fast_at_k_avg \
    --mode both

# ==================== Speedup@3 图表 ====================

echo ""
echo "========================================="
echo "  Part 2: Speedup@3 图表"
echo "========================================="

# 2.1 Motif 分层图 - Speedup@3（所有 difficulty 平均）
echo ""
echo "[4/6] Generating Motif figures (Speedup@3, all difficulties)..."
python scripts/analysis/plot_motif_figures.py \
    --model $MODEL \
    --threads $THREADS \
    --metric Speedup_at_k

# 2.2 Motif 分层图 - Speedup@3（按 difficulty 分层）
echo ""
echo "[5/6] Generating Motif figures (Speedup@3, by difficulty)..."
for diff in d1 d2 d3 d4; do
    echo "  - Difficulty: $diff"
    python scripts/analysis/plot_motif_figures.py \
        --model $MODEL \
        --threads $THREADS \
        --difficulty $diff \
        --metric Speedup_at_k
done

# 2.3 Difficulty 分层图 - Speedup@3
echo ""
echo "[6/6] Generating Difficulty figures (Speedup@3)..."
python scripts/analysis/plot_difficulty_figures.py \
    --model $MODEL \
    --threads $THREADS \
    --metric Speedup_at_k_avg \
    --mode both

# ==================== 完成 ====================

echo ""
echo "========================================="
echo "  ✅ 所有图表生成完成！"
echo "========================================="
echo ""
echo "生成的图表位置："
echo "  - Motif 图: analysis_summaries/motif_category_by_d_level/"
echo "  - Difficulty 图: analysis_summaries/difficulty_category/"
echo ""
echo "图表类型："
echo "  - Fast@3: 所有 Motif 和 Difficulty 分层图"
echo "  - Speedup@3: 所有 Motif 和 Difficulty 分层图"
echo ""
