# HPC-Bench 画图指南

## 📁 文件结构

```
scripts/analysis/
├── analysis.ipynb              # 数据处理 + 表格生成
├── plot_motif_figures.py       # Motif 分层图生成 ⭐
├── plot_difficulty_figures.py  # Difficulty 分层图生成 ⭐
└── PLOTTING_README.md          # 本文档
```

## 🎨 1. Motif 分层图生成

### 基本用法

```bash
# 生成所有 Motif 分层图（使用默认配置）
python scripts/analysis/plot_motif_figures.py

# 指定模型和线程数
python scripts/analysis/plot_motif_figures.py --model claude --threads 16
```

### 高级用法

#### 1. 只画特定 Difficulty

```bash
# 只画 d1 (Easy) 难度
python scripts/analysis/plot_motif_figures.py --difficulty d1

# 只画 d4 (Hard) 难度
python scripts/analysis/plot_motif_figures.py --difficulty d4
```

#### 2. 选择特定 Motifs

```bash
# 只画 3 个 motif
python scripts/analysis/plot_motif_figures.py \
    --motifs dense_linear_algebra stencil_computations n_body_methods
```

#### 3. 切换指标

```bash
# 画 Speedup@3 (默认是 Fast@3)
python scripts/analysis/plot_motif_figures.py --metric Speedup_at_k
```

#### 4. 生成 Motif × Difficulty 对比图

```bash
# 为 dense_linear_algebra 生成 4 条 difficulty 线的对比图
python scripts/analysis/plot_motif_figures.py \
    --comparison-motif dense_linear_algebra
```

### 组合示例

```bash
# 为 claude 模型，只画 d1 难度，选择 3 个 motif，同时生成对比图
python scripts/analysis/plot_motif_figures.py \
    --model claude \
    --threads 16 \
    --difficulty d1 \
    --motifs dense_linear_algebra stencil_computations n_body_methods \
    --comparison-motif dense_linear_algebra
```

---

## 🎯 2. Difficulty 分层图生成

### 基本用法

```bash
# 生成所有 Difficulty 分层图（单图模式，默认）
python scripts/analysis/plot_difficulty_figures.py

# 生成 2×5 网格大图
python scripts/analysis/plot_difficulty_figures.py --mode grid

# 两种模式都生成
python scripts/analysis/plot_difficulty_figures.py --mode both
```

### 高级用法

#### 1. 指定模型和线程数

```bash
# 指定模型
python scripts/analysis/plot_difficulty_figures.py --model claude --threads 16
```

#### 2. 切换指标

```bash
# 画 Speedup@3 (默认是 Fast@3)
python scripts/analysis/plot_difficulty_figures.py --metric Speedup_at_k_avg
```

#### 3. 选择画图模式

```bash
# 只生成单独图（10 张）
python scripts/analysis/plot_difficulty_figures.py --mode single

# 只生成网格图（1 张 2×5 大图）
python scripts/analysis/plot_difficulty_figures.py --mode grid

# 两种都生成（11 张）
python scripts/analysis/plot_difficulty_figures.py --mode both
```

### 组合示例

```bash
# 为 claude 模型生成 Fast@3 的网格图
python scripts/analysis/plot_difficulty_figures.py \
    --model claude \
    --threads 16 \
    --metric Fast_at_k_avg \
    --mode grid
```

---

## 📊 生成的图片

### 1. Motif 分层图
- **位置**: `analysis_summaries/motif_category_by_d_level/`
- **单图文件名**: `figure_fast_at_k_by_motif_{difficulty}_{EX}_{dataset_size}.pdf`
- **内容**: 多条 motif 的曲线（每个 motif 一条线）
- **数量**: 10 张图（2 EX × 5 dataset_size）

### 2. Motif × Difficulty 对比图（可选）
- **位置**: `analysis_summaries/motif_category_by_d_level/`
- **文件名**: `figure_fast_at_k_motif_diff_comparison_{motif}_{EX}_{dataset_size}.pdf`
- **内容**: 为一个 motif 画 4 条 difficulty 线
- **数量**: 10 张图（2 EX × 5 dataset_size）

### 3. Difficulty 分层图（单图模式）
- **位置**: `analysis_summaries/difficulty_category/`
- **文件名**: `figure_fast_at_k_by_difficulty_{EX}_{dataset_size}.pdf`
- **内容**: 4 条 difficulty 线（d1, d2, d3, d4）
- **数量**: 10 张图（2 EX × 5 dataset_size）

### 4. Difficulty 分层图（网格模式）
- **位置**: `analysis_summaries/difficulty_category/`
- **文件名**: `figure_fast_at_k_by_difficulty_grid.pdf`
- **内容**: 2×5 大图，包含所有 EX 和 dataset_size 组合
- **数量**: 1 张大图

## 🔧 配置说明

### 默认 Motifs（5 个）
```python
DEFAULT_MOTIFS = [
    'dense_linear_algebra',
    'stencil_computations',
    'image_and_video_processing',
    'n_body_methods',
    'statistical_computations'
]
```

### 可用的 Motifs
- `dense_linear_algebra`
- `stencil_computations`
- `sparse_linear_algebra`
- `image_and_video_processing`
- `graph_algorithms`
- `n_body_methods`
- `statistical_computations`
- `cryptography_and_encoding`
- `dynamic_programming`
- `sorting_and_searching`
- `medical_and_scientific`
- `utility`
- 等...

## 💡 使用提示

### 通用提示

1. **线程数选择**: EX2 推荐使用 `--threads 16`（最大线程数，性能最好）
2. **批量生成**: 使用提供的 shell 脚本可以一键生成所有图表

### Motif 图提示

1. **Difficulty 筛选**: 如果某些 motif 在特定 difficulty 下表现不好（曲线贴底），可以换个 difficulty
2. **自定义 Motifs**: 使用 `--motifs` 参数选择表现好的 motif，避免图中有贴底的线
3. **默认 Motifs**: 当前默认选择了 5 个代表性 motif，表现较好

### Difficulty 图提示

1. **单图 vs 网格**: 
   - 单图模式适合查看每个数据集的详细情况（10 张图）
   - 网格模式适合对比不同数据集（1 张大图，方便放在论文中）
2. **数据来源**: 使用 Table G（difficulty-level aggregation）
3. **4 条线**: 每张图显示 d1, d2, d3, d4 四个难度级别的曲线

## 🚀 批量生成示例脚本

### 示例 1: 生成所有 Motif 图

创建 `generate_all_motif_figures.sh`:

```bash
#!/bin/bash

echo "=== Generating Motif Figures ==="

# 为每个 difficulty 生成图
for diff in d1 d2 d3 d4; do
    echo "Generating figures for difficulty: $diff"
    python scripts/analysis/plot_motif_figures.py \
        --model claude \
        --threads 16 \
        --difficulty $diff \
        --metric Fast_at_k
done

# 生成整体图（所有 difficulty 平均）
echo "Generating figures for all difficulties"
python scripts/analysis/plot_motif_figures.py \
    --model claude \
    --threads 16 \
    --metric Fast_at_k

echo "✅ Motif figures done!"
```

### 示例 2: 生成所有 Difficulty 图

创建 `generate_all_difficulty_figures.sh`:

```bash
#!/bin/bash

echo "=== Generating Difficulty Figures ==="

# 生成单独图和网格图
python scripts/analysis/plot_difficulty_figures.py \
    --model claude \
    --threads 16 \
    --metric Fast_at_k_avg \
    --mode both

echo "✅ Difficulty figures done!"
```

### 示例 3: 一键生成所有图

创建 `generate_all_figures.sh`:

```bash
#!/bin/bash

echo "========================================="
echo "  HPC-Bench 完整图表生成"
echo "========================================="

# 1. Motif 分层图（所有 difficulty 平均）
echo ""
echo "[1/3] Generating Motif figures (all difficulties)..."
python scripts/analysis/plot_motif_figures.py \
    --model claude \
    --threads 16 \
    --metric Fast_at_k

# 2. Motif 分层图（按 difficulty 分层）
echo ""
echo "[2/3] Generating Motif figures (by difficulty)..."
for diff in d1 d2 d3 d4; do
    python scripts/analysis/plot_motif_figures.py \
        --model claude \
        --threads 16 \
        --difficulty $diff \
        --metric Fast_at_k
done

# 3. Difficulty 分层图
echo ""
echo "[3/3] Generating Difficulty figures..."
python scripts/analysis/plot_difficulty_figures.py \
    --model claude \
    --threads 16 \
    --metric Fast_at_k_avg \
    --mode both

echo ""
echo "========================================="
echo "  ✅ 所有图表生成完成！"
echo "========================================="
```

### 运行方式

```bash
# 赋予执行权限
chmod +x generate_all_motif_figures.sh
chmod +x generate_all_difficulty_figures.sh
chmod +x generate_all_figures.sh

# 运行
./generate_all_figures.sh
```
