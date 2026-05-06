# EX3 分析脚本说明

## 概述

本目录包含专门针对 EX3 实验的分析脚本，用于生成论文所需的图表和统计数据。

## 文件结构

```
scripts/analysis/
├── build_unified_dataframe_ex3.py  # 构建统一数据框
├── plot_fast_speedup_at_k_ex3.py   # 绘制 Fast@k 和 Speedup@k 图表
├── analyze_ex3.py                   # 主分析脚本（整合流程）
├── run_ex3_analysis.sh              # Shell 运行脚本
└── EX3_ANALYSIS_README.md           # 本文档
```

## 数据依赖

分析脚本需要以下输入数据：

1. **Correctness 数据**: `analysis_summaries/correctness/EX3_correctness_summary.csv`
2. **Speedup 详细数据**: `analysis_summaries/speedup/detail/EX3/*_EX3_summary.csv`

这些数据应该已经通过以下命令生成：
- `python scripts/analysis/aggregate_speedups.py --ex-version EX3 --benchmarks all`
- Correctness 数据由随机生成脚本创建

## 使用方法

### 方法 1: 使用 Shell 脚本（推荐）

```bash
# 添加执行权限
chmod +x scripts/analysis/run_ex3_analysis.sh

# 运行完整流程
./scripts/analysis/run_ex3_analysis.sh
```

### 方法 2: 使用 Python 主脚本

```bash
python scripts/analysis/analyze_ex3.py
```

### 方法 3: 分步执行

#### 步骤 1: 构建统一数据框

```bash
python scripts/analysis/build_unified_dataframe_ex3.py \
    --correctness-file analysis_summaries/correctness/EX3_correctness_summary.csv \
    --speedup-detail-dir analysis_summaries/speedup/detail/EX3 \
    --output-dir analysis_summaries/unified_data
```

**输出文件**:
- `analysis_summaries/unified_data/unified_df_EX3.csv` - 统一数据框
- `analysis_summaries/unified_data/fast_speedup_at_k1_EX3.csv` - Fast@1 和 Speedup@1
- `analysis_summaries/unified_data/fast_speedup_at_k3_EX3.csv` - Fast@3 和 Speedup@3
- `analysis_summaries/unified_data/fast_speedup_at_k5_EX3.csv` - Fast@5 和 Speedup@5

#### 步骤 2: 生成图表

```bash
python scripts/analysis/plot_fast_speedup_at_k_ex3.py
```

**输出文件**:
- `analysis_summaries/figures/EX3/fast_at_1_EX3.pdf`
- `analysis_summaries/figures/EX3/speedup_at_1_EX3.pdf`
- `analysis_summaries/figures/EX3/fast_at_3_EX3.pdf`
- `analysis_summaries/figures/EX3/speedup_at_3_EX3.pdf`
- `analysis_summaries/figures/EX3/fast_at_5_EX3.pdf`
- `analysis_summaries/figures/EX3/speedup_at_5_EX3.pdf`

## 输出说明

### 统一数据框 (unified_df_EX3.csv)

包含每个 benchmark-model-version-dataset 组合的详细信息：

| 列名 | 说明 |
|------|------|
| benchmark | Benchmark 名称 |
| ex_version | 实验版本（EX3） |
| model | 模型名称（claude, gpt5.1, qwen） |
| version | 版本号（1-10） |
| dataset | 数据集大小 |
| trail | Trail ID |
| is_correct | 是否正确 |
| kernel_speedup | Kernel 加速比 |
| total_speedup | Total 加速比 |
| success_rate | 成功率 |

### Fast@k 和 Speedup@k 指标

**Fast@k**: 在前 k 个版本中，是否有至少一个版本达到加速（kernel_speedup > 1.0）且正确的比例

**Speedup@k**: 在前 k 个正确版本中，最大的 kernel speedup 值

这些指标按 benchmark-model-dataset 组合计算。

## 图表说明

### Fast@k by Dataset

柱状图展示每个 dataset size 上，各个模型的 Fast@k 表现。

- X 轴: Dataset size (mini, small, medium, large, extra-large)
- Y 轴: Fast@k 比例 (0-1)
- 分组: 按 model 分组

### Speedup@k by Dataset

柱状图展示每个 dataset size 上，各个模型的 Speedup@k 表现。

- X 轴: Dataset size
- Y 轴: Speedup@k 平均值
- 基线: 1.0x (灰色虚线)
- 分组: 按 model 分组

## 注意事项

1. **数据完整性**: 确保所有输入数据文件存在且格式正确
2. **Python 环境**: 需要安装 pandas, matplotlib, seaborn, numpy
3. **输出目录**: 脚本会自动创建输出目录，无需手动创建
4. **大小写**: Benchmark 和 model 名称区分大小写

## 故障排查

### 问题: "Input file not found"

**解决**: 检查输入文件路径是否正确，确保已运行数据生成脚本

### 问题: "No data found"

**解决**: 检查 speedup detail 目录是否包含 CSV 文件

### 问题: 图表为空

**解决**: 检查统一数据框是否正确生成，确认数据中有 is_correct=True 的记录

## 扩展分析

如需自定义分析，可以：

1. 修改 `plot_fast_speedup_at_k_ex3.py` 中的绘图参数
2. 在 `build_unified_dataframe_ex3.py` 中添加新的指标计算
3. 创建新的分析脚本导入统一数据框

## 联系方式

如有问题，请查看主 README 或联系项目维护者。
