# Canneal Correctness Verification Guide

## 📋 输出文件说明

Canneal的输出包含以下关键信息：

```
PARSEC Benchmark Suite
Threadcount: 1
500000 swaps per temperature step
start temperature: 3500
netlist filename: input_data/large/input/input_large.nets
number of temperature steps: 700
locs created
locs assigned
netlist created. 5000 elements.
Initial routing cost: 1.00571e+06

========================================
Routing Optimization Results:
========================================
Initial cost: 1.00571e+06
Final cost:   562972
Improvement:  442737 (44.0224%)
========================================
```

### 关键指标

| 指标 | 说明 | 示例值 |
|------|------|--------|
| **Initial cost** | 初始路由成本（随机放置） | 1,005,710 |
| **Final cost** | 优化后的路由成本 | 562,972 |
| **Improvement** | 改进量（绝对值和百分比） | 442,737 (44.02%) |

## ⚠️ 重要：Canneal的特殊性

**Canneal使用模拟退火算法（Simulated Annealing）**，这是一个**随机优化算法**，因此：

### 1. 每次运行结果会不同

- ✅ **正常现象**：同一个binary，用相同参数运行多次，Final cost会略有不同
- ✅ **原因**：算法依赖随机数，每次探索的路径不同
- ⚠️ **不要期望**：两次运行得到完全相同的结果

### 2. 不同variant的结果会不同

| 差异程度 | 说明 | 是否正确 |
|---------|------|----------|
| **< 5%** | 正常的随机波动 | ✅ 正确 |
| **5-10%** | 可接受的差异（可能由编译优化、并行化导致） | ✅ 正确 |
| **> 10%** | 可能有问题，需要检查 | ⚠️ 需要调查 |
| **Final cost > Initial cost** | 优化失败 | ❌ 错误 |

### 3. Correctness的定义

对于Canneal，**正确性（Correctness）**的标准是：

✅ **必须满足**：
1. 程序正常运行，无崩溃（segfault, abort等）
2. Final cost < Initial cost（优化有效）
3. Improvement > 0
4. 输出格式正确，包含所有必要字段

❌ **不要求**：
1. Final cost必须相同
2. Improvement必须相同
3. 运行时间必须相同

## 🔧 使用Python验证工具

### 工具1: verify_correctness.py - 验证单个输出文件

**基本用法**：
```bash
# 运行程序并保存输出
./EX1_optimized_codes/canneal_gcc 1 1000 2000 input_data/mini/input/input_mini.nets 50 > output.txt

# 验证输出正确性
python3 verify_correctness.py output.txt
```

**输出示例**：
```
======================================================================
Canneal Correctness Verification
======================================================================
File: output.txt

✅ ALL CHECKS PASSED - Output is CORRECT

Summary:
  Initial Cost: 1,005,710
  Final Cost:   562,972
  Improvement:  442,737 (44.02%)
======================================================================
```

**验证内容**：
- ✅ 文件不为空
- ✅ 包含PARSEC header
- ✅ 包含netlist创建确认
- ✅ 包含Initial cost
- ✅ 包含Final cost
- ✅ 包含Improvement
- ✅ Final cost < Initial cost
- ✅ Improvement > 0

**选项**：
- `-q, --quiet`: 安静模式，减少输出

**返回值**：
- `0`: 验证通过
- `1`: 验证失败

### 工具2: compare_variants.py - 批量比较多个variant

**基本用法**：
```bash
# 比较所有variant在mini size上的表现
python3 compare_variants.py mini

# 比较所有variant在large size上的表现
python3 compare_variants.py large
```

**输出示例**：
```
======================================================================
Canneal Variant Comparison - Size: mini
======================================================================

Test Parameters:
  Size:     mini
  Netlist:  input_data/mini/input/input_mini.nets
  Swaps:    1000
  Steps:    50

======================================================================
Running Tests...
======================================================================

[Baseline] canneal_gcc
  ✅ PASS - Final cost: 1,234

[Variant]  canneal_gcc_variant1
  ✅ PASS - Final cost: 1,267

======================================================================
Summary
======================================================================

Total binaries tested: 2
✅ Passed: 2
Failed: 0

Comparison with Baseline:
Binary                         Final Cost    Diff from Baseline
----------------------------------------------------------------------
canneal_gcc                         1,234                    -
canneal_gcc_variant1                1,267        2.67% (✅ Normal)

Output files saved in: correctness_test/mini/
```

**功能**：
1. 自动找到所有variant binaries
2. 使用相同参数运行每个variant
3. 验证每个输出的正确性
4. 计算与baseline的差异
5. 生成对比报告

**输出文件**：
- `correctness_test/<size>/<binary_name>.txt` - 每个binary的输出
- `correctness_test/<size>/results.json` - JSON格式的结果汇总

**选项**：
- `size`: 测试的size (mini, small, medium, large, extra-large)
- `-o, --output-dir`: 输出目录（默认：correctness_test）

**返回值**：
- `0`: 所有variant都通过
- `1`: 至少有一个variant失败

## 📊 在Python代码中使用

### 示例1: 验证单个输出文件

```python
from verify_correctness import CorrectnessChecker

# 创建checker
checker = CorrectnessChecker("output.txt")

# 运行验证
success = checker.verify()

# 获取结果
if success:
    print(f"Initial cost: {checker.initial_cost}")
    print(f"Final cost: {checker.final_cost}")
    print(f"Improvement: {checker.improvement}")
else:
    print("Errors:")
    for error in checker.errors:
        print(f"  - {error}")
```

### 示例2: 批量验证多个文件

```python
from pathlib import Path
from verify_correctness import CorrectnessChecker

output_dir = Path("correctness_test/mini")
results = {}

for output_file in output_dir.glob("*.txt"):
    checker = CorrectnessChecker(output_file)
    success = checker.verify()
    results[output_file.name] = {
        'success': success,
        'final_cost': checker.final_cost,
        'improvement': checker.improvement
    }

# 分析结果
for name, result in results.items():
    if result['success']:
        print(f"✅ {name}: {result['final_cost']}")
    else:
        print(f"❌ {name}: FAILED")
```

### 示例3: 集成到测试框架

```python
import subprocess
from verify_correctness import CorrectnessChecker

def test_canneal_variant(binary, params):
    """Test a canneal variant"""
    # Run binary
    cmd = [binary] + params
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Save output
    output_file = f"test_{binary.name}.txt"
    with open(output_file, 'w') as f:
        f.write(result.stdout)
    
    # Verify correctness
    checker = CorrectnessChecker(output_file)
    success = checker.verify()
    
    return {
        'success': success,
        'final_cost': checker.final_cost,
        'errors': checker.errors
    }

# 使用示例
result = test_canneal_variant(
    "EX1_optimized_codes/canneal_gcc",
    ["1", "1000", "2000", "input_data/mini/input/input_mini.nets", "50"]
)

if result['success']:
    print(f"✅ Test passed: Final cost = {result['final_cost']}")
else:
    print(f"❌ Test failed: {result['errors']}")
```

## 🎯 Variant比较建议

### Step 1: 先验证格式正确性
```bash
# 确保所有variant都能正常运行
python3 compare_variants.py mini
```

### Step 2: 检查优化效果范围
```bash
# 运行多次，观察结果范围
for i in {1..5}; do
    echo "Run $i:"
    ./EX1_optimized_codes/canneal_gcc 1 1000 2000 input_data/mini/input/input_mini.nets 50 | grep "Final cost:"
done
```

### Step 3: 使用Python分析差异
```python
from pathlib import Path
from verify_correctness import CorrectnessChecker

# 读取baseline和variant结果
baseline_checker = CorrectnessChecker("correctness_test/mini/canneal_gcc.txt")
baseline_checker.verify()
baseline_final = baseline_checker.final_cost

variant_checker = CorrectnessChecker("correctness_test/mini/canneal_gcc_variant1.txt")
variant_checker.verify()
variant_final = variant_checker.final_cost

# 计算差异
diff_pct = abs(variant_final - baseline_final) / baseline_final * 100
print(f"Baseline: {baseline_final:,.0f}")
print(f"Variant:  {variant_final:,.0f}")
print(f"Difference: {diff_pct:.2f}%")

if diff_pct < 5:
    print("✅ Normal variation")
elif diff_pct < 10:
    print("⚠️  Acceptable difference")
else:
    print("❌ Large difference - needs investigation")
```

## 📝 总结

### ✅ Canneal的Correctness标准

1. **功能正确性**：
   - 程序不崩溃
   - Final cost < Initial cost
   - 输出格式正确

2. **性能可接受性**：
   - Final cost与baseline差异 < 10%
   - 运行时间合理（与baseline相差不超过2-3倍）

3. **不要求**：
   - Final cost完全相同
   - 每次运行结果一致

### 🔍 Debug建议

如果某个variant的Final cost明显更差（>10%差异）：

1. **检查编译选项**：确保优化级别一致
2. **检查浮点运算**：不同优化可能影响浮点精度
3. **检查并行化**：线程数、随机种子设置
4. **检查内存**：是否有内存错误（用valgrind检查）
5. **多次运行**：排除随机波动的影响

### 📂 相关文件

- `verify_correctness.py` - 单文件验证工具
- `compare_variants.py` - 批量variant比较工具
- `CORRECTNESS_README.md` - 本文档
- `correctness_test/*/` - 测试输出目录（自动创建）

### 🚀 快速开始

```bash
# 1. 验证单个输出
python3 verify_correctness.py test.txt

# 2. 比较所有variant
python3 compare_variants.py mini

# 3. 查看结果
cat correctness_test/mini/results.json
```

