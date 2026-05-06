# Canneal Correctness验证逻辑详解

## 📊 verify_correctness.py 的8个检查步骤

### ✅ 检查1: 文件存在且非空
```python
def check_file_exists(self):
    if not self.output_file.exists():
        self.errors.append("File not found")
        return False
    if self.output_file.stat().st_size == 0:
        self.errors.append("Output file is empty")
        return False
    return True
```
**目的**: 确保程序有输出，没有崩溃

---

### ✅ 检查2: PARSEC头部存在
```python
def check_header(self, content):
    if "PARSEC Benchmark Suite" not in content:
        self.errors.append("Missing PARSEC header")
        return False
    return True
```
**目的**: 确认这是canneal的正确输出格式

---

### ✅ 检查3: Netlist创建成功
```python
def check_netlist_created(self, content):
    pattern = r"netlist created\.\s+(\d+)\s+elements"
    match = re.search(pattern, content)
    if not match:
        self.errors.append("Netlist creation not confirmed")
        return False
    return True
```
**目的**: 确认输入数据加载成功

---

### ✅ 检查4: 提取Initial Cost
```python
def extract_costs(self, content):
    initial_pattern = r"Initial cost:\s+([\d.e+]+)"
    initial_match = re.search(initial_pattern, content)
    if not initial_match:
        self.errors.append("Initial cost not found")
        return False
    self.initial_cost = float(initial_match.group(1))
```
**目的**: 获取优化前的基准成本

---

### ✅ 检查5: 提取Final Cost
```python
    final_pattern = r"Final cost:\s+([\d.e+]+)"
    final_match = re.search(final_pattern, content)
    if not final_match:
        self.errors.append("Final cost not found")
        return False
    self.final_cost = float(final_match.group(1))
```
**目的**: 获取优化后的成本

---

### ✅ 检查6: 提取Improvement
```python
    improvement_pattern = r"Improvement:\s+([\d.e+-]+)"
    improvement_match = re.search(improvement_pattern, content)
    if not improvement_match:
        self.errors.append("Improvement not found")
        return False
    self.improvement = float(improvement_match.group(1))
```
**目的**: 获取改进量

---

### ✅ 检查7: Final Cost < Initial Cost ⭐核心检查
```python
def check_optimization_worked(self):
    if self.final_cost >= self.initial_cost:
        self.errors.append(
            f"Optimization failed: Final cost ({self.final_cost}) >= "
            f"Initial cost ({self.initial_cost})"
        )
        return False
```
**目的**: **这是最关键的correctness检查！**
- 如果 Final Cost >= Initial Cost，说明优化没有起作用
- 这表明代码有bug或算法失效

---

### ✅ 检查8: Improvement > 0 ⭐核心检查
```python
    if self.improvement <= 0:
        self.errors.append(f"Improvement is not positive: {self.improvement}")
        return False
    return True
```
**目的**: **第二个核心correctness检查！**
- Improvement应该等于 Initial Cost - Final Cost
- 如果<=0，说明计算有误或优化失败

---

## 📊 compare_variants.py 的额外验证

除了运行上述8个检查外，还会：

### 1️⃣ 比较不同variant的Final Cost差异

```python
def compare_with_baseline(self, variant_final, baseline_final):
    diff_pct = abs(variant_final - baseline_final) / baseline_final * 100
    
    if diff_pct < 5:
        return f"{diff_pct:.2f}% (✅ Normal)"
    elif diff_pct < 10:
        return f"{diff_pct:.2f}% (⚠️  Acceptable)"
    else:
        return f"{diff_pct:.2f}% (❌ Large diff)"
```

**判断标准**:
- **< 5%**: 正常的随机波动
- **5-10%**: 可接受（可能由于编译优化、并行化等）
- **> 10%**: 需要检查，可能有问题

---

## 🎯 完整的Correctness判断流程

```
输入: canneal输出文件
  │
  ├─► 检查1: 文件存在且非空？
  │    └─► ❌ NO → 程序崩溃/未运行 → FAIL
  │    └─► ✅ YES → 继续
  │
  ├─► 检查2: 包含PARSEC头部？
  │    └─► ❌ NO → 输出格式错误 → FAIL
  │    └─► ✅ YES → 继续
  │
  ├─► 检查3: Netlist创建成功？
  │    └─► ❌ NO → 输入加载失败 → FAIL
  │    └─► ✅ YES → 继续
  │
  ├─► 检查4-6: 提取成本数据
  │    └─► ❌ NO → 输出不完整 → FAIL
  │    └─► ✅ YES → 继续
  │
  ├─► 检查7: Final Cost < Initial Cost？ ⭐⭐⭐
  │    └─► ❌ NO → 优化失败！→ FAIL
  │    └─► ✅ YES → 继续
  │
  ├─► 检查8: Improvement > 0？ ⭐⭐⭐
  │    └─► ❌ NO → 计算错误！→ FAIL
  │    └─► ✅ YES → PASS
  │
  └─► 🎉 输出正确！
```

---

## 📋 实际示例

### 示例1: 正确的输出 ✅

```
Initial cost: 1,005,710
Final cost:   562,972
Improvement:  442,737

验证结果:
✅ Final cost < Initial cost? YES (562,972 < 1,005,710)
✅ Improvement > 0? YES (442,737 > 0)
✅ Improvement计算正确? YES (1,005,710 - 562,972 = 442,738 ≈ 442,737)
→ PASS
```

### 示例2: 优化失败的输出 ❌

```
Initial cost: 1,005,710
Final cost:   1,123,456
Improvement:  -117,746

验证结果:
❌ Final cost < Initial cost? NO (1,123,456 > 1,005,710)
❌ Improvement > 0? NO (-117,746 < 0)
→ FAIL - 优化算法没有工作！
```

### 示例3: 两个variant的比较

```
Baseline:
  Initial: 1,005,710
  Final:   562,972
  ✅ PASS

Variant 1:
  Initial: 1,005,710
  Final:   571,234
  ✅ PASS
  
差异: (571,234 - 562,972) / 562,972 = 1.47%
判断: ✅ Normal (< 5%)
```

```
Variant 2:
  Initial: 1,005,710
  Final:   702,456
  ✅ PASS (优化有效)
  
差异: (702,456 - 562,972) / 562,972 = 24.8%
判断: ❌ Large diff (> 10%) - 需要检查代码！
```

---

## 🔑 关键点总结

### Correctness的核心标准（必须满足）:

1. **Final Cost < Initial Cost** ⭐⭐⭐
   - 这是最重要的检查
   - 如果不满足，说明优化算法完全失效

2. **Improvement > 0** ⭐⭐⭐
   - 验证计算是否正确
   - Improvement = Initial Cost - Final Cost

3. **输出格式完整** ⭐
   - 程序没有崩溃
   - 所有必要字段都存在

### 可接受的差异:

- 不同运行之间Final Cost会不同（随机算法特性）
- 不同variant之间Final Cost差异 < 10% 是可以接受的
- 差异 > 10% 需要检查，但如果优化有效（Final < Initial），仍然算"正确"

### 不要求:

- ❌ Final Cost必须完全相同
- ❌ Improvement必须完全相同
- ❌ 运行时间必须相同
