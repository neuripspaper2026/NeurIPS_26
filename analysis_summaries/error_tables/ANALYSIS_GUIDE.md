# HPC-Bench Error Analysis Guide

This guide provides instructions for analyzing the error statistics tables generated for HPC-Bench evaluation.

---

## 📁 Overview of Generated Tables

The error analysis pipeline generates **5 types of tables** for each `(EX, dataset_size)` combination:

| Table Type | Filename Pattern | Description |
|------------|------------------|-------------|
| **Detail** | `error_detail_{EX}_{dataset_size}.csv` | Per-benchmark, per-model error breakdown |
| **Summary** | `error_summary_{EX}_{dataset_size}.csv` | Per-benchmark aggregation across all models |
| **Model Breakdown** | `error_model_breakdown_{EX}_{dataset_size}.csv` | Wide-format table with columns for each model |
| **Cross-EX Detail** | `error_detail_all_ex_{dataset_size}.csv` | Combined EX1 and EX2 detail data |
| **Cross-EX Summary** | `error_summary_all_ex_{dataset_size}.csv` | Combined EX1 and EX2 summary data |

---

## 🎯 Quick Start: Which Table Should I Use?

### **For Overall Analysis (Recommended Starting Point)**

**Use: `error_summary_all_ex_medium.csv`**

This table provides:
- ✅ Both EX1 and EX2 results (distinguished by `ex` column)
- ✅ Aggregated statistics across all 3 models (30 trials per benchmark)
- ✅ Both absolute counts and percentages
- ✅ Easy comparison of error patterns across benchmarks

**Example rows:**

```csv
ex,benchmark,total_trials,CF,RF,CE,correct,CF_pct,RF_pct,CE_pct,correct_pct
EX1,gemm,30,0,0,2,28,0.0,0.0,6.67,93.33
EX1,cfd,30,20,7,0,3,66.67,23.33,0.0,10.0
EX2,atax,30,1,0,0,29,3.33,0.0,0.0,96.67
```

---

## 📊 Error Type Definitions

The tables track **three types of errors** plus correct executions:

| Error Type | Code | Definition | Detection Method |
|------------|------|------------|------------------|
| **Compilation Failure** | CF | Code fails to compile | `compile_status == 'NC'` |
| **Runtime Failure** | RF | Code compiles but crashes during execution | `compile_status == 'success' AND correctness is NaN` |
| **Correctness Error** | CE | Code runs successfully but produces incorrect output | `correctness == 0` |
| **Correct** | - | Code compiles, runs, and produces correct output | `correctness == 1` |

### Notes:
- **Total trials per benchmark**: 30 (3 models × 10 trials each)
- **Percentage columns**: Calculated as `(count / total_trials) × 100`
- **Mutually exclusive**: Each trial falls into exactly one category (CF, RF, CE, or correct)

---

## 🔍 Recommended Analysis Workflow

### **Step 1: Overall Error Distribution**

**File:** `error_summary_all_ex_medium.csv`

**Goal:** Identify benchmarks with high error rates

**Analysis:**
1. Sort by `CF_pct` (descending) to find benchmarks with compilation difficulties
2. Sort by `CE_pct` (descending) to find benchmarks with correctness challenges
3. Sort by `correct_pct` (ascending) to find the hardest benchmarks overall

**Key Questions:**
- Which benchmarks have `CF_pct > 50%`? (Very difficult to compile)
- Which benchmarks have `CE_pct > 30%`? (Correctness is challenging)
- Which benchmarks have `correct_pct < 50%`? (Overall very difficult)
- How do error patterns differ between EX1 and EX2?

**Example Analysis:**
```
Benchmark: cfd (EX1)
- CF_pct: 66.67% → Compilation is extremely difficult
- RF_pct: 23.33% → Even when it compiles, it often crashes
- correct_pct: 10.0% → Only 3 out of 30 trials succeeded

Interpretation: This is a very challenging benchmark, possibly requiring:
- Advanced compiler directives
- Careful memory management
- Deep understanding of the algorithm
```

---

### **Step 2: Model Performance Comparison**

**File:** `error_model_breakdown_{EX}_{dataset_size}.csv`

**Goal:** Compare model strengths and weaknesses

**Analysis:**
1. For each benchmark, compare error counts across models
2. Identify benchmarks where one model significantly outperforms others
3. Look for complementary strengths (e.g., Model A good at X, Model B good at Y)

**Key Questions:**
- Which model has the lowest average CF count?
- Are there benchmarks where all models fail?
- Are there benchmarks where only one model succeeds?
- Which model is most consistent (lowest variance in errors)?

**Example Analysis:**
```
Benchmark: b+tree

claude:  CF=0, RF=0, CE=10, correct=0  → Compiles and runs, but always incorrect
gpt5.1:  CF=0, RF=0, CE=2,  correct=8  → High success rate
qwen:    CF=0, RF=0, CE=0,  correct=10 → Perfect performance

Interpretation: Qwen and GPT-5.1 understand this benchmark well, while
Claude struggles with correctness. This suggests different optimization
strategies or algorithmic understanding across models.
```

---

### **Step 3: Detailed Model-Specific Analysis**

**File:** `error_detail_all_ex_{dataset_size}.csv`

**Goal:** Examine individual model performance on each benchmark

**Analysis:**
1. Filter by specific model (e.g., `model == 'claude'`)
2. Analyze trial-by-trial consistency
3. Identify patterns in errors (e.g., always CF, or intermittent CE)

**Key Questions:**
- Is a model consistently failing, or are errors random?
- Which model is most reliable (highest correct percentage)?
- Are there benchmarks where a model has 0% success rate?

---

### **Step 4: Cross-EX Comparison**

**File:** `error_summary_all_ex_{dataset_size}.csv`

**Goal:** Compare difficulty between EX1 (serial) and EX2 (parallel)

**Analysis:**
1. For each benchmark, compare EX1 vs EX2 error rates
2. Identify benchmarks that become harder/easier with parallelization
3. Quantify the impact of adding parallelism

**Key Questions:**
- Does EX2 have higher CF rates than EX1? (Parallelization adds compilation complexity)
- Are there benchmarks where EX2 correct_pct < EX1 correct_pct?
- Which benchmarks show similar error patterns in both EX?

**Example Analysis:**
```
Benchmark: atax

EX1: CF=6.67%,  CE=0.0%,  correct=93.33%
EX2: CF=10.0%,  CE=3.33%, correct=86.67%

Interpretation: Adding OpenMP parallelization slightly increased difficulty:
- Compilation became harder (CF increased)
- Correctness challenges emerged (CE went from 0% to 3.33%)
- Overall success rate dropped by ~7%
```

---

## 📈 Suggested Metrics for Paper

Based on these tables, you can report:

### **1. Overall Statistics**
```
- Average correctness rate across all benchmarks
- Percentage of benchmarks with >80% correctness
- Most common error type (CF vs RF vs CE)
```

### **2. Model Comparison**
```
- Per-model average correctness rate
- Number of benchmarks where each model achieves 100% correctness
- Model complementarity (how often different models succeed on different benchmarks)
```

### **3. Benchmark Difficulty Classification**
```
Based on correct_pct:
- Easy:     correct_pct >= 80%
- Medium:   50% <= correct_pct < 80%
- Hard:     20% <= correct_pct < 50%
- Very Hard: correct_pct < 20%
```

### **4. Error Type Distribution**
```
Average across all benchmarks:
- CF_pct: X%
- RF_pct: Y%
- CE_pct: Z%

Interpretation: Most errors are due to [compilation/runtime/correctness] issues.
```

---

## 🛠️ Useful Commands for Quick Analysis

### **Find top 10 hardest benchmarks (by correct_pct):**
```bash
cat error_summary_all_ex_medium.csv | \
    grep "EX1" | \
    sort -t',' -k11 -n | \
    head -10
```

### **Find benchmarks with highest compilation failure rate:**
```bash
cat error_summary_all_ex_medium.csv | \
    grep "EX1" | \
    sort -t',' -k8 -nr | \
    head -10
```

### **Count benchmarks by difficulty (EX1, correct_pct >= 80%):**
```bash
cat error_summary_all_ex_medium.csv | \
    grep "EX1" | \
    awk -F',' '$11 >= 80.0 {count++} END {print "Easy benchmarks:", count}'
```

### **Compare EX1 vs EX2 average correctness:**
```bash
cat error_summary_all_ex_medium.csv | \
    awk -F',' 'NR>1 {sum[$1]+=$11; count[$1]++} END {
        for (ex in sum) print ex, "avg correct_pct:", sum[ex]/count[ex]
    }'
```

---

## 💡 Tips for Analysis

1. **Start broad, then narrow**: Begin with summary tables, then drill into specific benchmarks
2. **Look for patterns**: Group benchmarks by error type to identify common issues
3. **Compare across dimensions**: Model, EX version, dataset size
4. **Consider context**: High CF might indicate lack of necessary headers/libraries in model knowledge
5. **Report both counts and percentages**: Percentages are easier to compare, but counts show absolute impact

---

## 📊 Example Analysis Workflow

```python
import pandas as pd

# Load summary table
df = pd.read_csv('error_summary_all_ex_medium.csv')

# Filter EX1
df_ex1 = df[df['ex'] == 'EX1']

# Calculate average correctness
avg_correct = df_ex1['correct_pct'].mean()
print(f"EX1 average correctness: {avg_correct:.2f}%")

# Find hardest benchmarks
hardest = df_ex1.nsmallest(10, 'correct_pct')[['benchmark', 'CF_pct', 'CE_pct', 'correct_pct']]
print("\nTop 10 hardest benchmarks:")
print(hardest)

# Error type distribution
avg_cf = df_ex1['CF_pct'].mean()
avg_rf = df_ex1['RF_pct'].mean()
avg_ce = df_ex1['CE_pct'].mean()
print(f"\nError distribution:")
print(f"  CF: {avg_cf:.2f}%")
print(f"  RF: {avg_rf:.2f}%")
print(f"  CE: {avg_ce:.2f}%")
```

---

## 📝 Summary

**For quick overall analysis:**
→ Use `error_summary_all_ex_medium.csv`

**For model comparison:**
→ Use `error_model_breakdown_{EX}_{dataset_size}.csv`

**For detailed investigation:**
→ Use `error_detail_{EX}_{dataset_size}.csv` or `error_detail_all_ex_{dataset_size}.csv`

**Key metrics to report:**
- Average correctness rate per model
- Error type distribution (CF vs RF vs CE)
- Benchmark difficulty classification
- EX1 vs EX2 comparison

---

## ❓ Questions?

If you need clarification on:
- Column definitions → See "Error Type Definitions" section
- Which table to use → See "Quick Start" section
- How to compute metrics → See "Example Analysis Workflow" section
- Interpretation guidelines → See "Recommended Analysis Workflow" section

**Happy analyzing!** 📊
