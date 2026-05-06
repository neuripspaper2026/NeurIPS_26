# EX3 Configuration Files

This directory contains configuration files specific to EX3 (CUDA kernel optimization) benchmarks.

## Files Overview

### 1. `benchmark_args_ex3.json`
**Purpose:** Command-line arguments for each benchmark and dataset size

**Format:**
```json
{
  "benchmark_name": {
    "mini": ["arg1", "arg2", ...],
    "small": ["arg1", "arg2", ...],
    "medium": ["arg1", "arg2", ...],
    "large": ["arg1", "arg2", ...],
    "extra-large": ["arg1", "arg2", ...]
  }
}
```

**Example:**
```json
{
  "bfs": {
    "mini": ["./input_data/mini/input/graph1M.txt", "-o", "/dev/null"],
    "small": ["./input_data/small/input/graph2M.txt", "-o", "/dev/null"]
  }
}
```

**Status:** Template created, needs to be filled in with actual arguments

---

### 2. `benchmark_catalog_ex3.py`
**Purpose:** Code location metadata (start_line, end_line, target_file) for EX3 kernels

**Format:**
```python
BENCHMARK_RODINIA_EX3: BenchmarkMeta = {
    "benchmark_name": {
        "target_file": "kernel_file.cu",  # or list for multiple files
        "threshold": 0.002,
        "start_line": 1,  # or list for multiple files
        "end_line": 100   # or list for multiple files
    }
}
```

**Example:**
```python
"bilateral": {
    "target_file": "EX3_optimized_codes/bilateral_kernel.cu",
    "threshold": 0.002,
    "start_line": 80,
    "end_line": 106
}
```

**Status:** Template created, needs to be filled in with actual line numbers

---

### 3. `head_code_ex3.py`
**Purpose:** Header file dependencies for each EX3 benchmark

**Format:**
```python
def resolve_head_code_paths_ex3(manager, benchmark_name: str) -> List[Path]:
    extra_map = {
        "benchmark_name": [
            "EX3/benchmark_name/header1.h",
            "EX3/benchmark_name/header2.h",
        ]
    }
```

**Example:**
```python
"bilateral": [
    "EX3/bilateral/add_info.txt",
],
"leukocyte": [
    "EX3/leukocyte/track-ellipse.h",
]
```

**Status:** Template created with some known headers, needs completion

---

### 4. `runtime_hints_ex3.json`
**Purpose:** Expected runtime estimates for timeout configuration

**Format:**
```json
{
  "benchmark_name": {
    "mini": 0.5,
    "small": 1.0,
    "medium": 2.0,
    "large": 5.0,
    "extra-large": 10.0
  }
}
```

**Status:** ✅ Already completed

---

## Key Differences: EX3 vs EX1/EX2

| Aspect | EX1/EX2 | EX3 |
|--------|---------|-----|
| **Code Type** | C/C++ CPU code | CUDA GPU kernels |
| **Target File** | Main program file | Kernel file in `EX3_optimized_codes/` |
| **File Location** | `EX1/<benchmark>/` or `EX2/<benchmark>/` | `EX3/<benchmark>/EX3_optimized_codes/` |
| **Headers** | Shared utilities (polybench.h, etc.) | CUDA-specific headers |
| **Start/End Line** | Points to CPU algorithm | Points to CUDA kernel function |

### Example: bilateral benchmark

**EX1/EX2:**
- Target file: `EX1/bilateral/bilateralFilter-cpu.cpp`
- Lines: 95-160 (CPU filtering algorithm)

**EX3:**
- Target file: `EX3/bilateral/EX3_optimized_codes/bilateral_kernel.cu`
- Lines: 80-106 (CUDA kernel `d_bilateral_filter`)

---

## How to Fill In Configuration

### Step 1: Fill `benchmark_args_ex3.json`

For each benchmark, determine command-line arguments:

```bash
# Go to benchmark directory
cd EX3/<benchmark>

# Check run script or README
cat run
cat README

# Example for bfs:
./bfs ./input_data/mini/input/graph1M.txt -o /dev/null
```

Then add to JSON:
```json
"bfs": {
  "mini": ["./input_data/mini/input/graph1M.txt", "-o", "/dev/null"]
}
```

### Step 2: Fill `benchmark_catalog_ex3.py`

For each benchmark, locate the main CUDA kernel:

```bash
# Find kernel file
ls EX3/<benchmark>/EX3_optimized_codes/

# Open kernel file and find __global__ function
# Example: bilateral_kernel.cu line 80-106
```

Then update in Python:
```python
"bilateral": {
    "target_file": "EX3_optimized_codes/bilateral_kernel.cu",
    "threshold": 0.002,
    "start_line": 80,
    "end_line": 106
}
```

**Note for multi-kernel benchmarks:**
```python
"b+tree": {
    "target_file": ["EX3_optimized_codes/kernel_gpu_cuda.cu", 
                    "EX3_optimized_codes/kernel_gpu_cuda_2.cu"],
    "threshold": 0.002,
    "start_line": [1, 1],
    "end_line": [44, 63]
}
```

### Step 3: Fill `head_code_ex3.py`

Check if benchmark has header files:

```bash
# Look for .h files
ls EX3/<benchmark>/*.h
ls EX3/<benchmark>/*.c  # Sometimes .c files are included as headers

# Check add_info.txt or similar documentation
ls EX3/<benchmark>/add_info.txt
```

Then update:
```python
"benchmark_name": [
    "EX3/benchmark_name/header1.h",
    "EX3/benchmark_name/add_info.txt",
]
```

---

## Quick Reference: Completed Benchmarks

From your work so far, here are some known values:

### bilateral
- **Args:** See existing `benchmark_args.json` (lines 1051-1087)
- **Target:** `EX3_optimized_codes/bilateral_kernel.cu` line 80 (`d_bilateral_filter` function)
- **Headers:** `add_info.txt`

### b+tree
- **Kernels:** 
  - `EX3_optimized_codes/kernel_gpu_cuda.cu` (entire file, ~44 lines)
  - `EX3_optimized_codes/kernel_gpu_cuda_2.cu` (entire file, ~63 lines)

### backprop
- **Kernel:** `EX3_optimized_codes/backprop_cuda_kernel.cu` (entire file, ~90 lines)
- **Headers:** `backprop.h`

### bfs
- **Kernels:**
  - `EX3_optimized_codes/kernel.cu` (entire file, ~23 lines - `Kernel` function)
  - `EX3_optimized_codes/kernel2.cu` (entire file, ~39 lines - `Kernel2` function)

---

## TODO List

- [ ] Fill in `benchmark_args_ex3.json` with actual command-line arguments
- [ ] Fill in `benchmark_catalog_ex3.py` with start_line/end_line for all kernels
- [ ] Complete `head_code_ex3.py` with all necessary headers
- [ ] Test configuration with pipeline scripts
- [ ] Update main pipeline to use EX3 configs

---

## Notes

1. **Kernel Files:** EX3 kernels are in `EX3_optimized_codes/` subdirectory, not root
2. **Multiple Kernels:** Some benchmarks (b+tree, bfs) have multiple kernel files
3. **Time Measurement:** All EX3 benchmarks should output `KERNEL_TIME` and `TOTAL_TIME`
4. **Arguments:** Most benchmarks use `-o /dev/null` or `-o <output_file>` format

---

## Integration

Once configurations are complete, update main scripts to use EX3 configs:

```python
# In your main pipeline script
if experiment == "EX3":
    from configs.EX3_configs.benchmark_catalog_ex3 import BENCHMARK_RODINIA_EX3
    from configs.EX3_configs.head_code_ex3 import resolve_head_code_paths_ex3
    import json
    
    with open("configs/EX3_configs/benchmark_args_ex3.json") as f:
        args_config = json.load(f)
```
