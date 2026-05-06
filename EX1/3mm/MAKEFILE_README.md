# 3mm Benchmark Makefile User Guide

## Overview

This Makefile provides a standardized build system for the 3mm benchmark, supporting:
- Automatic compiler detection (gcc/clang)
- Error-tolerant compilation (failures don't interrupt the process)
- Multiple dataset sizes (mini, small, medium, large, extra-large)
- Serial compilation (no OpenMP)

## Quick Start

```bash
# Show help information
make help

# Build all variants (default)
make

# Build baseline only
make baseline

# Clean all binaries
make clean
```

## Project Structure

```
3mm/
├── Makefile                     # Main build script
├── 3mm.h                        # Dataset size definitions
├── 3mm.c                        # Original benchmark (reference)
├── EX1_optimized_codes/
│   ├── 3mm.c                   # Baseline version
│   ├── 3mm_claude_v1.c         # Claude optimization v1
│   ├── ...                     # Claude v2-v10
│   ├── 3mm_llama4_v1.c         # Llama4 optimization v1
│   └── ...                     # Llama4 v2-v10
└── EX5_correctness/            # Correctness verification scripts
```

## Source File Statistics

- **Total**: 21 source files
  - 1 Baseline: `3mm.c`
  - 10 Claude variants: `3mm_claude_v1.c` ~ `3mm_claude_v10.c`
  - 10 Llama4 variants: `3mm_llama4_v1.c` ~ `3mm_llama4_v10.c`

- **Generated binaries**: 105 (21 × 5 sizes)

## Dataset Sizes

| Size | Macro Definition | NI | NJ | NK | NL | NM |
|------|------------------|-----|-----|-----|-----|-----|
| mini | `MINI_DATASET` | 16 | 18 | 20 | 22 | 24 |
| small | `SMALL_DATASET` | 40 | 50 | 60 | 70 | 80 |
| medium | `MEDIUM_DATASET` | 180 | 190 | 200 | 210 | 220 |
| large | `LARGE_DATASET` | 800 | 900 | 1000 | 1100 | 1200 |
| extra-large | `EXTRALARGE_DATASET` | 1600 | 1800 | 2000 | 2200 | 2400 |

**Note**: Although the header file `3mm.h` uses `EXTRALARGE_DATASET` (no hyphen) for the macro definition, the binary file naming uses `extra-large` (with hyphen) to maintain consistency with 2mm.

## Make Target Reference

### Build Targets

| Target | Description |
|--------|-------------|
| `make` or `make all` | Build all variants with all dataset sizes (default) |
| `make variants` | Same as above |
| `make baseline` | Build only baseline binaries (3mm.c) for all 5 sizes |
| `make clean` | Remove all generated binaries |

### Test Targets

| Target | Description |
|--------|-------------|
| `make test-mini` | Run baseline with mini dataset |
| `make test-small` | Run baseline with small dataset |
| `make test-medium` | Run baseline with medium dataset |
| `make test-large` | Run baseline with large dataset |
| `make test-extra-large` | Run baseline with extra-large dataset |
| `make test-all` | Run all tests |

## Binary Naming Convention

Format: `<source_base>_<compiler>_<dataset_size>`

Examples:
```
3mm_gcc_mini                    # Baseline mini
3mm_gcc_large                   # Baseline large
3mm_claude_v1_gcc_mini          # Claude v1 mini
3mm_claude_v1_gcc_extra-large   # Claude v1 extra-large
3mm_llama4_v5_gcc_medium        # Llama4 v5 medium
```

## Compilation Details

### Compilation Flags
```makefile
CFLAGS := -O3
INCLUDES := -I../../utilities -I.
LIBS := -lm
```

### PolyBench Integration
Each binary is linked with the PolyBench utility library:
```
../../utilities/polybench.c
```

### Compilation Command Examples
```bash
# Mini dataset
gcc -DMINI_DATASET -O3 -I../../utilities -I. \
    EX1_optimized_codes/3mm.c ../../utilities/polybench.c \
    -lm -o EX1_optimized_codes/3mm_gcc_mini

# Extra-large dataset
gcc -DEXTRALARGE_DATASET -O3 -I../../utilities -I. \
    EX1_optimized_codes/3mm_claude_v1.c ../../utilities/polybench.c \
    -lm -o EX1_optimized_codes/3mm_claude_v1_gcc_extra-large
```

## Running Examples

### Run and View Output
```bash
# Run baseline mini
./EX1_optimized_codes/3mm_gcc_mini

# Run a variant
./EX1_optimized_codes/3mm_claude_v1_gcc_large
```

### Save Output to File
```bash
# Save complete output (including markers)
./EX1_optimized_codes/3mm_gcc_mini > output.txt 2>&1

# Or use tee to display and save simultaneously
./EX1_optimized_codes/3mm_gcc_mini 2>&1 | tee output.txt
```

### Measure Execution Time
```bash
# Using time command
time ./EX1_optimized_codes/3mm_gcc_large

# Or using Python script
python3 << 'EOF'
import subprocess
import time

start = time.time()
result = subprocess.run(
    ['./EX1_optimized_codes/3mm_gcc_large'],
    capture_output=True,
    text=True
)
elapsed = time.time() - start

print(f"Elapsed: {elapsed:.3f}s")
with open('output.txt', 'w') as f:
    f.write(result.stdout)
EOF
```

## Correctness Verification

### Output Format
The program output follows the PolyBench standard format:
```
==BEGIN DUMP_ARRAYS==
begin dump: G
<numerical data>
end   dump: G
==END   DUMP_ARRAYS==
```

### Extract Data for Comparison
```python
import re

def extract_array_data(output_text):
    """Extract array data from output"""
    pattern = r'begin dump: (\w+)\s+(.*?)\s+end\s+dump: \1'
    match = re.search(pattern, output_text, re.DOTALL)
    if match:
        array_name = match.group(1)
        data = match.group(2).strip()
        numbers = [float(x) for x in data.split()]
        return array_name, numbers
    return None, None

# Compare baseline and variant
with open('baseline_output.txt') as f:
    _, baseline_nums = extract_array_data(f.read())

with open('variant_output.txt') as f:
    _, variant_nums = extract_array_data(f.read())

# Calculate difference
max_diff = max(abs(b - v) for b, v in zip(baseline_nums, variant_nums))
print(f"Max difference: {max_diff:.2e}")

# Verify correctness (typical tolerance=1e-6)
is_correct = max_diff < 1e-6
print(f"Correctness: {'✓ PASS' if is_correct else '✗ FAIL'}")
```

## Error Handling

### Compilation Errors
The Makefile uses error-tolerant design. When a variant fails to compile:
1. Print error message
2. Continue compiling other variants
3. Report success/failure statistics at the end

Example output:
```
[5/20] Building 3mm_claude_v3...
  ✗ Failed to build 3mm_claude_v3 (continuing...)
[6/20] Building 3mm_claude_v4...
  Compiling 3mm_claude_v4 (mini)...
  ...
⚠ 1 variant(s) failed to compile (continued with others)
  Successfully built: 19/20 variants
```

### Runtime Errors
If a program crashes or times out during execution:
```bash
# Use timeout to limit execution time
timeout 300 ./EX1_optimized_codes/3mm_gcc_large

# Or use subprocess.run with timeout in Python
result = subprocess.run(
    ['./EX1_optimized_codes/3mm_gcc_large'],
    capture_output=True,
    timeout=300  # 5 minutes
)
```

## Performance Analysis

### Compilation Time
Complete compilation of all 105 binaries takes approximately **~20 seconds** (varies by machine).

### Estimated Runtime

| Dataset | Estimated Time |
|---------|---------------|
| mini | < 0.01s |
| small | ~0.1s |
| medium | ~2s |
| large | ~50s |
| extra-large | ~400s |

Actual runtime depends on optimization quality and hardware performance.

## Advanced Features

### Modify Compilation Options
Edit `CFLAGS` in Makefile:
```makefile
# Add extra optimization options
CFLAGS := -O3 -march=native -funroll-loops
```

### Change Compiler
```bash
# Use clang for compilation
CC=clang make clean
CC=clang make
```

Or modify compiler detection order:
```makefile
# Prefer clang
CC := $(shell command -v clang 2>/dev/null)
ifdef CC
    COMPILER_TAG := clang
else
    CC := $(shell command -v gcc 2>/dev/null)
    ...
```

### Parallel Compilation
Use `-j` flag to speed up compilation:
```bash
# Use 4 parallel jobs
make -j4

# Auto-detect CPU cores
make -j$(nproc)
```

## Differences from 2mm

| Feature | 2mm | 3mm |
|---------|-----|-----|
| Dataset Macro | `EXTRA_LARGE_DATASET` | `EXTRALARGE_DATASET` |
| Binary Suffix | `*_extra-large` | `*_extra-large` |
| Make Target | `test-extra-large` | `test-extra-large` |
| Source Files | 21 | 21 |
| Number of Matrices | 4 (A, B, C, D) | 7 (A-G) |

**Important Note**: Although 2mm and 3mm use different macro definitions in their header files (`EXTRA_LARGE_DATASET` vs `EXTRALARGE_DATASET`), both use `*_extra-large` (with hyphen) for binary file naming to maintain a consistent naming convention.

## Troubleshooting

### Issue 1: "No compiler found"
**Cause**: gcc or clang not installed on the system  
**Solution**:
```bash
# CentOS/RHEL
sudo yum install gcc

# Ubuntu/Debian
sudo apt-get install gcc

# macOS
xcode-select --install
```

### Issue 2: "polybench.c: No such file"
**Cause**: Incorrect PolyBench utility library path  
**Solution**: Ensure `../../utilities/polybench.c` exists

### Issue 3: Cannot run after compilation
**Cause**: Missing execution permissions  
**Solution**:
```bash
chmod +x EX1_optimized_codes/*_gcc_*
```

### Issue 4: clean target doesn't remove all files
**Cause**: File permissions or filename mismatch  
**Solution**:
```bash
# Manual cleanup
rm -f EX1_optimized_codes/*_gcc_*
rm -f EX1_optimized_codes/*_clang_*
```

## Best Practices

1. **First-time use: compile baseline first**
   ```bash
   make baseline
   make test-mini  # Verify it runs correctly
   ```

2. **Clean before batch compilation**
   ```bash
   make clean
   make
   ```

3. **Save all outputs for comparison**
   ```bash
   mkdir -p outputs
   for bin in EX1_optimized_codes/*_gcc_mini; do
       name=$(basename $bin)
       ./$bin > outputs/${name}.txt 2>&1
   done
   ```

4. **Use Python scripts for automated verification**
   ```python
   import subprocess
   from pathlib import Path
   
   binaries = Path('EX1_optimized_codes').glob('*_gcc_mini')
   for binary in binaries:
       result = subprocess.run(
           [str(binary)],
           capture_output=True,
           text=True,
           timeout=60
       )
       # Process output...
   ```

## Summary

This Makefile provides:
- ✅ Automated build process
- ✅ Error tolerance mechanism
- ✅ Standardized naming convention
- ✅ Easy to extend and maintain
- ✅ Complete testing support

Follow this guide to efficiently compile and test all 3mm benchmark variants.
