# 2mm Benchmark

## Overview

This is the 2mm (two matrix multiplications) benchmark, including the baseline and multiple optimized variants.

## Quick Start

```bash
# 1. Build all variants and all dataset sizes
make

# 2. Run the baseline
./EX1_optimized_codes/2mm_gcc_mini
./EX1_optimized_codes/2mm_gcc_large

# 3. Run variants
./EX1_optimized_codes/2mm_claude_v1_gcc_mini
./EX1_optimized_codes/2mm_llama4_v2_gcc_large
```

## Dataset Sizes

| Size | NI | NJ | NK | NL | Est. runtime |
|------|----|----|----|----|--------------|
| mini | 16 | 18 | 22 | 24 | ~0.002s |
| small | 40 | 50 | 70 | 80 | ~0.005s |
| medium | 180 | 190 | 210 | 220 | ~0.07s |
| large | 800 | 900 | 1100 | 1200 | ~3.7s |
| extralarge | 1600 | 1800 | 2200 | 2400 | ~30s |

## Makefile Commands

### Basic

```bash
make              # Build all variants and all sizes (21 sources × 5 sizes = 105 binaries)
make all          # Same as above
make baseline     # Build only the baseline (2mm.c) for 5 sizes
make variants     # Build baseline + all variants
make clean        # Clean all generated binaries
make help         # Show help
```

### Test

```bash
make test-mini        # Test mini
make test-small       # Test small
make test-medium      # Test medium
make test-large       # Test large
make test-extralarge  # Test extralarge
make test-all         # Test all sizes
```

## Binary Naming

Generated binaries follow:

```
<source_base>_<compiler>_<size>
```

Examples:
- `2mm_gcc_mini` - baseline mini build with gcc
- `2mm_gcc_large` - baseline large build with gcc
- `2mm_claude_v1_gcc_mini` - claude_v1 variant mini build
- `2mm_llama4_v2_gcc_large` - llama4_v2 variant large build

## Usage Examples

### Example 1: Build and run baseline

```bash
# Build baseline
make baseline

# Run different sizes
./EX1_optimized_codes/2mm_gcc_mini
./EX1_optimized_codes/2mm_gcc_small
./EX1_optimized_codes/2mm_gcc_medium
./EX1_optimized_codes/2mm_gcc_large
./EX1_optimized_codes/2mm_gcc_extralarge
```

### Example 2: Build all variants

```bash
# Build everything
make

# This produces:
# - 5 baseline binaries (2mm_gcc_mini, _small, _medium, _large, _extralarge)
# - 5 binaries for each variant (e.g., 2mm_claude_v1_gcc_mini, ...)
# - Total: 21 sources × 5 sizes = 105 binaries
```

### Example 3: Run variants

```bash
# Claude variants
./EX1_optimized_codes/2mm_claude_v1_gcc_mini
./EX1_optimized_codes/2mm_claude_v1_gcc_large

# Llama4 variants
./EX1_optimized_codes/2mm_llama4_v2_gcc_mini
./EX1_optimized_codes/2mm_llama4_v2_gcc_large
```

### Example 4: Performance testing

```bash
# Measure runtime for different sizes
for size in mini small medium large; do
    echo "Testing $size:"
    time ./EX1_optimized_codes/2mm_gcc_$size > /dev/null
done
```

## Directory Layout

```
2mm/
├── Makefile                              # Main Makefile
├── README.md                             # This document
├── MAKEFILE_README.md                    # Makefile details
├── 2mm.c                                 # Baseline source
├── 2mm.h                                 # Header
├── EX1_optimized_codes/
│   ├── 2mm.c                            # Baseline source
│   ├── 2mm_claude_v1.c                  # Claude variant 1
│   ├── 2mm_claude_v2.c                  # Claude variant 2
│   ├── ...                              # More variants
│   ├── 2mm_gcc_mini                     # Built binaries
│   ├── 2mm_gcc_small
│   ├── 2mm_gcc_medium
│   ├── 2mm_gcc_large
│   ├── 2mm_gcc_extralarge
│   ├── 2mm_claude_v1_gcc_mini           # Variant binaries
│   └── ...                              # More binaries
└── EX5_correctness/                     # Correctness checks
```

## Compiler Support

The Makefile auto-detects available compilers:
- Prefer `gcc`
- If gcc is unavailable, fall back to `clang`
- Binaries are marked as `*_gcc_*` or `*_clang_*` accordingly

## Makefile Features

### 1. Auto compiler detection

Automatically choose gcc or clang based on availability.

### 2. Error handling

If a variant fails to build, others continue, and failures are summarized at the end.

### 3. Dynamic source detection

Automatically detect all `.c` files under `EX1_optimized_codes/` and build them.

### 4. Dataset size macros

Use preprocessor macros for dataset sizes:
- `-DMINI_DATASET`
- `-DSMALL_DATASET`
- `-DMEDIUM_DATASET`
- `-DLARGE_DATASET`
- `-DEXTRALARGE_DATASET`

## Troubleshooting

### Issue: Compiler not found

**Error**: `No compiler found`

**Fix**: Install gcc or clang

```bash
# Ubuntu/Debian
sudo apt-get install gcc

# CentOS/RHEL
sudo yum install gcc
```

### Issue: Build failures

**Error**: Some variants fail to compile

**Fix**: This is expected for buggy variants. `make variants` keeps building others and reports failures at the end.

### Issue: Segmentation fault

**Possible cause**: extralarge dataset may need lots of memory

**Fix**:
```bash
# Increase stack limit
ulimit -s unlimited
./EX1_optimized_codes/2mm_gcc_extralarge
```

## Performance Notes

Performance varies across variants:
- Baseline is the unoptimized reference
- Claude variants are Claude-generated optimizations
- Llama4 variants are Llama4-generated optimizations

Use the same dataset sizes to compare performance fairly.

## License

Follows the PolyBench/C license.

## Contact

- PolyBench: http://polybench.sourceforge.net
