# BICG Benchmark Makefile User Guide

## Overview

This Makefile provides a standardized build system for the BICG (BiConjugate Gradient) benchmark, supporting:
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

## Algorithm Overview

BICG (BiConjugate Gradient) is a linear algebra kernel that performs:
- **Operation**: s = A^T × r and q = A × p
- **Input**: Matrix A (M × N), vectors r (M), p (N)
- **Output**: Vectors s (N) and q (M)
- **Steps**:
  1. Compute q = A × p (matrix-vector multiplication)
  2. Compute s = A^T × r (transposed matrix-vector multiplication)
- **Complexity**: O(M × N)

## Dataset Sizes

| Size | Macro Definition | M | N | Description |
|------|------------------|---|---|-------------|
| mini | `MINI_DATASET` | 38 | 42 | Quick testing |
| small | `SMALL_DATASET` | 116 | 124 | Small-scale testing |
| medium | `MEDIUM_DATASET` | 390 | 410 | Medium-scale testing |
| large | `LARGE_DATASET` | 1900 | 2100 | Large-scale testing |
| extra-large | `EXTRALARGE_DATASET` | 1800 | 2200 | Extra large-scale testing |

**Note**: The header file uses `EXTRALARGE_DATASET` (no underscore), but binaries use `extra-large` (with hyphen) for consistency.

## Binary Naming Convention

Format: `<source_base>_<compiler>_<dataset_size>`

Examples:
```
bicg_gcc_mini
bicg_gcc_large
bicg_claude_v1_gcc_mini
bicg_claude_v1_gcc_extra-large
bicg_llama4_v5_gcc_medium
```

## Make Targets

### Build Targets

| Target | Description |
|--------|-------------|
| `make` or `make all` | Build all variants (21 × 5 = 105 binaries) |
| `make baseline` | Build only baseline (bicg.c) for all 5 sizes |
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

## Source Files

- **Total**: 21 source files
  - 1 Baseline: `bicg.c`
  - 10 Claude variants: `bicg_claude_v1.c` ~ `bicg_claude_v10.c`
  - 10 Llama4 variants: `bicg_llama4_v1.c` ~ `bicg_llama4_v10.c`
- **Generated binaries**: 105 (21 × 5 sizes)
- **All variants compile successfully**

## Compilation Examples

```bash
# Compile baseline only
make baseline

# Compile all variants
make

# Compile specific size
make EX1_optimized_codes/bicg_gcc_large

# Run tests
make test-all
```

## Running Examples

```bash
# Run baseline mini
./EX1_optimized_codes/bicg_gcc_mini

# Save output to file
./EX1_optimized_codes/bicg_gcc_mini > output.txt 2>&1

# Measure execution time
time ./EX1_optimized_codes/bicg_gcc_large
```

## Output Format

```
==BEGIN DUMP_ARRAYS==
begin dump: s
<vector s data>
end   dump: s
begin dump: q
<vector q data>
end   dump: q
==END   DUMP_ARRAYS==
```

## Performance Estimates

| Dataset | M | N | Estimated Time |
|---------|---|---|----------------|
| mini | 38 | 42 | < 0.01s |
| small | 116 | 124 | ~0.01s |
| medium | 390 | 410 | ~0.1s |
| large | 1900 | 2100 | ~2s |
| extra-large | 1800 | 2200 | ~2s |

## Consistency with Other Benchmarks

| Benchmark | Header Macro | Binary Suffix | Status |
|-----------|--------------|---------------|--------|
| 2mm | `EXTRA_LARGE_DATASET` | `_extra-large` | ✅ |
| 3mm | `EXTRALARGE_DATASET` | `_extra-large` | ✅ |
| ADI | `EXTRALARGE_DATASET` | `_extra-large` | ✅ |
| ATAX | `EXTRALARGE_DATASET` | `_extra-large` | ✅ |
| BICG | `EXTRALARGE_DATASET` | `_extra-large` | ✅ |

## Summary

- ✅ All 21 variants compile successfully
- ✅ 105 binaries (21 × 5 sizes)
- ✅ Standardized naming convention
- ✅ Error-tolerant compilation
- ✅ Serial execution (no OpenMP)
- ✅ Complete testing support

For detailed usage and troubleshooting, see the main template guide at `.cursor/commands/polybench-EX1-makefile.md`.
