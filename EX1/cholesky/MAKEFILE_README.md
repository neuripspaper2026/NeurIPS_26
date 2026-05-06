# Cholesky Benchmark Makefile User Guide

## Overview

Standardized build system for Cholesky decomposition benchmark supporting:
- Automatic compiler detection (gcc/clang)
- Error-tolerant compilation
- 5 dataset sizes (mini, small, medium, large, extra-large)
- Serial compilation (no OpenMP)

## Quick Start

```bash
make              # Build all variants
make baseline     # Build baseline only
make clean        # Remove binaries
```

## Algorithm

Cholesky decomposition factors a symmetric positive-definite matrix A into L × L^T where L is lower triangular.

## Dataset Sizes

| Size | N | Description |
|------|---|-------------|
| mini | 40 | Quick testing |
| small | 120 | Small-scale |
| medium | 400 | Medium-scale |
| large | 2000 | Large-scale |
| extra-large | 4000 | Extra large-scale |

**Note**: Header uses `EXTRALARGE_DATASET`, binaries use `extra-large`.

## Source Files

- **Total**: 15 source files
  - 1 Baseline: `cholesky.c`
  - 7 Claude variants: `cholesky_claude_v1-v7.c`
  - 7 Llama4 variants: `cholesky_llama4_v1-v7.c`
- **Generated binaries**: 75 (15 × 5 sizes)

## Binary Naming

Format: `<source>_<compiler>_<size>`

Examples:
```
cholesky_gcc_mini
cholesky_claude_v1_gcc_large
cholesky_llama4_v3_gcc_extra-large
```

## Make Targets

| Target | Description |
|--------|-------------|
| `make` | Build all variants (75 binaries) |
| `make baseline` | Build baseline for all 5 sizes |
| `make clean` | Remove all binaries |
| `make test-all` | Run all tests |

## Running

```bash
./EX1_optimized_codes/cholesky_gcc_mini
./EX1_optimized_codes/cholesky_gcc_large > output.txt 2>&1
time ./EX1_optimized_codes/cholesky_gcc_extra-large
```

## Consistency

| Benchmark | Header Macro | Binary Suffix | Sources | Status |
|-----------|--------------|---------------|---------|--------|
| 2mm | `EXTRA_LARGE_DATASET` | `_extra-large` | 21 | ✅ |
| 3mm | `EXTRALARGE_DATASET` | `_extra-large` | 21 | ✅ |
| ADI | `EXTRALARGE_DATASET` | `_extra-large` | 21 | ✅ |
| ATAX | `EXTRALARGE_DATASET` | `_extra-large` | 21 | ✅ |
| BICG | `EXTRALARGE_DATASET` | `_extra-large` | 21 | ✅ |
| Cholesky | `EXTRALARGE_DATASET` | `_extra-large` | 15 | ✅ |

## Summary

- ✅ 15 variants
- ✅ 75 binaries (15 × 5 sizes)
- ✅ Standardized naming
- ✅ Serial execution

For details, see `.cursor/commands/polybench-EX1-makefile.md`.
