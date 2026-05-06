# Correlation Benchmark - Makefile User Guide

## Overview

This directory contains the standardized Makefile for the `correlation` benchmark from the PolyBench suite. The Makefile automates the compilation of multiple source file variants across 5 different dataset sizes.

## Benchmark Information

- **Benchmark Name**: Correlation
- **Source Files**: 21 total
  - 1 baseline: `correlation.c`
  - 10 Claude variants: `correlation_claude_v1.c` to `correlation_claude_v10.c`
  - 10 Llama4 variants: `correlation_llama4_v1.c` to `correlation_llama4_v10.c`
- **Successfully Compiled**: 14 sources (1 baseline + 10 Claude + 3 Llama4)
- **Compilation Errors**: 7 sources (Llama4 v1, v5, v6, v7, v8, v9, v10)
- **Dataset Sizes**: 5 (mini, small, medium, large, extra-large)
- **Total Binaries**: 70 (14 successful sources × 5 sizes)
- **Compilation Mode**: Serial (no OpenMP)

## Quick Start

### Build All Binaries

```bash
make
```

This will compile all source files for all 5 dataset sizes. Successfully compiles 14 out of 21 sources, generating 70 binaries (7 sources fail to compile and are automatically skipped).

### Build Only Baseline

```bash
make baseline
```

This will compile only the baseline `correlation.c` for all 5 dataset sizes.

### Run Quick Test

```bash
make test-mini
```

### Clean Up

```bash
make clean
```

## Available Commands

| Command | Description |
|---------|-------------|
| `make` | Build all variants and all sizes (default) |
| `make all` | Same as `make` |
| `make variants` | Build all variant binaries |
| `make baseline` | Build only baseline binaries (5 total) |
| `make test-mini` | Run baseline with mini dataset |
| `make test-small` | Run baseline with small dataset |
| `make test-medium` | Run baseline with medium dataset |
| `make test-large` | Run baseline with large dataset |
| `make test-extra-large` | Run baseline with extra-large dataset |
| `make test-all` | Run all test sizes |
| `make clean` | Remove all generated binaries |
| `make help` | Show help message with system info |

## Binary Naming Convention

All binaries follow the standardized naming pattern:

```
<source_base>_<compiler>_<size>
```

**Examples**:
- `correlation_gcc_mini` - baseline with mini dataset
- `correlation_gcc_extra-large` - baseline with extra-large dataset
- `correlation_claude_v3_gcc_large` - Claude variant 3 with large dataset
- `correlation_llama4_v7_gcc_small` - Llama4 variant 7 with small dataset

## Dataset Sizes

| Size | Macro | M | N | Array Size |
|------|-------|---|---|------------|
| mini | `MINI_DATASET` | 28 | 32 | ~3KB |
| small | `SMALL_DATASET` | 80 | 100 | ~24KB |
| medium | `MEDIUM_DATASET` | 240 | 260 | ~237KB |
| large | `LARGE_DATASET` | 1200 | 1400 | ~6.4MB |
| extra-large | `EXTRALARGE_DATASET` | 2600 | 3000 | ~30.4MB |

## Technical Details

### Compilation Strategy

- **One binary per source file per dataset size**: Each combination of source file and dataset size produces a separate executable
- **Compile-time constants**: Dataset sizes are defined using `-D` macros for optimal performance
- **No intermediate files**: Source files are compiled directly to binaries without `.o` files
- **Serial execution**: No OpenMP or multi-threading support

### Compiler Detection

The Makefile automatically detects available compilers in this order:
1. `gcc` (preferred)
2. `clang` (fallback)

The detected compiler is used in binary names (e.g., `_gcc_` or `_clang_`).

### Error Handling

The Makefile is designed to be robust:
- If a variant fails to compile, the build continues with other variants
- A summary report shows which variants succeeded/failed
- Use `make clean` before rebuilding if you encounter issues

## Running Binaries

### Basic Execution

```bash
./EX1_optimized_codes/correlation_gcc_mini
```

### Redirect Output to File

```bash
./EX1_optimized_codes/correlation_gcc_large > output.txt
```

### Measure Execution Time

```bash
time ./EX1_optimized_codes/correlation_gcc_large
```

## Project Structure

```
correlation/
├── Makefile                          # This standardized Makefile
├── MAKEFILE_README.md                # This file
├── correlation.h                     # Header with dataset size definitions
├── correlation.c                     # Baseline source code
├── EX1_optimized_codes/              # Source files directory
│   ├── correlation.c                 # Baseline (symlink or copy)
│   ├── correlation_claude_v*.c       # Claude-optimized variants
│   ├── correlation_llama4_v*.c       # Llama4-optimized variants
│   └── *_gcc_*                       # Generated binaries (105 total)
└── input_data/                       # Input data for different sizes
    ├── mini/
    ├── small/
    ├── medium/
    ├── large/
    └── extra-large/
```

## Compilation Details

### Compiler Flags

```bash
CFLAGS = -O3
INCLUDES = -I../../utilities -I.
LIBS = -lm
```

### No OpenMP

This Makefile compiles for **serial execution only**:
- No `-fopenmp` flag
- No `-lpthread` flag
- Single-threaded execution for fair performance comparison

### Dataset Macro Mapping

The Makefile uses the `EXTRALARGE_DATASET` macro (no underscore between EXTRA and LARGE) as defined in `correlation.h`, but the binary suffix uses `extra-large` (with hyphen) for consistency across all benchmarks.

## Troubleshooting

### Issue: Compilation Errors

**Solution**: Check if a specific variant has syntax errors. The Makefile will skip failed variants and continue building others.

### Issue: "No compiler found" Error

**Solution**: Install gcc or clang:

```bash
# On Ubuntu/Debian
sudo apt-get install gcc

# On CentOS/RHEL
sudo yum install gcc
```

### Issue: Binaries Not Generated

**Solution**: Run `make clean` first, then `make` again:

```bash
make clean
make
```

### Issue: Permission Denied When Running Binary

**Solution**: Ensure binaries are executable:

```bash
chmod +x EX1_optimized_codes/*_gcc_*
```

## Performance Testing

To measure performance across different variants:

```bash
# Test all variants with large dataset
for binary in EX1_optimized_codes/*_gcc_large; do
    echo "Testing $binary"
    time ./$binary > /dev/null
done
```

## Known Issues

### Compilation Errors

**7 variants fail to compile** (source files exist but binaries cannot be generated):

- ✗ `correlation_llama4_v1.c`
- ✗ `correlation_llama4_v5.c`
- ✗ `correlation_llama4_v6.c`
- ✗ `correlation_llama4_v7.c`
- ✗ `correlation_llama4_v8.c`
- ✗ `correlation_llama4_v9.c`
- ✗ `correlation_llama4_v10.c`

**14 variants compile successfully**:
- ✓ baseline: `correlation.c`
- ✓ All 10 Claude variants (v1-v10)
- ✓ 3 Llama4 variants: v2, v3, v4

The Makefile's error handling mechanism automatically skips these problematic files and continues building the remaining variants.

## Notes

- **14 out of 21** source files compile successfully, generating **70 binaries** (14 × 5 sizes)
- 7 Llama4 variants contain compilation errors and are automatically skipped
- Binary files are generated in the `EX1_optimized_codes/` directory
- The Makefile follows the standardized template used across all PolyBench benchmarks in this project
- For correctness verification, compare outputs using the `EX5_correctness/` directory

## Summary

This standardized Makefile provides:
- ✅ Automated compilation for 21 source files (14 compile successfully)
- ✅ Support for 5 dataset sizes
- ✅ Robust error handling (automatically skips 7 failing variants)
- ✅ Consistent binary naming
- ✅ Serial compilation (no OpenMP)
- ✅ Easy-to-use commands
- ✅ 70 total binaries for comprehensive performance testing

