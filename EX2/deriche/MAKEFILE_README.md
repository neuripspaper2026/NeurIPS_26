# Deriche Benchmark - Makefile User Guide

## Overview

This directory contains the standardized Makefile for the `deriche` benchmark from the PolyBench suite. The Makefile automates the compilation of multiple source file variants across 5 different dataset sizes.

## Benchmark Information

- **Benchmark Name**: Deriche (Edge Detection Filter)
- **Source Files**: 21 total
  - 1 baseline: `deriche.c`
  - 10 Claude variants: `deriche_claude_v1.c` to `deriche_claude_v10.c`
  - 10 Llama4 variants: `deriche_llama4_v1.c` to `deriche_llama4_v10.c`
- **Successfully Compiled**: 20 sources (1 baseline + 10 Claude + 9 Llama4)
- **Compilation Errors**: 1 source (Llama4 v9)
- **Dataset Sizes**: 5 (mini, small, medium, large, extra-large)
- **Total Binaries**: 100 (20 successful sources × 5 sizes)
- **Compilation Mode**: Serial (no OpenMP)

## Quick Start

### Build All Binaries

```bash
make
```

This will compile all source files for all 5 dataset sizes. Successfully compiles 20 out of 21 sources, generating 100 binaries (1 source fails to compile and is automatically skipped).

### Build Only Baseline

```bash
make baseline
```

This will compile only the baseline `deriche.c` for all 5 dataset sizes.

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
- `deriche_gcc_mini` - baseline with mini dataset
- `deriche_gcc_extra-large` - baseline with extra-large dataset
- `deriche_claude_v3_gcc_large` - Claude variant 3 with large dataset
- `deriche_llama4_v7_gcc_small` - Llama4 variant 7 with small dataset

## Dataset Sizes

| Size | Macro | W | H | Image Size | Memory |
|------|-------|---|---|------------|--------|
| mini | `MINI_DATASET` | 64 | 64 | 64×64 | ~16KB |
| small | `SMALL_DATASET` | 192 | 128 | 192×128 | ~96KB |
| medium | `MEDIUM_DATASET` | 720 | 480 | 720×480 | ~1.3MB |
| large | `LARGE_DATASET` | 4096 | 2160 | 4096×2160 (4K) | ~34MB |
| extra-large | `EXTRALARGE_DATASET` | 7680 | 4320 | 7680×4320 (8K) | ~127MB |

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
./EX1_optimized_codes/deriche_gcc_mini
```

### Redirect Output to File

```bash
./EX1_optimized_codes/deriche_gcc_large > output.txt
```

### Measure Execution Time

```bash
time ./EX1_optimized_codes/deriche_gcc_large
```

## Project Structure

```
deriche/
├── Makefile                          # This standardized Makefile
├── MAKEFILE_README.md                # This file
├── deriche.h                         # Header with dataset size definitions
├── deriche.c                         # Baseline source code
├── EX1_optimized_codes/              # Source files directory
│   ├── deriche.c                     # Baseline (symlink or copy)
│   ├── deriche_claude_v*.c           # Claude-optimized variants
│   ├── deriche_llama4_v*.c           # Llama4-optimized variants
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

The Makefile uses the `EXTRALARGE_DATASET` macro (no underscore between EXTRA and LARGE) as defined in `deriche.h`, but the binary suffix uses `extra-large` (with hyphen) for consistency across all benchmarks.

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

**1 variant fails to compile** (source file exists but binary cannot be generated):

- ✗ `deriche_llama4_v9.c`

**20 variants compile successfully**:
- ✓ baseline: `deriche.c`
- ✓ All 10 Claude variants (v1-v10)
- ✓ 9 Llama4 variants: v1-v8, v10

The Makefile's error handling mechanism automatically skips this problematic file and continues building the remaining variants.

## Notes

- **20 out of 21** source files compile successfully, generating **100 binaries** (20 × 5 sizes)
- 1 Llama4 variant contains compilation errors and is automatically skipped
- Binary files are generated in the `EX1_optimized_codes/` directory
- The Makefile follows the standardized template used across all PolyBench benchmarks in this project
- For correctness verification, compare outputs using the `EX5_correctness/` directory
- Deriche is an edge detection filter benchmark, commonly used in image processing

## Summary

This standardized Makefile provides:
- ✅ Automated compilation for 21 source files (20 compile successfully)
- ✅ Support for 5 dataset sizes (from 64×64 to 8K resolution)
- ✅ Robust error handling (automatically skips 1 failing variant)
- ✅ Consistent binary naming
- ✅ Serial compilation (no OpenMP)
- ✅ Easy-to-use commands
- ✅ 100 total binaries for comprehensive performance testing

