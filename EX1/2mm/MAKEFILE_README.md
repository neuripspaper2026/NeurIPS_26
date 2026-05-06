# 2mm Benchmark Makefile User Guide

## Overview

This Makefile provides a standardized build system for the 2mm (two matrix multiplications) benchmark, supporting:
- Automatic compiler detection (gcc/clang)
- Error-tolerant compilation (failures don't interrupt the process)
- Multiple dataset sizes (mini, small, medium, large, extra-large)
- Serial compilation (no OpenMP)

## Compiler Auto-Detection

The Makefile automatically detects available compilers on your system:
- Priority: `gcc` first
- Fallback: `clang` if gcc is unavailable
- Generated binaries are automatically suffixed with the compiler name (e.g., `2mm_gcc_mini` or `2mm_clang_mini`)

## Dataset Sizes

5 different dataset sizes are supported:

| Size | Macro Definition | Use Case |
|------|------------------|----------|
| mini | `MINI_DATASET` | Quick testing |
| small | `SMALL_DATASET` | Small-scale testing |
| medium | `MEDIUM_DATASET` | Medium-scale testing |
| large | `LARGE_DATASET` | Large-scale testing |
| extra-large | `EXTRA_LARGE_DATASET` | Extra large-scale testing |

## Main Targets

### Basic Compilation

```bash
make              # Build all files (baseline + all variants, 5 sizes each)
                  # - 21 source files × 5 sizes = 105 binaries

make all          # Same as above

make baseline     # Build only baseline (2mm.c) for all 5 sizes
                  # - Generates 2mm_gcc_mini, _small, _medium, _large, _extra-large

make variants     # Build baseline + all variants (all sizes)
```

### Testing

```bash
make test-mini           # Run baseline mini test
make test-small          # Run baseline small test
make test-medium         # Run baseline medium test
make test-large          # Run baseline large test
make test-extra-large    # Run baseline extra-large test
make test-all            # Run all size tests
```

### Cleanup

```bash
make clean        # Remove all generated binaries
```

### Help

```bash
make help         # Display detailed help information
```

## Binary Naming Convention

Generated binaries follow this naming convention:

```
<source_base>_<compiler>_<size>
```

Examples:
- `2mm_gcc_mini` - baseline mini size, compiled with gcc
- `2mm_gcc_large` - baseline large size, compiled with gcc
- `2mm_claude_v1_gcc_mini` - claude_v1 variant mini size, compiled with gcc
- `2mm_claude_v1_gcc_large` - claude_v1 variant large size, compiled with gcc
- `2mm_llama4_v5_clang_medium` - llama4_v5 variant medium size, compiled with clang

## Feature Description

### 1. Error Handling ✓

When a source file fails to compile, the Makefile will:
- Report the error information
- Continue compiling other files
- Display success/failure statistics at the end

Example output:
```
[5/20] Building 2mm_variant_x...
  ✗ Failed to build 2mm_variant_x (continuing...)
[6/20] Building 2mm_variant_y...
  Compiling 2mm_variant_y (mini)...
  ...
⚠ 1 variant(s) failed to compile (continued with others)
  Successfully built: 19/20 variants
```

### 2. Dynamic Source File Detection

The Makefile automatically scans all `.c` files in the `EX1_optimized_codes/` directory, eliminating the need to manually maintain source file lists.

### 3. No Intermediate Files

The compilation process directly generates executable files without creating `.o` intermediate files.

## Directory Structure

```
2mm/
├── Makefile                    # Main Makefile
├── 2mm.c                       # Baseline source (reference only)
├── 2mm.h                       # Header file
├── EX1_optimized_codes/        # Source code and generated binaries
│   ├── 2mm.c                   # Baseline source
│   ├── 2mm_claude_v*.c         # Claude variants
│   ├── 2mm_llama4_v*.c         # Llama4 variants
│   ├── 2mm_gcc_mini            # Generated binary files
│   ├── 2mm_gcc_small
│   └── ...
└── EX5_correctness/            # Correctness verification output
```

## Dependencies

- Compiler: gcc or clang
- Math library: libm (`-lm`)
- Polybench toolkit: `../../utilities/polybench.c`

## Usage Examples

### Example 1: Compile All Files (Recommended)

```bash
make              # Compile baseline + all variants
```

### Example 2: Compile and Run Baseline Mini

```bash
make baseline     # Compile baseline only
./EX1_optimized_codes/2mm_gcc_mini
```

### Example 3: Clean and Recompile All Files

```bash
make clean
make              # Compile all files
```

### Example 4: Run All Tests

```bash
make test-all
```

### Example 5: Compile Baseline Only

```bash
make baseline     # Only compile 2mm.c for all sizes
```

### Example 6: Compile Specific Variant and Size

```bash
make EX1_optimized_codes/2mm_claude_v1_gcc_large
```

## Performance Measurement

For performance measurement, you can use the existing scripts:

```bash
./measure_gcc_performance_times.sh
./measure_clang_performance_times.sh
```

These scripts will run all variant and size combinations and record execution times.

## Troubleshooting

### Issue: Compiler Not Found

**Error Message**: `No compiler found. Please install gcc or clang`

**Solution**: Install gcc or clang

```bash
# Ubuntu/Debian
sudo apt-get install gcc

# CentOS/RHEL
sudo yum install gcc

# macOS
xcode-select --install
```

### Issue: Cannot Find polybench.c

**Error Message**: `No such file or directory: ../../utilities/polybench.c`

**Solution**: Ensure you're in the correct directory structure, with the utilities directory two levels up.

### Issue: Compilation Failure

If a variant fails to compile, check the source file for syntax errors. The Makefile will continue compiling other files without interrupting the entire compilation process.

## Technical Implementation Details

### Makefile Core Features

1. **Dynamic Compiler Detection**:
   ```makefile
   CC := $(shell command -v gcc 2>/dev/null)
   ifdef CC
       COMPILER_TAG := gcc
   else
       CC := $(shell command -v clang 2>/dev/null)
       COMPILER_TAG := clang
   endif
   ```

2. **Automatic Source File Scanning**:
   ```makefile
   ALL_SOURCES := $(wildcard $(SRC_DIR)/*.c)
   SOURCE_BASES := $(basename $(notdir $(ALL_SOURCES)))
   ```

3. **Error Handling Loop**:
   ```makefile
   for base in $(SOURCE_BASES); do
       if ! $(MAKE) ... 2>&1; then
           failed=$$((failed+1))
           echo "  ✗ Failed (continuing...)"
       fi
   done
   ```

4. **Pattern Rules**:
   ```makefile
   $(SRC_DIR)/%_$(COMPILER_TAG)_mini: $(SRC_DIR)/%.c
       $(CC) -DMINI_DATASET $(CFLAGS) ... -o $@
   ```

## Contributor Notes

When adding new variants:
1. Place the new `.c` file in the `EX1_optimized_codes/` directory
2. No need to modify the Makefile
3. Run `make variants` to automatically compile the new variant

## Copyright and License

This Makefile follows the PolyBench/C license agreement.
