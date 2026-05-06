# Durbin Benchmark - Makefile User Guide

## Benchmark Information
- **Name**: Durbin (Levinson-Durbin Recursion)
- **Source Files**: 21 (1 baseline + 10 Claude + 10 Llama4)
- **Dataset Sizes**: 5 (mini, small, medium, large, extra-large)
- **Total Binaries**: 105 (21 × 5)
- **Compilation**: Serial (no OpenMP)

## Quick Start
```bash
make              # Build all binaries
make baseline     # Build only baseline
make test-mini    # Test mini dataset
make clean        # Remove binaries
```

## Dataset Sizes
| Size | N | Array Size |
|------|---|------------|
| mini | 40 | ~1KB |
| small | 120 | ~11KB |
| medium | 400 | ~128KB |
| large | 2000 | ~3.2MB |
| extra-large | 4000 | ~12.8MB |

## Binary Naming
`<source>_<compiler>_<size>`

Examples:
- `durbin_gcc_mini`
- `durbin_claude_v3_gcc_large`

## Technical Details
- **Compiler**: Auto-detected (gcc/clang)
- **Flags**: `-O3 -lm`
- **Serial**: No `-fopenmp`
- **Direct compilation**: No `.o` files

## Notes
- All 21 source files compile successfully
- Binaries generated in `EX1_optimized_codes/`
- Durbin solves Toeplitz systems using recursion
- Part of PolyBench linear algebra kernels
