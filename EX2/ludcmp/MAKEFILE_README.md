# LUDCMP Benchmark - Makefile User Guide

## Benchmark Information
- **Name**: LUDCMP (LU Decomposition with Linear Solver)
- **Source Files**: 12 (1 baseline + 10 Claude + 1 Llama4)
- **Successfully Compiled**: 12 (100%) ✓
- **Dataset Sizes**: 5 (mini, small, medium, large, extra-large)
- **Total Binaries**: 60 (12 × 5)
- **Compilation**: Serial (no OpenMP)

## Quick Start
```bash
make              # Build all binaries
make test-mini    # Test mini dataset
```

## Dataset Sizes (Matrix Dimensions)
| Size | N |
|------|---|
| mini | 40 |
| small | 120 |
| medium | 400 |
| large | 2000 |
| extra-large | 4000 |

## Notes
- LUDCMP performs LU decomposition and solves Ax=b
- Part of PolyBench linear algebra solvers
- All binaries in `EX1_optimized_codes/`
- **100% compilation success rate** ✓
- Only 1 Llama4 variant (unusual compared to other benchmarks)
