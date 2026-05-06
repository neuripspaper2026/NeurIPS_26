# SYRK Benchmark - Makefile User Guide

## Benchmark Information
- **Name**: SYRK (Symmetric Rank-K Update)
- **Source Files**: 14 (1 baseline + 10 Claude + 3 Llama4)
- **Successfully Compiled**: 14 (100%) ✓
- **Dataset Sizes**: 5 (mini, small, medium, large, extra-large)
- **Total Binaries**: 70 (14 × 5)
- **Compilation**: Serial (no OpenMP)

## Quick Start
```bash
make              # Build all binaries
make test-mini    # Test mini dataset
```

## Dataset Sizes (Matrix Dimensions)
| Size | M | N |
|------|---|---|
| mini | 20 | 30 |
| small | 60 | 80 |
| medium | 200 | 240 |
| large | 1000 | 1200 |
| extra-large | 2000 | 2600 |

## Notes
- SYRK performs symmetric rank-k update: C = alpha*A*A^T + beta*C
- Part of PolyBench linear algebra BLAS operations
- All binaries in `EX1_optimized_codes/`
- **100% compilation success rate** ✓
- Only 3 Llama4 variants (fewer than typical 10)
