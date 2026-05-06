# SYR2K Benchmark - Makefile User Guide

## Benchmark Information
- **Name**: SYR2K (Symmetric Rank-2K Update)
- **Source Files**: 20 (1 baseline + 10 Claude + 9 Llama4)
- **Successfully Compiled**: 20 (100%) ✓
- **Dataset Sizes**: 5 (mini, small, medium, large, extra-large)
- **Total Binaries**: 100 (20 × 5)
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
- SYR2K performs symmetric rank-2k update: C = alpha*A*B^T + alpha*B*A^T + beta*C
- Part of PolyBench linear algebra BLAS operations
- All binaries in `EX1_optimized_codes/`
- **100% compilation success rate** ✓
- 9 Llama4 variants (close to typical 10)
