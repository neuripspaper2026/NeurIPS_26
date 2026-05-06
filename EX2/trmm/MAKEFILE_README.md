# TRMM Benchmark - Makefile User Guide

## Benchmark Information
- **Name**: TRMM (Triangular Matrix Multiply)
- **Source Files**: 13 (1 baseline + 10 Claude + 2 Llama4)
- **Successfully Compiled**: 13 (100%) ✓
- **Dataset Sizes**: 5 (mini, small, medium, large, extra-large)
- **Total Binaries**: 65 (13 × 5)
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
- TRMM performs triangular matrix multiplication: B = alpha*A*B
- Part of PolyBench linear algebra BLAS operations
- All binaries in `EX1_optimized_codes/`
- **100% compilation success rate** ✓
- Only 2 Llama4 variants (fewer than typical 10)
