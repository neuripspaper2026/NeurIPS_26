# GRAMSCHMIDT Benchmark - Makefile User Guide

## Benchmark Information
- **Name**: GRAMSCHMIDT (Gram-Schmidt Orthogonalization)
- **Source Files**: 21 (1 baseline + 10 Claude + 10 Llama4)
- **Successfully Compiled**: 21 (100%) ✓
- **Dataset Sizes**: 5 (mini, small, medium, large, extra-large)
- **Total Binaries**: 105 (21 × 5)
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
- GRAMSCHMIDT orthogonalizes matrix columns
- Part of PolyBench linear algebra solvers
- All binaries in `EX1_optimized_codes/`
- **100% compilation success rate** ✓
