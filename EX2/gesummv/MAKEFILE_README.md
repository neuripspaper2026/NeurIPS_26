# GESUMMV Benchmark - Makefile User Guide

## Benchmark Information
- **Name**: GESUMMV (GEneral Scalar, Matrix, Vector multiplication)
- **Source Files**: 16 (1 baseline + 10 Claude + 5 Llama4)
- **Successfully Compiled**: 16 (100%) ✓
- **Dataset Sizes**: 5 (mini, small, medium, large, extra-large)
- **Total Binaries**: 80 (16 × 5)
- **Compilation**: Serial (no OpenMP)

## Quick Start
```bash
make              # Build all binaries
make test-mini    # Test mini dataset
```

## Dataset Sizes (Matrix/Vector Dimensions)
| Size | N |
|------|---|
| mini | 30 |
| small | 90 |
| medium | 250 |
| large | 1300 |
| extra-large | 2800 |

## Notes
- GESUMMV performs: y = αAx + βBy
- Part of PolyBench linear algebra kernels
- All binaries in `EX1_optimized_codes/`
- **100% compilation success rate** ✓
