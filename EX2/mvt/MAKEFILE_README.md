# MVT Benchmark - Makefile User Guide

## Benchmark Information
- **Name**: MVT (Matrix-Vector Transpose Operations)
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

## Dataset Sizes (Vector Length)
| Size | N |
|------|---|
| mini | 40 |
| small | 120 |
| medium | 400 |
| large | 2000 |
| extra-large | 4000 |

## Notes
- MVT performs matrix-vector transpose operations (y1 = A·x1 and y2 = A^T·x2)
- Part of PolyBench linear algebra kernels
- All binaries in `EX1_optimized_codes/`
- **100% compilation success rate** ✓
