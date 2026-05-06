# GEMM Benchmark - Makefile User Guide

## Benchmark Information
- **Name**: GEMM (General Matrix Multiply)
- **Source Files**: 21 (1 baseline + 10 Claude + 10 Llama4)
- **Dataset Sizes**: 5 (mini, small, medium, large, extra-large)
- **Total Binaries**: 105 (21 × 5)
- **Compilation**: Serial (no OpenMP)

## Quick Start
```bash
make              # Build all binaries
make test-mini    # Test mini dataset
```

## Dataset Sizes (Matrix Dimensions)
| Size | NI | NJ | NK | Operation |
|------|----|----|----|-----------| 
| mini | 20 | 25 | 30 | 20×25 × 25×30 |
| small | 60 | 70 | 80 | 60×70 × 70×80 |
| medium | 200 | 220 | 240 | 200×220 × 220×240 |
| large | 1000 | 1100 | 1200 | 1000×1100 × 1100×1200 |
| extra-large | 2000 | 2300 | 2600 | 2000×2300 × 2300×2600 |

## Notes
- GEMM performs C = α·A×B + β·C
- Core BLAS Level-3 operation
- All binaries in `EX1_optimized_codes/`
- Part of PolyBench linear algebra kernels
