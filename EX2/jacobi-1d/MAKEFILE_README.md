# JACOBI-1D Benchmark - Makefile User Guide

## Benchmark Information
- **Name**: JACOBI-1D (1D Jacobi Stencil)
- **Source Files**: 21 (1 baseline + 10 Claude + 10 Llama4)
- **Successfully Compiled**: 20 (1 baseline + 10 Claude + 9 Llama4)
- **Compilation Errors**: 1 (Llama4 v9)
- **Dataset Sizes**: 5 (mini, small, medium, large, extra-large)
- **Total Binaries**: 100 (20 × 5)
- **Compilation**: Serial (no OpenMP)

## Quick Start
```bash
make              # Build all binaries
make test-mini    # Test mini dataset
```

## Dataset Sizes (Array Size × Time Steps)
| Size | TSTEPS | N |
|------|--------|---|
| mini | 20 | 30 |
| small | 40 | 120 |
| medium | 100 | 400 |
| large | 500 | 2000 |
| extra-large | 1000 | 4000 |

## Known Issues
**1 Llama4 variant fails to compile**:
- ✗ jacobi-1d_llama4_v9.c

**20 variants compile successfully** (95%):
- ✓ baseline + All 10 Claude + 9 Llama4 (v1-v8, v10)

## Notes
- JACOBI-1D implements 1D stencil computation
- Part of PolyBench stencils category
- All binaries in `EX1_optimized_codes/`
- **High compilation success rate: 95%** ✓
