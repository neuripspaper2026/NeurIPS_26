# FDTD-2D Benchmark - Makefile User Guide

## Benchmark Information
- **Name**: FDTD-2D (Finite-Difference Time-Domain 2D)
- **Source Files**: 19 (1 baseline + 10 Claude + 8 Llama4)
- **Successfully Compiled**: 15 (1 baseline + 10 Claude + 4 Llama4)
- **Compilation Errors**: 4 (Llama4 v1, v2, v4, v5)
- **Dataset Sizes**: 5 (mini, small, medium, large, extra-large)
- **Total Binaries**: 75 (15 × 5)
- **Compilation**: Serial (no OpenMP)

## Quick Start
```bash
make              # Build all binaries
make test-mini    # Test mini dataset
```

## Dataset Sizes (Time Steps × Grid)
| Size | TMAX | NX | NY | Grid Size |
|------|------|----|----|-----------|
| mini | 20 | 20 | 30 | 20×30 |
| small | 40 | 60 | 80 | 60×80 |
| medium | 100 | 200 | 240 | 200×240 |
| large | 500 | 1000 | 1200 | 1000×1200 |
| extra-large | 1000 | 2000 | 2600 | 2000×2600 |

## Known Issues
**4 variants fail to compile**:
- ✗ fdtd-2d_llama4_v1.c
- ✗ fdtd-2d_llama4_v2.c
- ✗ fdtd-2d_llama4_v4.c
- ✗ fdtd-2d_llama4_v5.c

**15 variants compile successfully** (79%):
- ✓ baseline + All Claude + 4 Llama4 (v3, v6, v7, v8)

## Notes
- FDTD-2D simulates electromagnetic wave propagation
- Part of PolyBench stencils category
- All binaries in `EX1_optimized_codes/`
