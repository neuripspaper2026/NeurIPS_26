# JACOBI-2D Benchmark - Makefile User Guide

## Benchmark Information
- **Name**: JACOBI-2D (2D Jacobi Stencil)
- **Source Files**: 11 (1 baseline + 10 Claude)
- **Successfully Compiled**: 11 (100%) ✓
- **Dataset Sizes**: 5 (mini, small, medium, large, extra-large)
- **Total Binaries**: 55 (11 × 5)
- **Compilation**: Serial (no OpenMP)

## Quick Start
```bash
make              # Build all binaries
make test-mini    # Test mini dataset
```

## Dataset Sizes (Grid Size × Time Steps)
| Size | TSTEPS | N | Grid Size |
|------|--------|---|-----------|
| mini | 20 | 40 | 40² |
| small | 40 | 120 | 120² |
| medium | 100 | 400 | 400² |
| large | 500 | 2000 | 2000² |
| extra-large | 1000 | 4000 | 4000² |

## Notes
- JACOBI-2D implements 2D stencil computation
- Part of PolyBench stencils category
- All binaries in `EX1_optimized_codes/`
- **100% compilation success rate** ✓
