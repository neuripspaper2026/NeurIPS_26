# SEIDEL-2D Benchmark - Makefile User Guide

## Benchmark Information
- **Name**: SEIDEL-2D (Seidel 2D Iterative Stencil)
- **Source Files**: 15 (1 baseline + 10 Claude + 4 Llama4)
- **Successfully Compiled**: 15 (100%) ✓
- **Dataset Sizes**: 5 (mini, small, medium, large, extra-large)
- **Total Binaries**: 75 (15 × 5)
- **Compilation**: Serial (no OpenMP)

## Quick Start
```bash
make              # Build all binaries
make test-mini    # Test mini dataset
```

## Dataset Sizes (Grid Size × Time Steps)
| Size | TSTEPS | N |
|------|--------|---|
| mini | 20 | 40 |
| small | 40 | 120 |
| medium | 100 | 400 |
| large | 500 | 2000 |
| extra-large | 1000 | 4000 |

## Notes
- SEIDEL-2D performs iterative stencil computation
- Part of PolyBench stencils category
- All binaries in `EX1_optimized_codes/`
- **100% compilation success rate** ✓
- Only 4 Llama4 variants (fewer than typical 10)
