# TRISOLV Benchmark - Makefile User Guide

## Benchmark Information
- **Name**: TRISOLV (Triangular Linear System Solver)
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

## Dataset Sizes (Problem Size)
| Size | N |
|------|---|
| mini | 40 |
| small | 120 |
| medium | 400 |
| large | 2000 |
| extra-large | 4000 |

## Notes
- TRISOLV solves triangular linear system: Lx = b
- Part of PolyBench linear algebra solvers
- All binaries in `EX1_optimized_codes/`
- **100% compilation success rate** ✓
