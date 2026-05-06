# HEAT-3D Benchmark - Makefile User Guide

## Benchmark Information
- **Name**: HEAT-3D (3D Heat Equation)
- **Source Files**: 18 (1 baseline + 10 Claude + 7 Llama4)
- **Successfully Compiled**: 13 (1 baseline + 10 Claude + 2 Llama4)
- **Compilation Errors**: 5 (Llama4 v4, v6, v8, v9, v10)
- **Dataset Sizes**: 5 (mini, small, medium, large, extra-large)
- **Total Binaries**: 65 (13 × 5)
- **Compilation**: Serial (no OpenMP)

## Quick Start
```bash
make              # Build all binaries
make test-mini    # Test mini dataset
```

## Dataset Sizes (3D Grid × Time Steps)
| Size | TSTEPS | N | Grid Size |
|------|--------|---|-----------|
| mini | 20 | 10 | 10³ |
| small | 40 | 20 | 20³ |
| medium | 100 | 40 | 40³ |
| large | 500 | 120 | 120³ |
| extra-large | 1000 | 200 | 200³ |

## Known Issues
**5 Llama4 variants fail to compile**:
- ✗ heat-3d_llama4_v4.c
- ✗ heat-3d_llama4_v6.c
- ✗ heat-3d_llama4_v8.c
- ✗ heat-3d_llama4_v9.c
- ✗ heat-3d_llama4_v10.c

**13 variants compile successfully** (72%):
- ✓ baseline + All 10 Claude + 2 Llama4 (v1, v2, v3, v5, v7)

## Notes
- HEAT-3D simulates 3D heat diffusion
- Part of PolyBench stencils category
- All binaries in `EX1_optimized_codes/`
