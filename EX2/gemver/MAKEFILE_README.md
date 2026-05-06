# GEMVER Benchmark - Makefile User Guide

## Benchmark Information
- **Name**: GEMVER (GEneral Matrix Vector multiplication)
- **Source Files**: 20 (1 baseline + 10 Claude + 9 Llama4)
- **Successfully Compiled**: 16 (1 baseline + 10 Claude + 5 Llama4)
- **Compilation Errors**: 4 (Llama4 v1, v2, v7, v10)
- **Dataset Sizes**: 5 (mini, small, medium, large, extra-large)
- **Total Binaries**: 80 (16 × 5)
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

## Known Issues
**4 Llama4 variants fail to compile**:
- ✗ gemver_llama4_v1.c
- ✗ gemver_llama4_v2.c
- ✗ gemver_llama4_v7.c
- ✗ gemver_llama4_v10.c

**16 variants compile successfully** (80%):
- ✓ baseline + All 10 Claude + 5 Llama4 (v3, v4, v5, v6, v8, v9)

## Notes
- GEMVER performs: A = A + u₁v₁ᵀ + u₂v₂ᵀ, w = βAx
- Part of PolyBench linear algebra kernels
- All binaries in `EX1_optimized_codes/`
