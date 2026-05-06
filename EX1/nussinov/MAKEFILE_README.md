# NUSSINOV Benchmark - Makefile User Guide

## Benchmark Information
- **Name**: NUSSINOV (RNA Secondary Structure Prediction)
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

## Dataset Sizes (Sequence Length)
| Size | N |
|------|---|
| mini | 180 |
| small | 500 |
| medium | 1800 |
| large | 2500 |
| extra-large | 5000 |

## Notes
- NUSSINOV algorithm predicts RNA secondary structure using dynamic programming
- Part of PolyBench bioinformatics kernels
- All binaries in `EX1_optimized_codes/`
- **100% compilation success rate** ✓
