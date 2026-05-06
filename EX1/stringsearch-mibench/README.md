# String Search Benchmark (Parameterized)

Pratt-Boyer-Moore string search benchmark with configurable workload sizes.

## Quick Start

```bash
# Build
make

# Run with different sizes
./EX1_optimize_codes/pbmsrch_gcc mini
./EX1_optimize_codes/pbmsrch_gcc small
./EX1_optimize_codes/pbmsrch_gcc medium
./EX1_optimize_codes/pbmsrch_gcc large
./EX1_optimize_codes/pbmsrch_gcc extra-large

# Test all sizes at once
make test-sizes
```

## Available Sizes

| Size          | Iterations | Total Searches  | Estimated Runtime |
|--------------|-----------|-----------------|-------------------|
| mini         | 1         | 1,333           | ~0.014s           |
| small        | 5,000     | 6,665,000       | ~1.5s             |
| medium       | 20,000    | 26,660,000      | ~6s               |
| large        | 80,000    | 106,640,000     | ~24s              |
| extra-large  | 320,000   | 426,560,000     | ~96s              |

## How It Works

The benchmark performs string search operations using the **Pratt-Boyer-Moore algorithm**. Workload scales by repeating 1,333 individual searches multiple times based on the selected size.

## Customizing Sizes

Edit `EX1_optimize_codes/pbmsrch_parameterized.c` lines 82-88:

```c
static const BenchmarkConfig SIZE_CONFIGS[] = {
    {"mini",        1},      /* 1 iteration */
    {"small",       5000},   /* 5,000 iterations */
    {"medium",      20000},  /* 20,000 iterations */
    {"large",       80000},  /* 80,000 iterations */
    {"extra-large", 320000}, /* 320,000 iterations */
};
```

After editing, recompile:
```bash
make clean && make
```

## Makefile Targets

- `make` - Build parameterized binary
- `make test-sizes` - Test all sizes with timing
- `make clean` - Remove binaries

## Files

- **`EX1_optimize_codes/pbmsrch_parameterized.c`** - Parameterized source code
- **`EX1_optimize_codes/pbmsrch_gcc`** - Compiled binary
- **`Makefile`** - Build configuration

## Algorithm

The Pratt-Boyer-Moore string search algorithm:
- Preprocesses the pattern to build a skip table
- Efficiently searches by skipping characters
- Average-case time complexity: O(n/m) where n=text length, m=pattern length

## Notes

- Runtime may vary based on system performance
- Output can be suppressed with `> /dev/null` for pure timing tests
- All sizes produce identical results per iteration
