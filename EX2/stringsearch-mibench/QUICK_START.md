# String Search Benchmark - Quick Start

## Build

```bash
make
```

## Run with Different Sizes

```bash
./EX1_optimize_codes/pbmsrch_gcc mini
./EX1_optimize_codes/pbmsrch_gcc small
./EX1_optimize_codes/pbmsrch_gcc medium
./EX1_optimize_codes/pbmsrch_gcc large
./EX1_optimize_codes/pbmsrch_gcc extra-large
```

## Test All Sizes

```bash
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

## Adjust Sizes

Edit `EX1_optimize_codes/pbmsrch_parameterized.c` lines 82-88, then `make clean && make`

## More Info

See `README.md` for complete documentation

