# B+Tree Data Generator

## Overview

`generate.py` creates input data files for the B+Tree benchmark in the correct format.

## Input Format

### mil.txt (Key File)
```
<num_keys>
<key_0>
<key_1>
...
<key_n-1>
```

- First line: total number of keys (integer)
- Following lines: one integer key per line
- Keys are inserted into the B+Tree in order

### command.txt (Command File)
```
j <start> <count>
k <count>
```

- `j <start> <count>`: Range query starting at key `start`, retrieving `count` entries
- `k <count>`: Execute `count` random point queries
  - ⚠️ **CUDA artifact**: Code contains a 65,535 check, but this is **NOT enforced** for CPU-only execution
  - For CPU-only (no GPU, no parallel), larger counts (100K-300K) work fine

## Presets

| Preset | Keys | File Size | Description |
|--------|------|-----------|-------------|
| mini | 1M | ~7 MB | Mini dataset |
| small | 4M | ~30 MB | Small dataset |
| medium | 8M | ~60 MB | Medium dataset |
| large | 20M | ~150 MB | Large dataset |
| extra-large | 24M | ~180 MB | Extra-large dataset |

## Usage

### Generate with preset
```bash
# Generate mini dataset (sequential keys)
python generate.py synthetic --out-dir input_data/mini/input --preset mini

# Generate small dataset
python generate.py synthetic --out-dir input_data/small/input --preset small

# Generate with random keys
python generate.py synthetic --out-dir input_data/medium/input --preset medium --pattern random
```

### Generate custom size
```bash
# Custom number of keys
python generate.py synthetic --out-dir input_data/custom/input --num-keys 5000000

# With specific seed
python generate.py synthetic --out-dir input_data/test/input --num-keys 100000 --seed 123
```

### Key Patterns

- `sequential`: Keys 0, 1, 2, ..., N-1 (default, best for B+Tree insertion order)
- `random`: Randomly shuffled unique keys
- `mixed`: 80% sequential + 20% random with gaps

## Examples

```bash
# Regenerate mini dataset
python generate.py synthetic --out-dir input_data/mini/input --preset mini --overwrite

# Generate all sizes
for size in mini small medium large extra-large; do
    python generate.py synthetic --out-dir input_data/$size/input --preset $size
done

# Run benchmark with generated data
./EX1_optimized_codes/b+tree_gcc file input_data/mini/input/mil.txt \
    command input_data/mini/input/command.txt \
    -o input_data/mini/output/result.txt
```

## Output Structure

```
input_data/
├── mini/
│   ├── input/
│   │   ├── mil.txt      # 1M keys
│   │   └── command.txt  # Test commands
│   └── output/
│       └── result.txt   # Benchmark output
├── small/
│   ├── input/
│   └── output/
...
```

## Notes

- Sequential pattern is recommended for B+Tree as it represents typical insertion order
- Random pattern tests worst-case tree balancing behavior
- Command queries are automatically scaled based on dataset size
- Large datasets (64M+) may take several minutes to generate

