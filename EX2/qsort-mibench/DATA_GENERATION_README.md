# qsort-mibench Data Generator

This document describes how to generate input data for the qsort-mibench benchmark.

## Overview

The qsort benchmark sorts 3D vertex points by their distance from the origin. Each vertex has three coordinates (x, y, z) as 32-bit signed integers.

## Input Format

- **File format**: Plain text
- **Structure**: One vertex per line
- **Fields**: Three tab-separated integers (x, y, z)
- **Example**:
  ```
  1681692777	846930886	1804289383
  424238335	1957747793	1714636915
  596516649	1649760492	719885386
  ```

## Size Presets

| Preset | Vertices | File Size | Runtime (approx) | Speedup vs mini |
|--------|----------|-----------|------------------|-----------------|
| `mini` | 60,000 | ~2 MB | ~0.06s | 1.0x |
| `small` | 300,000 | ~10 MB | ~0.25s | 4.4x |
| `medium` | 2,000,000 | ~63 MB | ~1.7s | 30.4x |
| `large` | 10,000,000 | ~315 MB | ~9.0s | 158.5x |
| `extra-large` | 30,000,000 | ~943 MB | ~28.0s | 493.8x |

⚠️ **Configuration**: The code uses `MAXARRAY=31000000` with malloc() to handle all sizes.

## Usage

### Generate All Standard Sizes

```bash
# Generate mini (reference size)
python generate.py synthetic --preset mini --out-dir input_data/mini/input

# Generate small
python generate.py synthetic --preset small --out-dir input_data/small/input

# Generate medium
python generate.py synthetic --preset medium --out-dir input_data/medium/input

# Generate large
python generate.py synthetic --preset large --out-dir input_data/large/input

# Generate extra-large
python generate.py synthetic --preset extra-large --out-dir input_data/extra-large/input
```

### Generate Custom Size

```bash
# Generate 100K vertices with custom seed
python generate.py synthetic --count 100000 --seed 42 --out-dir data/custom
```

### Command-Line Options

**Synthetic generation:**
```
python generate.py synthetic [OPTIONS]

Required:
  --out-dir PATH        Output directory (created if needed)

Optional:
  --preset NAME         Use size preset (mini|small|medium|large|extra-large)
  --count INT           Number of vertices (required if no preset)
  --seed INT            Random seed (default: 12345)
  --name FILENAME       Custom output filename
  --prefix STR          Filename prefix (default: 'input')
  --suffix STR          Filename suffix (default: '.dat')
  --overwrite           Overwrite existing files
  --write-meta          Write metadata file (.meta)
```

## Running the Benchmark

### Compile the benchmark

```bash
make
```

### Run with default MAXARRAY (mini and small only)

```bash
# Run mini size (50K vertices)
./EX1_optimized_codes/qsort_large_gcc input_data/mini/input/input_mini.dat > output_mini.txt

# Run small size (60K vertices, max for default)
./EX1_optimized_codes/qsort_large_gcc input_data/small/input/input_small.dat > output_small.txt
```

### For larger sizes: Modify MAXARRAY

To use medium, large, or extra-large sizes:

1. **Edit source code** (`qsort_large.c` or variant in `EX1_optimized_codes/`):
   ```c
   #define MAXARRAY 600000  /* Change from 60000 to 600000 */
   ```

2. **Recompile**:
   ```bash
   make clean && make
   ```

3. **Run**:
   ```bash
   # Now you can run larger sizes
   ./EX1_optimized_codes/qsort_large_gcc input_data/medium/input/input_medium.dat > output_medium.txt
   ./EX1_optimized_codes/qsort_large_gcc input_data/large/input/input_large.dat > output_large.txt
   ./EX1_optimized_codes/qsort_large_gcc input_data/extra-large/input/input_extra-large.dat > output_xlarge.txt
   ```

### Automated testing

Use the provided test script:

```bash
./test_all_sizes.sh
```

This will test all available sizes and show which ones work with your current MAXARRAY setting.

## Data Characteristics

- **Coordinate range**: Full 32-bit signed integer range [-2³¹, 2³¹-1]
- **Distribution**: Uniform random distribution
- **Reproducibility**: Fixed seed ensures identical output across runs
- **Format**: ASCII text (human-readable, easy to verify)

## Validation

To verify generated data:

```bash
# Check line count
wc -l input_data/mini/input/input_mini.dat
# Should output: 50000

# Check format (each line has 3 integers)
head -5 input_data/mini/input/input_mini.dat

# Run benchmark and check output
./EX1_optimized_codes/qsort_large_gcc input_data/mini/input/input_mini.dat | head -20
```

## Notes

- The generator uses a fixed default seed (12345) for reproducibility
- Output files are tab-separated for compatibility with the original benchmark
- Coordinate values use full 32-bit integer range to stress-test sorting algorithms
- File size scales linearly with vertex count (~30 bytes per vertex)

## Troubleshooting

**Problem**: "File already exists" error  
**Solution**: Use `--overwrite` flag or delete existing file

**Problem**: Out of memory during generation  
**Solution**: Generate smaller batches or use a machine with more RAM

**Problem**: Benchmark shows 0 vertices sorted  
**Solution**: Check input file format (must be tab-separated, not space-separated)

