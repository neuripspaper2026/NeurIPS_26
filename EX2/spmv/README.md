# SPMV Input Generator

This `generate.py` script produces input data files for the SPMV (Sparse Matrix-Vector Multiplication) benchmark in the expected Matrix Market format.

## File Format

### Matrix Market Format (.mtx)
Text-based sparse matrix format (COO - Coordinate format):
- **Header**: `%%MatrixMarket matrix coordinate real [symmetric/general]`
- **Size line**: `rows cols nonzeros`
- **Data lines**: `row col value` (1-indexed, one entry per line)

### Vector Format (vector.bin)
Binary dense vector file:
- **Format**: Little-endian float32 array
- **Size**: `dim` float values (where `dim` = matrix column count)

## Presets

The generator includes several size presets for sparse matrices:

| Preset | Rows | Cols | Nonzeros (NNZ) | Density | Description |
|--------|------|------|----------------|---------|-------------|
| mini | 128 | 128 | 512 | ~3% | Tiny for testing |
| small | 1138 | 1138 | 2596 | ~0.2% | Reference small (like 1138_bus) |
| medium | 11948 | 11948 | 80519 | ~0.06% | Medium (like bcsstk18) |
| large | 146689 | 146689 | 1009977 | ~0.005% | Large (like Dubcova3) |
| extra-large | 300000 | 300000 | 3000000 | ~0.003% | Extra large scale |

**Formula**: Density = NNZ / (rows × cols)

You can also override individual parameters with explicit arguments.

## Usage

### Synthetic Generation

Generate random sparse matrix data using predefined patterns:

```bash
python generate.py synthetic \
  --out-dir <OUTPUT_DIR> \
  --count <NUM_FILES> \
  --preset <SIZE> \
  --kind <PATTERN> \
  --seed <SEED>
```

#### Parameters:

- `--out-dir PATH`: Output directory (created if needed)
- `--count INT`: Number of datasets to generate (default: 1)
- `--preset {mini,small,medium,large,extra-large}`: Preset size
- `--rows INT`: Override number of rows
- `--cols INT`: Override number of columns
- `--nnz INT`: Override number of nonzero entries
- `--kind {random,diagonal,banded}`: Sparse matrix pattern (default: random)
  - `random`: Random sparse entries
  - `diagonal`: Diagonal matrix with narrow bandwidth
  - `banded`: Banded matrix with wider bandwidth
- `--symmetric`: Generate symmetric matrix (upper triangular stored)
- `--seed INT`: Base random seed (default: 0)
- `--start-index INT`: Starting index for naming (default: 0)
- `--digits INT`: Zero-padding width for indices (default: 4)
- `--overwrite`: Allow overwriting existing files
- `--fixed-name`: Use fixed naming (no index prefix)

#### Examples:

**Generate mini dataset for testing:**
```bash
python generate.py synthetic \
  --out-dir input_data/mini/input \
  --count 1 \
  --preset mini \
  --seed 0 \
  --fixed-name \
  --overwrite
```

**Generate 5 small random sparse matrices:**
```bash
python generate.py synthetic \
  --out-dir input_data/test/input \
  --count 5 \
  --preset small \
  --kind random \
  --seed 42 \
  --start-index 0 \
  --digits 4
```

**Generate medium symmetric banded matrix:**
```bash
python generate.py synthetic \
  --out-dir input_data/medium_sym/input \
  --count 1 \
  --preset medium \
  --kind banded \
  --symmetric \
  --seed 100 \
  --fixed-name
```

**Generate custom-sized sparse matrix:**
```bash
python generate.py synthetic \
  --out-dir input_data/custom/input \
  --count 1 \
  --rows 5000 \
  --cols 5000 \
  --nnz 25000 \
  --kind random \
  --seed 0 \
  --fixed-name
```

### From-Sources Generation

Convert existing Matrix Market files to the benchmark format:

```bash
python generate.py from-sources \
  --src-glob <GLOB_PATTERN> \
  --out-dir <OUTPUT_DIR>
```

#### Parameters:

- `--src-glob PATTERN`: Glob pattern for existing .mtx files (e.g., `"data/*.mtx"`)
- `--out-dir PATH`: Output directory (created if needed)
- `--start-index INT`: Starting index for naming (default: 0)
- `--digits INT`: Zero-padding width for indices (default: 4)
- `--overwrite`: Allow overwriting existing files
- `--fixed-name`: Use fixed naming (no index prefix)

#### Examples:

**Convert all .mtx files from a directory:**
```bash
python generate.py from-sources \
  --src-glob "original_data/*.mtx" \
  --out-dir input_data/converted/input \
  --fixed-name \
  --overwrite
```

**Convert specific matrix files with numbering:**
```bash
python generate.py from-sources \
  --src-glob "matrices/test_*.mtx" \
  --out-dir input_data/test/input \
  --start-index 0 \
  --digits 4
```

## Running the Benchmark

After generating input data, run the SPMV benchmark:

```bash
# Using mini dataset
./EX1_optimized_codes/spmv_gcc \
  -i input_data/mini/input/matrix.mtx,input_data/mini/input/vector.bin \
  -o input_data/mini/output/result.bin

# Using small dataset
./EX1_optimized_codes/spmv_gcc \
  -i input_data/small/input/1138_bus.mtx,input_data/small/input/vector.bin \
  -o input_data/small/output/result.bin
```

## Output Files

For each generated dataset, the script creates:
- `[prefix]matrix.mtx` - Sparse matrix in Matrix Market format
- `[prefix]vector.bin` (or `vector.bin`) - Dense vector in binary format
- `[prefix]matrix.meta.json` - Metadata file with dimensions and generation info

## Notes

- Matrix Market format is 1-indexed (rows and columns start from 1)
- The generator internally uses 0-indexing and converts when writing
- Symmetric matrices only store upper triangular entries
- Vector dimension must match matrix column count
- All floating-point values use little-endian float32 format

