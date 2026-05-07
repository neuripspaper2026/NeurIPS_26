# Stencil Input Generator

This generator creates 3D grid input data for the **stencil** benchmark, which performs iterative 7-point stencil computations on 3D grids.

## Input Data Format

The stencil benchmark expects binary input files containing 3D grids of `float32` values:

```
Format: Raw binary file
- Data type: float32 (4 bytes per value, little-endian)
- Layout: z-major, then y, then x (for(z) for(y) for(x))
- Size: nx × ny × nz float32 values
- No header, just raw float32 values
```

**File naming convention**: `{nx}x{ny}x{nz}.bin`

Example: `512x512x64.bin` contains a 512×512×64 grid (67,108,864 bytes)

## Usage

### Basic Syntax

```bash
python generate.py <subcommand> [options]
```

**Subcommands**:
- `synthetic`: Generate synthetic data with various patterns
- `from-sources`: Convert external binary files to stencil format

### Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `--out-dir PATH` | Output directory (created if needed) | Required |
| `--preset NAME` | Use predefined grid size | None |
| `--nx INT` | Grid size in X dimension | From preset |
| `--ny INT` | Grid size in Y dimension | From preset |
| `--nz INT` | Grid size in Z dimension | From preset |
| `--overwrite` | Overwrite existing files | False |

## Available Presets

| Preset | nx | ny | nz | Grid Size | File Size |
|--------|----|----|----|-----------|-----------| 
| `mini` | 64 | 64 | 16 | 65,536 | 256 KB |
| `small` | 128 | 128 | 32 | 524,288 | 2 MB |
| `medium` | 256 | 256 | 48 | 3,145,728 | 12 MB |
| `default` | 512 | 512 | 64 | 16,777,216 | 64 MB |
| `large` | 768 | 768 | 96 | 56,623,104 | 216 MB |
| `extra-large` | 1024 | 1024 | 128 | 134,217,728 | 512 MB |

## Synthetic Data Generation

### Data Patterns

#### 1. **Random** (default)
Random float values uniformly distributed in a range.

```bash
python generate.py synthetic \
  --out-dir input_data/default/input \
  --preset default \
  --kind random \
  --min-val 0.0 \
  --max-val 1.0 \
  --count 1 \
  --fixed-name
```

**Options**:
- `--min-val FLOAT`: Minimum value (default: 0.0)
- `--max-val FLOAT`: Maximum value (default: 1.0)
- `--seed INT`: Random seed for reproducibility

#### 2. **Uniform**
Constant value throughout the grid.

```bash
python generate.py synthetic \
  --out-dir input_data/test/input \
  --preset small \
  --kind uniform \
  --value 1.0 \
  --count 1 \
  --fixed-name
```

**Options**:
- `--value FLOAT`: The constant value (default: 1.0)

#### 3. **Gaussian**
Gaussian distribution centered in the grid.

```bash
python generate.py synthetic \
  --out-dir input_data/gaussian/input \
  --preset medium \
  --kind gaussian \
  --count 1 \
  --fixed-name
```

Creates a smooth bell-shaped distribution centered at the grid center.

#### 4. **Checkerboard**
Alternating block pattern for testing boundary conditions.

```bash
python generate.py synthetic \
  --out-dir input_data/checkerboard/input \
  --preset small \
  --kind checkerboard \
  --count 1 \
  --fixed-name
```

### File Naming Options

| Option | Description | Default |
|--------|-------------|---------|
| `--fixed-name` | Use `{nx}x{ny}x{nz}.bin` naming | False |
| `--start-index INT` | Starting index for sequential naming | 0 |
| `--digits INT` | Zero-padding width for indices | 4 |
| `--count INT` | Number of files to generate | 1 |

**Without `--fixed-name`**: Files are named `input_0000_{nx}x{ny}x{nz}.bin`, `input_0001_{nx}x{ny}x{nz}.bin`, etc.

**With `--fixed-name`**: File is named `{nx}x{ny}x{nz}.bin` (only use with `--count 1`)

## Converting External Data

Convert raw binary files to stencil format:

```bash
# Convert files matching a glob pattern
python generate.py from-sources \
  --src-glob "raw_data/*.bin" \
  --out-dir input_data/converted/input \
  --preset default \
  --overwrite

# Convert specific files
python generate.py from-sources \
  --src-list file1.bin file2.bin file3.bin \
  --out-dir input_data/custom/input \
  --nx 256 --ny 256 --nz 64
```

**Notes**:
- Source files are expected to be raw binary `float32` data (little-endian)
- Files smaller than expected size are zero-padded
- Files larger than expected size are truncated

## Complete Examples

### Example 1: Generate Default Dataset
```bash
# Create default-sized grid with random values
python generate.py synthetic \
  --out-dir input_data/default/input \
  --preset default \
  --kind random \
  --seed 42 \
  --count 1 \
  --fixed-name

# Output: input_data/default/input/512x512x64.bin (64 MB)
```

### Example 2: Generate Small Dataset
```bash
# Create small grid with Gaussian pattern
python generate.py synthetic \
  --out-dir input_data/small/input \
  --preset small \
  --kind gaussian \
  --seed 123 \
  --count 1 \
  --fixed-name

# Output: input_data/small/input/128x128x32.bin (2 MB)
```

### Example 3: Generate Multiple Mini Datasets
```bash
# Create 10 mini grids for testing
python generate.py synthetic \
  --out-dir input_data/mini/input \
  --preset mini \
  --kind random \
  --seed 0 \
  --count 10 \
  --start-index 0 \
  --digits 4

# Output: 
#   input_data/mini/input/input_0000_64x64x16.bin
#   input_data/mini/input/input_0001_64x64x16.bin
#   ...
#   input_data/mini/input/input_0009_64x64x16.bin
```

### Example 4: Generate Custom-Sized Grid
```bash
# Create custom 200×200×50 grid
python generate.py synthetic \
  --out-dir input_data/custom/input \
  --nx 200 \
  --ny 200 \
  --nz 50 \
  --kind uniform \
  --value 0.5 \
  --count 1 \
  --fixed-name

# Output: input_data/custom/input/200x200x50.bin (7.63 MB)
```

### Example 5: Generate Large Dataset
```bash
# Create large grid for performance testing
python generate.py synthetic \
  --out-dir input_data/large/input \
  --preset large \
  --kind random \
  --seed 999 \
  --count 1 \
  --fixed-name

# Output: input_data/large/input/768x768x96.bin (216 MB)
```

### Example 6: Generate Extra-Large Dataset
```bash
# Create extra-large grid
python generate.py synthetic \
  --out-dir input_data/extra-large/input \
  --preset extra-large \
  --kind random \
  --seed 2024 \
  --count 1 \
  --fixed-name

# Output: input_data/extra-large/input/1024x1024x128.bin (512 MB)
```

## Running the Benchmark

After generating input data, run the stencil benchmark:

```bash
# Using default preset (512×512×64 grid, 100 iterations)
./EX1_optimized_codes/stencil_gcc \
  -i input_data/default/input/512x512x64.bin \
  -o input_data/default/output/result.bin \
  -- 512 512 64 100

# Using small preset (128×128×32 grid, 100 iterations)
./EX1_optimized_codes/stencil_gcc \
  -i input_data/small/input/128x128x32.bin \
  -o input_data/small/output/result.bin \
  -- 128 128 32 100

# Using mini preset with fewer iterations
./EX1_optimized_codes/stencil_gcc \
  -i input_data/mini/input/64x64x16.bin \
  -o input_data/mini/output/result.bin \
  -- 64 64 16 50
```

**Command line arguments**:
- `-i <input_file>`: Input binary file
- `-o <output_file>`: Output file (optional)
- `-- <nx> <ny> <nz> <iterations>`: Grid dimensions and iteration count

**Note**: The grid dimensions passed to the benchmark **must match** the dimensions used to generate the input file.

## Preset Comparison

### Computational Cost

The computational cost scales with `nx × ny × nz × iterations`:

| Preset | Grid Elements | Iterations | Total Ops (×10⁶) | Relative Cost |
|--------|---------------|------------|------------------|---------------|
| mini | 65,536 | 50 | 3.3 | 1× |
| small | 524,288 | 100 | 52.4 | 16× |
| medium | 3,145,728 | 100 | 314.6 | 95× |
| default | 16,777,216 | 100 | 1,677.7 | 508× |
| large | 56,623,104 | 100 | 5,662.3 | 1,715× |
| extra-large | 134,217,728 | 100 | 13,421.8 | 4,066× |

### Memory Requirements

| Preset | Input Size | Working Memory | Total Memory |
|--------|-----------|----------------|--------------|
| mini | 256 KB | 512 KB | ~1 MB |
| small | 2 MB | 4 MB | ~6 MB |
| medium | 12 MB | 24 MB | ~36 MB |
| default | 64 MB | 128 MB | ~192 MB |
| large | 216 MB | 432 MB | ~648 MB |
| extra-large | 512 MB | 1 GB | ~1.5 GB |

The benchmark uses two copies of the grid (current and next), plus additional overhead.

## Technical Details

### Data Layout

The stencil code reads the grid in **z-major order**:

```c
for (iz = 0; iz < nz; iz++) {
    for (iy = 0; iy < ny; iy++) {
        for (ix = 0; ix < nx; ix++) {
            float value = grid[index++];
            // Process value
        }
    }
}
```

**Index calculation**: `index = iz * (ny * nx) + iy * nx + ix`

### 7-Point Stencil

The benchmark applies a 7-point stencil operator that reads:
- Current cell
- 6 neighboring cells (±x, ±y, ±z directions)

This pattern is iteratively applied across the grid for the specified number of iterations.

### Boundary Conditions

Cells at the grid boundaries have special handling since they don't have all 6 neighbors. The code typically:
- Skips boundary cells, or
- Uses zero/constant boundary conditions, or
- Uses periodic boundary conditions

## Validation

To verify generated data:

```bash
# Check file size
stat -c "%s bytes" input_data/default/input/512x512x64.bin
# Expected: 67108864 bytes (512 × 512 × 64 × 4)

# Read first few values (little-endian float32)
od -A none -t f4 -N 40 input_data/small/input/128x128x32.bin
```

## Troubleshooting

### Error: "Invalid grid dimensions"
- Ensure nx, ny, nz are all positive integers
- Check that `--preset` or all of `--nx`, `--ny`, `--nz` are specified

### Error: "Grid size mismatch"
- The grid dimensions must match exactly when running the benchmark
- Check that the input file was generated with the correct dimensions

### Error: "Output file already exists"
- Use `--overwrite` flag to replace existing files
- Or manually delete old files before regenerating

### Benchmark shows "time ≈ 0"
- Input file format may be incorrect (check endianness, data type)
- Grid dimensions passed to benchmark may not match the input file
- Input file may be corrupted or truncated

## License

This generator is part of the HPC-Bench benchmark suite and follows the Parboil benchmark license (University of Illinois).

