# TPACF Input Generator

This generator creates astronomical point data for the **TPACF** (Two-Point Angular Correlation Function) benchmark, which computes angular correlations between data points and random points on the celestial sphere.

## Input Data Format

The TPACF benchmark expects text files containing celestial coordinates:

```
Format: Text file with two columns
- Column 1: RA (Right Ascension) in degrees [0, 360)
- Column 2: DEC (Declination) in degrees [-90, 90]
- Space-separated values
- One point per line
```

**File naming convention**:
- `Datapnts.1`: Data points file
- `Randompnts.1`, `Randompnts.2`, ..., `Randompnts.N`: Random points files

**Example**:
```
0.0078478 -9.7811413
0.0092176 -9.8322327
0.0272316 0.5153435
```

## Usage

### Basic Syntax

```bash
python generate.py <subcommand> [options]
```

**Subcommands**:
- `synthetic`: Generate synthetic astronomical data
- `from-sources`: Convert external data files to TPACF format

### Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `--out-dir PATH` | Output directory (created if needed) | Required |
| `--preset NAME` | Use predefined size configuration | None |
| `--npoints INT` | Number of points per file | From preset |
| `--random-count INT` | Number of random point files | From preset |
| `--overwrite` | Overwrite existing files | False |

## Available Presets

| Preset | Points/File | Random Files | Total Points | Dataset Size |
|--------|-------------|--------------|--------------|--------------|
| `mini` | 100 | 100 | 10,100 | ~200 KB |
| `small` | 487 | 100 | 49,187 | ~1 MB |
| `medium` | 4,096 | 100 | 413,696 | ~8.5 MB |
| `large` | 10,391 | 100 | 1,049,491 | ~22 MB |
| `extra-large` | 20,000 | 100 | 2,020,000 | ~42 MB |

**Note**: All presets use 100 random files to match the TPACF benchmark requirements.

## Synthetic Data Generation

### Data Distribution Patterns

#### 1. **Uniform** (default)
Uniformly distributed points on the celestial sphere using equal-area projection.

```bash
python generate.py synthetic \
  --out-dir input_data/small/input \
  --preset small \
  --data-kind uniform \
  --seed 42 \
  --overwrite
```

**Best for**: Standard benchmark testing, unbiased correlation analysis.

#### 2. **Clustered**
Points organized in clusters to simulate galaxy clusters or star formations.

```bash
python generate.py synthetic \
  --out-dir input_data/test/input \
  --preset medium \
  --data-kind clustered \
  --seed 123 \
  --overwrite
```

**Best for**: Testing correlation detection in non-uniform distributions.

#### 3. **Grid**
Points arranged in a regular grid pattern with small random jitter.

```bash
python generate.py synthetic \
  --out-dir input_data/grid/input \
  --preset small \
  --data-kind grid \
  --seed 0 \
  --overwrite
```

**Best for**: Testing edge cases and validation.

### Synthetic Generation Options

| Option | Description | Default |
|--------|-------------|---------|
| `--data-kind` | Distribution pattern: `uniform`, `clustered`, `grid` | `uniform` |
| `--seed INT` | Random seed for reproducibility | None |

**Note**: Random point files are always generated with uniform distribution regardless of `--data-kind`.

## Converting External Data

Convert existing astronomical data to TPACF format:

```bash
# Convert files matching a glob pattern
python generate.py from-sources \
  --src-glob "raw_data/*.txt" \
  --out-dir input_data/converted/input \
  --preset small \
  --overwrite

# Convert specific files (first file = data, rest = random)
python generate.py from-sources \
  --src-list data.txt random1.txt random2.txt \
  --out-dir input_data/custom/input \
  --npoints 1000 \
  --random-count 2
```

**Notes**:
- First source file becomes `Datapnts.1`
- Subsequent files become `Randompnts.1`, `Randompnts.2`, etc.
- Files with fewer points are padded with random uniform points
- Files with more points are truncated
- If fewer than `random-count + 1` files are provided, missing random files are generated

## Complete Examples

### Example 1: Generate Mini Dataset
```bash
# Quick testing with minimal data
python generate.py synthetic \
  --out-dir input_data/mini/input \
  --preset mini \
  --data-kind uniform \
  --seed 0 \
  --overwrite

# Output:
#   Datapnts.1 (100 points)
#   Randompnts.1 through Randompnts.100 (100 points each)
```

### Example 2: Generate Small Dataset
```bash
# Standard small dataset
python generate.py synthetic \
  --out-dir input_data/small/input \
  --preset small \
  --data-kind uniform \
  --seed 42 \
  --overwrite

# Output:
#   Datapnts.1 (487 points)
#   Randompnts.1 through Randompnts.100 (487 points each)
```

### Example 3: Generate Medium Dataset
```bash
# Medium-sized dataset
python generate.py synthetic \
  --out-dir input_data/medium/input \
  --preset medium \
  --data-kind uniform \
  --seed 123 \
  --overwrite

# Output:
#   Datapnts.1 (4,096 points)
#   Randompnts.1 through Randompnts.100 (4,096 points each)
```

### Example 4: Generate Large Dataset
```bash
# Large dataset
python generate.py synthetic \
  --out-dir input_data/large/input \
  --preset large \
  --data-kind uniform \
  --seed 999 \
  --overwrite

# Output:
#   Datapnts.1 (10,391 points)
#   Randompnts.1 through Randompnts.100 (10,391 points each)
```

### Example 5: Generate Extra-Large Dataset
```bash
# Extra-large dataset
python generate.py synthetic \
  --out-dir input_data/extra-large/input \
  --preset extra-large \
  --data-kind uniform \
  --seed 2024 \
  --overwrite

# Output:
#   Datapnts.1 (20,000 points)
#   Randompnts.1 through Randompnts.100 (20,000 points each)
```

### Example 6: Generate Custom Dataset with Clustered Data
```bash
# Custom size with clustered distribution
python generate.py synthetic \
  --out-dir input_data/clustered/input \
  --npoints 2000 \
  --random-count 50 \
  --data-kind clustered \
  --seed 555 \
  --overwrite

# Output:
#   Datapnts.1 (2,000 points, clustered)
#   Randompnts.1 through Randompnts.50 (2,000 points each, uniform)
```

## Running the Benchmark

After generating input data, run the TPACF benchmark:

```bash
# Method 1: Using script to build file list
cd ${REPO_ROOT}/EX1/tpacf

INPUT_DIR="input_data/small/input"
FILES="${INPUT_DIR}/Datapnts.1"
for i in {1..100}; do 
  FILES="${FILES},${INPUT_DIR}/Randompnts.$i"
done

./EX1_optimized_codes/tpacf_gcc \
  -i ${FILES} \
  -o input_data/small/output/result.txt \
  -- -n 100 -p 487

# Method 2: One-line command
cd ${REPO_ROOT}/EX1/tpacf && \
INPUT_DIR="input_data/medium/input" && \
FILES="${INPUT_DIR}/Datapnts.1" && \
for i in {1..100}; do FILES="${FILES},${INPUT_DIR}/Randompnts.$i"; done && \
./EX1_optimized_codes/tpacf_gcc -i ${FILES} -o input_data/medium/output/result.txt -- -n 100 -p 4096
```

**Command line parameters**:
- **`-i`**: Comma-separated list of input files (no spaces!)
  - First file: Data points (`Datapnts.1`)
  - Rest: Random points (`Randompnts.1` through `Randompnts.N`)
- **`-o`**: Output file
- **`--`**: Separator between Parboil and program arguments
- **`-n`**: Number of random files (must match actual count)
- **`-p`**: Number of points per file (must match data)

## Preset Comparison

### Computational Cost

The TPACF algorithm computes correlations between all pairs of points, so computational cost scales as:
- **DD**: Data-Data pairs = O(N²)
- **DR**: Data-Random pairs = O(N² × M) where M = random_count
- **RR**: Random-Random pairs = O(N² × M)

| Preset | Points | Files | Total Pairs | Relative Cost |
|--------|--------|-------|-------------|---------------|
| mini | 100 | 100 | ~1M | 1× |
| small | 487 | 100 | ~24M | 24× |
| medium | 4,096 | 100 | ~1.7B | 1,680× |
| large | 10,391 | 100 | ~10.9B | 10,800× |
| extra-large | 20,000 | 100 | ~40.2B | 40,000× |

### Memory Requirements

| Preset | Per-File Memory | Total Memory | File Size |
|--------|----------------|--------------|-----------|
| mini | ~2 KB | ~200 KB | ~200 KB |
| small | ~10 KB | ~1 MB | ~1 MB |
| medium | ~82 KB | ~8.5 MB | ~8.5 MB |
| large | ~208 KB | ~22 MB | ~22 MB |
| extra-large | ~400 KB | ~42 MB | ~42 MB |

## Technical Details

### Astronomical Coordinates

- **RA (Right Ascension)**: Longitude on the celestial sphere, [0, 360) degrees
- **DEC (Declination)**: Latitude on the celestial sphere, [-90, 90] degrees

The benchmark converts these to Cartesian coordinates:
```c
x = cos(ra) * cos(dec)
y = sin(ra) * cos(dec)
z = sin(dec)
```

### Uniform Sphere Sampling

To generate uniformly distributed points on a sphere:
1. Sample `u` uniformly from [-1, 1] (represents cos(dec))
2. Compute `dec = arcsin(u)` 
3. Sample `ra` uniformly from [0, 360)

This ensures equal-area projection without polar bunching.

### Angular Correlation Function

The two-point correlation function computes:
```
w(θ) = DD/RR - 1
```

Where:
- **DD**: Count of data-data pairs at angular separation θ
- **DR**: Count of data-random pairs at angular separation θ
- **RR**: Count of random-random pairs at angular separation θ

## Validation

To verify generated data:

```bash
# Check number of files
ls input_data/small/input/ | grep -c Randompnts
# Expected: 100

# Check number of points per file
wc -l input_data/small/input/Datapnts.1
# Expected: 487

wc -l input_data/small/input/Randompnts.1
# Expected: 487

# Check data format (should be two floats per line)
head -3 input_data/small/input/Datapnts.1
```

## Troubleshooting

### Error: "read X points out of Y"
- The `-p` parameter doesn't match the actual number of points in files
- Verify with: `wc -l input_data/*/input/Datapnts.1`
- Regenerate data with correct `--npoints` value

### Error: "Invalid RA/DEC values"
- RA must be in [0, 360)
- DEC must be in [-90, 90]
- Check source files for out-of-range values

### File count mismatch
- Ensure `-n` parameter matches the number of `Randompnts.*` files
- Standard is 100 random files for all presets

### Benchmark runs but produces unexpected results
- Verify data format: two space-separated floats per line
- Check for non-numeric data or empty lines
- Ensure consistent point counts across all files

## License

This generator is part of the HPC-Bench benchmark suite and follows the Parboil benchmark license (University of Illinois).

