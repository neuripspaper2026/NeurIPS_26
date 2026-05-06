# MRI-Q Input Generator

This `generate.py` script produces input data files for the MRI-Q benchmark in the expected binary format.

## File Format

Binary format (little-endian):
- `numK`: int32 (number of K-space samples)
- `numX`: int32 (number of X-space samples)
- `kx[numK]`: float32 array (K-space X coordinates)
- `ky[numK]`: float32 array (K-space Y coordinates)
- `kz[numK]`: float32 array (K-space Z coordinates)
- `x[numX]`: float32 array (X-space X coordinates)
- `y[numX]`: float32 array (X-space Y coordinates)
- `z[numX]`: float32 array (X-space Z coordinates)
- `phiR[numK]`: float32 array (phi real part)
- `phiI[numK]`: float32 array (phi imaginary part)

Total size: `8 + (numK * 5 + numX * 3) * 4` bytes

## Presets

The generator includes several size presets:

| Preset | numK | numX | Approx X-Grid | File Size | Description |
|--------|------|------|---------------|-----------|-------------|
| mini | 1,024 | 8,192 | 20³ | ~50 KB | Tiny, for quick tests |
| small | 3,072 | 32,768 | 32³ | ~445 KB | Reference small dataset |
| medium | 8,192 | 131,072 | 32×32×128 | ~1.7 MB | Moderate scale |
| large | 24,576 | 262,144 | 64³ | ~4.5 MB | Large scale |
| extra-large | 49,152 | 524,288 | 80³ | ~9 MB | Extra large scale |

You can also override individual parameters with explicit arguments.

## Usage

### Synthetic Generation

Generate random sample data using predefined distributions:

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
- `--numK INT`: Override numK (K-space samples)
- `--numX INT`: Override numX (X-space samples)
- `--kind {uniform,spiral}`: K-space sampling pattern (default: uniform)
  - `uniform`: Random uniform distribution in K-space
  - `spiral`: Spiral trajectory sampling
- `--seed INT`: Random seed (default: 0)
- `--start-index INT`: Starting index for filenames (default: 0)
- `--digits INT`: Zero-padding width (default: 4)
- `--overwrite`: Overwrite existing files
- `--fixed-name`: Use preset name for output

#### Examples:

**Generate 1 mini dataset:**
```bash
python EX1/mri-q/generate.py synthetic \
  --out-dir EX1/mri-q/mini/input \
  --count 1 --preset mini --seed 0 --fixed-name
```

**Generate 1 small dataset (reference):**
```bash
python EX1/mri-q/generate.py synthetic \
  --out-dir EX1/mri-q/small_gen/input \
  --count 1 --preset small --seed 42
```
This creates: `32_32_32_dataset.bin` (32³ X-space grid)

**Generate 1 medium dataset:**
```bash
python EX1/mri-q/generate.py synthetic \
  --out-dir EX1/mri-q/medium/input \
  --count 1 --preset medium --kind uniform --seed 123
```

**Generate 1 large dataset:**
```bash
python EX1/mri-q/generate.py synthetic \
  --out-dir EX1/mri-q/large_gen/input \
  --count 1 --preset large --seed 0
```
This creates: `64_64_64_dataset.bin` (64³ X-space grid)

**Generate 1 extra-large dataset:**
```bash
python EX1/mri-q/generate.py synthetic \
  --out-dir EX1/mri-q/extra_large/input \
  --count 1 --preset extra-large --seed 999
```
This creates: `80_80_80_dataset.bin` (80³ X-space grid)

**Generate with spiral K-space trajectory:**
```bash
python EX1/mri-q/generate.py synthetic \
  --out-dir EX1/mri-q/spiral_test/input \
  --count 1 --preset medium --kind spiral --seed 456
```

**Custom size (override preset):**
```bash
python EX1/mri-q/generate.py synthetic \
  --out-dir EX1/mri-q/custom/input \
  --count 1 --numK 5000 --numX 50000 \
  --kind uniform --seed 777
```

**Batch generation (5 files with different seeds):**
```bash
python EX1/mri-q/generate.py synthetic \
  --out-dir EX1/mri-q/batch/input \
  --count 5 --preset small --seed 100 \
  --start-index 0 --digits 4
```

### From Sources

Copy and normalize existing `.bin` files:

```bash
python generate.py from-sources \
  --src-glob "<GLOB_PATTERN>" \
  --out-dir <OUTPUT_DIR>
```

#### Parameters:

- `--src-glob PATTERN`: Glob pattern for source `.bin` files
- `--out-dir PATH`: Output directory
- `--start-index INT`: Starting index for output filenames
- `--digits INT`: Zero-padding width
- `--overwrite`: Overwrite existing files
- `--fixed-name`: Keep original filenames

#### Examples:

**Copy existing small dataset:**
```bash
python EX1/mri-q/generate.py from-sources \
  --src-glob "EX1/mri-q/input_data/small/input/*.bin" \
  --out-dir EX1/mri-q/small_copy/input \
  --fixed-name
```

**Batch copy and rename:**
```bash
python EX1/mri-q/generate.py from-sources \
  --src-glob "raw_data/*.bin" \
  --out-dir EX1/mri-q/normalized/input \
  --start-index 0 --digits 4
```

## Running the Benchmark

After generating data, run the benchmark:

```bash
cd EX1/mri-q
make
./EX1_optimized_codes/mri-q small/input/32_32_32_dataset.bin output.bin
```

Or with specific variants:

```bash
./EX1_optimized_codes/mri-q_gcc_claude_v10 large/input/64_64_64_dataset.bin output.bin
```

## Metadata

Each generated dataset includes a `.meta.json` sidecar file for debugging:
- `numK`: Number of K-space samples
- `numX`: Number of X-space samples
- `kind`: Sampling pattern (uniform/spiral)
- `seed`: Random seed used

## Notes

- All binary data uses little-endian format
- K-space coordinates typically range from -0.5 to 0.5
- X-space coordinates form a 3D grid (when numX is a perfect cube)
- Phi values (phiR, phiI) are complex numbers representing the MRI signal
- File sizes grow with `O(numK + numX)` where numK typically stores 5 floats per sample and numX stores 3 floats per sample

## Size Comparisons

| Preset | Total Floats | File Size | K-Space | X-Space |
|--------|--------------|-----------|---------|---------|
| mini | 29,696 | ~119 KB | 1K | 8K |
| small | 113,664 | ~455 KB | 3K | 32K |
| medium | 434,176 | ~1.7 MB | 8K | 128K |
| large | 909,312 | ~3.6 MB | 24K | 256K |
| extra-large | 1,818,624 | ~7.3 MB | 48K | 512K |

Formula: `Total Floats = numK × 5 + numX × 3`


