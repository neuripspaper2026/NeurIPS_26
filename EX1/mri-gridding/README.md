# MRI-gridding Input Generator

This `generate.py` script produces input data files for the MRI-gridding benchmark in the expected `.uks` and `.uks.data` binary format.

## File Format

- `.uks`: Text configuration file containing MRI acquisition parameters
- `.uks.data`: Binary file with reconstruction samples (24 bytes per sample)
  - Each sample: `kX, kY, kZ, real, imag, sdc` (6 floats, little-endian)

## Presets

The generator includes several size presets:

| Preset | Samples | Grid Size | Description |
|--------|---------|-----------|-------------|
| mini | 100,000 | 128³ | Tiny, for quick tests |
| small | 2,655,910 | 256³ | Reference small dataset |
| medium | 5,000,000 | 384³ | Moderate scale |
| large | 10,000,000 | 512³ | Large scale |
| extra-large | 20,000,000 | 640³ | Extra large scale |

You can also override individual parameters with explicit arguments.

## Usage

### Synthetic Generation

Generate random sample data using predefined distributions:

```bash
python generate.py synthetic \
  --out-dir <OUTPUT_DIR> \
  --count <NUM_FILES> \
  --preset <SIZE> \
  --kind <DISTRIBUTION> \
  --seed <SEED>
```

#### Parameters:

- `--out-dir PATH`: Output directory (created if needed)
- `--count INT`: Number of datasets to generate (default: 1)
- `--preset {mini,small,medium,large,extra-large}`: Preset size
- `--num-samples INT`: Override number of samples
- `--grid-size-x INT`, `--grid-size-y INT`, `--grid-size-z INT`: Override grid dimensions
- `--kind {uniform,radial}`: Sample distribution (default: uniform)
  - `uniform`: Random uniform distribution in k-space
  - `radial`: Radial trajectory sampling
- `--seed INT`: Random seed (default: 0)
- `--start-index INT`: Starting index for filenames (default: 0)
- `--digits INT`: Zero-padding width (default: 4)
- `--overwrite`: Overwrite existing files
- `--fixed-name`: Use preset name for output (e.g., `small.uks`)

#### Examples:

**Generate 1 mini dataset:**
```bash
python EX1/mri-gridding/generate.py synthetic \
  --out-dir EX1/mri-gridding/mini/input \
  --count 1 --preset mini --seed 0 --fixed-name
```

**Generate 5 small datasets with sequential numbering:**
```bash
python EX1/mri-gridding/generate.py synthetic \
  --out-dir EX1/mri-gridding/small_batch/input \
  --count 5 --preset small --kind uniform \
  --seed 42 --start-index 0 --digits 4
```
This creates: `mrig_0000.uks`, `mrig_0001.uks`, ..., `mrig_0004.uks`

**Generate medium dataset with radial sampling:**
```bash
python EX1/mri-gridding/generate.py synthetic \
  --out-dir EX1/mri-gridding/medium/input \
  --count 1 --preset medium --kind radial \
  --seed 123 --fixed-name
```

**Generate extra-large dataset:**
```bash
python EX1/mri-gridding/generate.py synthetic \
  --out-dir EX1/mri-gridding/extra_large/input \
  --count 1 --preset extra-large --seed 0 --fixed-name
```

**Custom size (override preset):**
```bash
python EX1/mri-gridding/generate.py synthetic \
  --out-dir EX1/mri-gridding/custom/input \
  --count 1 --num-samples 3000000 \
  --grid-size-x 300 --grid-size-y 300 --grid-size-z 300 \
  --kind uniform --seed 999
```

### From Sources

Copy and normalize existing `.uks` files:

```bash
python generate.py from-sources \
  --src-glob "<GLOB_PATTERN>" \
  --out-dir <OUTPUT_DIR>
```

#### Parameters:

- `--src-glob PATTERN`: Glob pattern for source `.uks` files
- `--out-dir PATH`: Output directory
- `--start-index INT`: Starting index for output filenames
- `--digits INT`: Zero-padding width
- `--overwrite`: Overwrite existing files
- `--fixed-name`: Keep original filenames

#### Examples:

**Copy existing small dataset:**
```bash
python EX1/mri-gridding/generate.py from-sources \
  --src-glob "EX1/mri-gridding/input_data/small/input/*.uks" \
  --out-dir EX1/mri-gridding/small_copy/input \
  --fixed-name
```

**Batch copy and rename:**
```bash
python EX1/mri-gridding/generate.py from-sources \
  --src-glob "raw_data/*.uks" \
  --out-dir EX1/mri-gridding/normalized/input \
  --start-index 0 --digits 4
```

## Running the Benchmark

After generating data, run the benchmark:

```bash
cd EX1/mri-gridding
./lbm_gcc <parameters> < small/input/small.uks
# or specify via command line as needed by your benchmark binary
```

Refer to the benchmark's `DESCRIPTION` file for exact parameter formats.

## Metadata

Each generated dataset includes a `.meta.json` sidecar file for debugging:
- `num_samples`: Number of reconstruction samples
- `grid_size`: 3D grid dimensions
- `kind`: Distribution type
- `seed`: Random seed used

## Notes

- All binary data uses little-endian float32 format
- Each sample is 24 bytes: 3 k-space coords + 2 complex signal + 1 density compensation
- Default parameters: kmax=150³, acq_size=60³, recon_size=60³, oversample=5.0, kernel_width=5.0, use_lut=1

