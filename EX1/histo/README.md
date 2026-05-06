## Histogram Input Data Generator

### Overview
`generate.py` produces binary input files (`img.bin`) for the histogram benchmark. It supports:
- `synthetic`: generate random pixel data with uniform or hotspot distributions
- `from-sources`: validate and copy existing `img.bin` files with consistent naming

### Requirements
- Python 3.8+
- Packages: `numpy`

Install:
```bash
pip install numpy
```

### File Format
Each `img.bin` contains:
- **Header** (16 bytes): 4 × uint32 little-endian values
  - `img_width`: image width in pixels
  - `img_height`: image height in pixels
  - `histo_width`: histogram width (number of bins horizontally)
  - `histo_height`: histogram height (number of bins vertically)
- **Body** (`img_width * img_height * 4` bytes): pixel values as uint32 little-endian, each in range `[0, histo_width * histo_height - 1]`

A sidecar `.meta.json` file is written alongside each output for debugging and reference.

### Presets
Built-in size presets (editable in source):
```python
PRESETS = {
    "mini":        (256,  256,  256, 1),  # (img_w, img_h, histo_w, histo_h)
    "default":     (1024, 1024, 1024, 1),
    "medium":      (2048, 2048, 1024, 1),
    "large":       (4096, 4096, 1024, 1),
    "extra-large": (8192, 8192, 1024, 1),
}
```

### CLI Synopsis
```bash
python EX1/histo/generate.py synthetic [options]
python EX1/histo/generate.py from-sources [options]
```

---

## Subcommand: `synthetic`
Generate synthetic histogram input files.

### Options
- `--out-dir PATH` (required): output directory (created recursively if missing)
- `--count INT` (default: 1): number of files to generate
- `--preset {mini,default,medium,large,extra-large}` (default: `default`): preset size configuration
- `--img-width INT`: override image width (must be used with `--img-height`)
- `--img-height INT`: override image height (must be used with `--img-width`)
- `--histo-width INT`: override histogram width (must be used with `--histo-height`)
- `--histo-height INT`: override histogram height (must be used with `--histo-width`)
- `--kind {uniform,hotspots}` (default: `uniform`): distribution type
  - `uniform`: uniformly random pixel values across all bins
  - `hotspots`: mix of random background and concentrated hotspot bins
- `--seed INT` (default: 0): base random seed for reproducibility
- `--start-index INT` (default: 0): starting index for file numbering
- `--digits INT` (default: 4): zero-padding width for indices
- `--overwrite`: allow overwriting existing files
- `--fixed-name`: write output as `img.bin` without index (use for single-file presets like `default/input/img.bin`)

### Examples

**Generate default-size input with fixed name:**
```bash
python EX1/histo/generate.py synthetic \
  --out-dir EX1/histo/input_data/default/input_gen \
  --count 1 --preset default --kind uniform --fixed-name
```
Output: `EX1/histo/input_data/default/input_gen/img.bin` (1024×1024 image, 1024×1 histogram)

**Generate multiple large inputs with indexed names:**
```bash
python EX1/histo/generate.py synthetic \
  --out-dir EX1/histo/input_data/large/input_gen \
  --count 3 --preset large --kind hotspots \
  --start-index 0 --digits 4
```
Output: `img_0000.bin`, `img_0001.bin`, `img_0002.bin` (each 4096×4096 with hotspot distribution)

**Generate mini-size input:**
```bash
python EX1/histo/generate.py synthetic \
  --out-dir EX1/histo/input_data/mini/input_gen \
  --count 1 --preset mini --kind uniform --fixed-name
```
Output: `img.bin` (256×256 image, 256×1 histogram)

**Generate medium-size input:**
```bash
python EX1/histo/generate.py synthetic \
  --out-dir EX1/histo/input_data/medium/input_gen \
  --count 1 --preset medium --kind uniform --fixed-name
```
Output: `img.bin` (2048×2048 image, 1024×1 histogram)

**Generate extra-large input:**
```bash
python EX1/histo/generate.py synthetic \
  --out-dir EX1/histo/input_data/extra-large/input_gen \
  --count 1 --preset extra-large --kind uniform --fixed-name
```
Output: `img.bin` (8192×8192 image, 1024×1 histogram)

**Custom size (e.g., 3000×2000 image, 512×1 histogram):**
```bash
python EX1/histo/generate.py synthetic \
  --out-dir EX1/histo/input_data/custom/input \
  --img-width 3000 --img-height 2000 \
  --histo-width 512 --histo-height 1 \
  --kind uniform --seed 42 --fixed-name
```

---

## Subcommand: `from-sources`
Copy and validate existing `img.bin` files with optional renaming.

### Options
- `--src-glob STR` (required): glob pattern for input files, e.g., `"input_data/default/input/*.bin"`
- `--out-dir PATH` (required): output directory (created recursively if missing)
- `--start-index INT` (default: 0): starting index for file numbering
- `--digits INT` (default: 4): zero-padding width for indices
- `--overwrite`: allow overwriting existing files
- `--fixed-name`: write output as `img.bin` without index

### Examples

**Copy and validate existing files with indexed naming:**
```bash
python EX1/histo/generate.py from-sources \
  --src-glob "EX1/histo/input_data/default/input/*.bin" \
  --out-dir EX1/histo/input_data/default/input_copy \
  --start-index 0 --digits 4
```

**Copy single file with fixed name:**
```bash
python EX1/histo/generate.py from-sources \
  --src-glob "EX1/histo/input_data/large/input/img.bin" \
  --out-dir EX1/histo/input_data/large/input_backup \
  --fixed-name --overwrite
```

---

## Running the Histogram Benchmark
After generating input data:
```bash
./histo_binary -i input_data/default/input_gen/img.bin -o output.bmp -- 20 4
```
Where:
- First argument after `--`: number of iterations
- Second argument: (varies by implementation)

Refer to `DESCRIPTION` files in existing `input_data/` subdirectories for parameter hints.

---

## Rules and Notes
- Output directories are created recursively.
- By default, existing files are **not** overwritten; use `--overwrite` to replace.
- All dimensions and pixel values are validated before writing.
- A `.meta.json` sidecar is written next to each output for reference (contains dimensions, seed, source, etc.).
- The `--fixed-name` option is useful when generating a single file to match existing directory structures (e.g., `default/input/img.bin`).
- For reproducibility, specify `--seed INT`.

---

## Preset Comparison

| Preset       | Image Size    | Histogram Bins | Total Pixels | File Size (approx) |
|--------------|---------------|----------------|--------------|---------------------|
| mini         | 256×256       | 256×1          | 65,536       | ~256 KB             |
| default      | 1024×1024     | 1024×1         | 1,048,576    | ~4 MB               |
| medium       | 2048×2048     | 1024×1         | 4,194,304    | ~16 MB              |
| large        | 4096×4096     | 1024×1         | 16,777,216   | ~64 MB              |
| extra-large  | 8192×8192     | 1024×1         | 67,108,864   | ~256 MB             |

Use larger presets for stress testing and performance benchmarking; use smaller presets for quick validation.

