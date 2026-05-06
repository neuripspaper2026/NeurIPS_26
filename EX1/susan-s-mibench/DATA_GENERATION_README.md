# SUSAN Data Generation Guide

## Overview

This document describes how to generate PGM (Portable Gray Map) image files for the SUSAN (Smallest Univalue Segment Assimilating Nucleus) corner/edge detection benchmark.

## PGM Format

SUSAN uses the **P5 (binary) PGM format**:

```
P5
# CREATOR: comment (optional)
<width> <height>
255
<binary pixel data: width×height bytes>
```

- **Header**: ASCII text
- **Pixel data**: Raw binary bytes (0-255), row-major order
- **Size**: File size ≈ width × height + ~50 bytes (header)

## Preset Sizes

| Preset | Dimensions | File Size | Description |
|--------|------------|-----------|-------------|
| **mini** | 76×95 | ~7 KB | Mini size (matches input_small.pgm) |
| **small** | 2048×1536 | ~3.0 MB | Small size |
| **medium** | 4096×3072 | ~12 MB | Medium size |
| **large** | 8192×6144 | ~48 MB | Large size |
| **extra-large** | 16384×12288 | ~192 MB | Extra-large size |

## Image Patterns

The generator supports multiple synthetic patterns:

1. **random**: Pure random noise (default)
   - Good for general-purpose testing
   - Maximum entropy

2. **gradient**: Horizontal intensity gradient
   - Smooth transitions
   - Tests edge detection

3. **checkerboard**: 8×8 pixel checkerboard
   - High-contrast corners
   - Tests corner detection

4. **edges**: Regular horizontal/vertical edges
   - Structured edge features every 32 pixels
   - Tests edge detection specifically

5. **corners**: Corner-like features
   - Circular regions with edge overlays
   - Optimal for SUSAN corner detector

## Usage

### 1. Show Available Presets

```bash
python generate.py info
```

### 2. Generate Synthetic Images

**Basic usage with preset:**
```bash
python generate.py synthetic --preset mini --out-dir input_data/mini/input
```

**With custom pattern:**
```bash
python generate.py synthetic --preset large --pattern corners --out-dir input_data/large/input
```

**With custom seed:**
```bash
python generate.py synthetic --preset medium --pattern edges --seed 42 --out-dir input_data/medium/input
```

**Custom dimensions:**
```bash
python generate.py synthetic --width 640 --height 480 --out-dir custom/ --name my_image.pgm
```

### 3. Generate All Sizes

**Batch generation script:**
```bash
#!/bin/bash
for size in mini small medium large extra-large; do
    python generate.py synthetic --preset $size --out-dir input_data/$size/input
    echo "✓ Generated $size"
done
```

### 4. Run SUSAN Benchmark

**Basic smoothing mode:**
```bash
./EX1_optimized_codes/susan_gcc input_data/mini/input/input_mini.pgm output.pgm -s
```

**Edge detection mode:**
```bash
./EX1_optimized_codes/susan_gcc input_data/large/input/input_large.pgm output.pgm -e
```

**Corner detection mode:**
```bash
./EX1_optimized_codes/susan_gcc input_data/large/input/input_large.pgm output.pgm -c
```

## Command-Line Options

### `synthetic` Subcommand

| Option | Type | Description |
|--------|------|-------------|
| `--preset` | str | Preset size (mini/small/medium/large/extra-large) |
| `--width` | int | Image width in pixels (overrides preset) |
| `--height` | int | Image height in pixels (overrides preset) |
| `--pattern` | str | Pattern type (random/gradient/checkerboard/edges/corners) |
| `--seed` | int | Random seed (default: 12345) |
| `--out-dir` | str | Output directory (required, created if needed) |
| `--name` | str | Output filename (default: auto-generated) |
| `--overwrite` | flag | Overwrite existing files |

## Reproducibility

- **Deterministic**: Same parameters → identical output
- **Seed control**: Use `--seed` to control randomness
- **Default seed**: 12345

**Example:**
```bash
# These two commands produce identical files
python generate.py synthetic --preset mini --seed 100 --out-dir test1/
python generate.py synthetic --preset mini --seed 100 --out-dir test2/
```

## Validation

The generator validates:
- ✓ Dimensions are positive
- ✓ Dimensions ≤ 8192×8192
- ✓ Output directory can be created
- ✓ Pixel count matches width×height
- ✓ File doesn't exist (unless `--overwrite`)

## File Naming Convention

**Auto-generated names:**
- With preset: `input_<preset>.pgm` (e.g., `input_mini.pgm`)
- Custom size: `input_<width>x<height>.pgm` (e.g., `input_640x480.pgm`)

**Custom names:**
```bash
python generate.py synthetic --preset mini --out-dir data/ --name my_test.pgm
```

## Examples

### Example 1: Generate Mini Size

```bash
python generate.py synthetic --preset mini --out-dir input_data/mini/input
```

**Output:**
```
Generating 76×95 PGM image (pattern: random, seed: 12345)...
✓ Generated input_data/mini/input/input_mini.pgm (76×95, 7,320 bytes)
```

### Example 2: Generate with Different Patterns

```bash
# Random noise
python generate.py synthetic --preset large --pattern random --seed 1 --out-dir test/

# Corners (best for SUSAN)
python generate.py synthetic --preset large --pattern corners --seed 2 --out-dir test/

# Edges
python generate.py synthetic --preset large --pattern edges --seed 3 --out-dir test/
```

### Example 3: Custom Dimensions

```bash
python generate.py synthetic --width 1024 --height 768 --pattern gradient --out-dir custom/
```

## Troubleshooting

**Error: "File already exists"**
- Solution: Use `--overwrite` flag or delete existing file

**Error: "Must specify --width and --height or use --preset"**
- Solution: Either use `--preset <name>` OR specify both `--width` and `--height`

**Error: "Invalid dimensions"**
- Solution: Ensure width and height are positive and ≤ 8192

**Binary runs but produces no output:**
- Check PGM format is correct (use `file input.pgm`)
- Verify header has correct dimensions
- Ensure pixel data size matches width×height

## Integration with Makefile

The `susan-s-mibench/Makefile` can run benchmarks on generated data:

```bash
# Generate data
python generate.py synthetic --preset mini --out-dir input_data/mini/input

# Run benchmark (requires manual input specification)
./EX1_optimized_codes/susan_gcc input_data/mini/input/input_mini.pgm output.pgm -s
```

## Notes

- **Image content**: Generated images are synthetic and may not resemble real photos
- **Corner detection**: Use `--pattern corners` for best SUSAN corner detector testing
- **Edge detection**: Use `--pattern edges` for edge detector testing
- **Smoothing**: Any pattern works for smoothing mode
- **Reproducibility**: Always use the same seed for identical results

