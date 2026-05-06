## generate.py - SAD Input Data Generator

### Overview
This tool generates binary input pairs for a Sum of Absolute Differences (SAD) program. It can:
- Synthesize gray-scale image pairs at one or multiple resolutions
- Convert existing image pairs into `.bin` files with size alignment rules
- Name outputs in either `ref_XXXX.bin/cur_XXXX.bin` or `reference_XXXX.bin/frame_XXXX.bin`

All outputs are now written in Parboil16 format required by the SAD code:
- 2-byte little-endian width (uint16), then 2-byte little-endian height (uint16)
- Followed by width*height pixels as 16-bit little-endian values
- Pixels originate from 8-bit grayscale and are upcasted to 16-bit
- Dimensions must be multiples of 16

### Requirements
- Python 3.8+
- Packages: `numpy`, `Pillow`

Install:
```bash
pip install numpy pillow
```

### File Format
- Binary `.bin`: Parboil16 (uint16 header W,H; uint16 pixel data, little-endian)
- Sidecar `.shape.txt`: a text file next to each `.bin` containing: `W H` (one line)

### Presets
Built-in presets (you can change them in the source if needed):
```python
PRESETS = {
    "default": (640, 640),
    "large": (1920, 1920)
}
```

### CLI Synopsis
```bash
python EX1/sad/generate.py synthetic [options]
python EX1/sad/generate.py from-images [options]
```

### Subcommand: synthetic
Generate synthetic pairs for one or multiple sizes.

Options:
- `--out-dir STR` (required): output directory (created recursively if missing)
- `--count INT` (default: 10): number of pairs per size
- `--preset {default,large}`: use a preset size if no custom size is provided
- `--width INT` and `--height INT`: override preset with a single custom size; both must be provided and be multiples of 16
- `--sizes STR`: multiple sizes list like `"640x640,1280x736,1920x1088"` (each must be multiple of 16); takes precedence over `--width/--height`
- `--by-size-subdir`: if set, outputs go into subfolders `out-dir/WxH/`
- `--size-in-name`: include size tag in filenames, e.g., `ref_640x640_0000.bin`
- `--kinds LIST`: choose from `shift noise checkerboard gradient blocks_mix` (space-separated); defaults to all
- `--seed INT` (default: 0): base RNG seed
- `--dx INT` (default: 4), `--dy INT` (default: 2): translation when applicable
- `--name-style {sad,alt}` (default: `sad`): naming style
  - `sad`: `ref_XXXX.bin` + `cur_XXXX.bin`
  - `alt`: `reference_XXXX.bin` + `frame_XXXX.bin`
- `--start-index INT` (default: 0): starting index for filenames
- `--digits INT` (default: 4): zero-padding width for indices

Examples:
```bash
# Single custom size with labels in file names
python EX1/sad/generate.py synthetic \
  --out-dir out \
  --count 10 \
  --width 1280 --height 736 \
  --size-in-name --start-index 0 --digits 4

# Multiple sizes at once, one subdir per size
python EX1/sad/generate.py synthetic \
  --out-dir out \
  --count 5 \
  --sizes "640x480,1280x736,1920x1088" \
  --by-size-subdir --size-in-name

# Using a preset and alternate naming style
python EX1/sad/generate.py synthetic \
  --out-dir out/large \
  --count 10 \
  --preset large \
  --name-style alt
```

### Subcommand: from-images
Convert two image lists into aligned `.bin` pairs.

Options:
- `--ref-glob STR` (required): glob for reference images, e.g., `"ref_dir/*.png"`
- `--cur-glob STR` (required): glob for current images, e.g., `"cur_dir/*.png"` (counts must match `--ref-glob`)
- `--out-dir STR` (required): output directory (created recursively if missing)
- `--preset {default,large}`: target size if no custom size is provided
- `--width INT`, `--height INT`: custom target size (multiples of 16)
- `--fit-mode {crop,pad,resize}` (default: `crop`): how to match target size
  - `crop`: crop top-left to target size (or down to nearest multiple of 16 when no explicit target)
  - `pad`: pad with black to target size (or up to nearest multiple of 16 when no explicit target)
  - `resize`: resample to target size (may change aspect ratio)
- `--name-style {sad,alt}`: naming style as above
- `--start-index INT` (default: 0): starting index for filenames
- `--digits INT` (default: 4): zero-padding width for indices

Examples:
```bash
# Crop to nearest multiples of 16, keep default naming
python EX1/sad/generate.py from-images \
  --ref-glob "refs/*.png" \
  --cur-glob "curs/*.png" \
  --out-dir out/default \
  --fit-mode crop

# Force exact target size (must be multiples of 16)
python EX1/sad/generate.py from-images \
  --ref-glob "refs/*.jpg" \
  --cur-glob "curs/*.jpg" \
  --out-dir out/alt \
  --width 1280 --height 736 \
  --fit-mode resize \
  --name-style alt
```

### Using the Outputs with SAD
Run the SAD binary with any generated pair:
```bash
./sad -i ref_0000.bin,cur_0000.bin -o result.bin
# or with alternate naming
./sad -i reference_0000.bin,frame_0000.bin -o result.bin
```

### Rules and Notes
- Every size must be a multiple of 16 in both width and height.
- Priority of size options: `--sizes` > `--width/--height` > `--preset`.
- When `--sizes` is used, each listed size gets `--count` pairs.
- When `--by-size-subdir` is on, outputs are placed in `out-dir/WxH/`.
- When `--size-in-name` is on, size tags like `640x480` are included in file names.
- A `.shape.txt` file is written next to each output `.bin` recording `W H`.
 - Outputs are Parboil16 by default and directly consumable by the provided SAD code.

### Synthetic Content Types
The `--kinds` option can include one or more of:
- `shift`: checkerboard reference with translated current image `(dx, dy)`
- `noise`: independent random noise frames
- `checkerboard`: differing checkerboard tile sizes between ref and cur
- `gradient`: horizontal vs vertical gradient
- `blocks_mix`: rectangles over a gradient, then translated


