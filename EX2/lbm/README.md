## LBM Obstacle Field Generator

### Overview
`generate.py` produces obstacle field files (`.of`) for the Lattice Boltzmann Method benchmark. It supports:
- `synthetic`: generate obstacle patterns (empty, random, channel walls, cavity)
- `from-sources`: validate and copy existing `.of` files

### Requirements
- Python 3.8+
- No external packages required

### File Format
Each `.of` file is a text file with `SIZE_Z * (SIZE_Y + 1)` lines:
- Each line contains `SIZE_X` characters: `.` = fluid cell, `o` = obstacle cell
- A blank line appears after each `Y` slice (between `Z` layers)
- Total file size: `SIZE_Z * SIZE_Y * SIZE_X + (SIZE_Y + 1) * SIZE_Z` bytes

A sidecar `.meta.json` is written alongside each output for reference.

### Presets
Built-in size presets (editable in source):
```python
PRESETS = {
    "mini":       (60,  60,  80),   # (SIZE_X, SIZE_Y, SIZE_Z)
    "short":      (120, 120, 150),
    "medium":     (180, 180, 200),
    "long":       (240, 240, 250),
    "extra-long": (300, 300, 300),
}
```

### CLI Synopsis
```bash
python EX1/lbm/generate.py synthetic [options]
python EX1/lbm/generate.py from-sources [options]
```

---

## Subcommand: `synthetic`
Generate synthetic obstacle fields.

### Options
- `--out-dir PATH` (required): output directory (created recursively)
- `--count INT` (default: 1): number of files to generate
- `--preset {mini,short,medium,long,extra-long}`: preset size
- `--size-x INT`: override SIZE_X
- `--size-y INT`: override SIZE_Y
- `--size-z INT`: override SIZE_Z
- `--kind {empty,random,channel,cavity}` (default: `empty`): obstacle pattern
  - `empty`: all fluid (no obstacles)
  - `random`: random obstacles with density `--density`
  - `channel`: top/bottom walls for channel flow
  - `cavity`: box walls with open top (lid-driven cavity)
- `--density FLOAT` (default: 0.05): obstacle density for `random` kind
- `--seed INT` (default: 0): base RNG seed
- `--start-index INT` (default: 0): starting index
- `--digits INT` (default: 4): zero-padding width
- `--overwrite`: allow overwriting existing files
- `--fixed-name`: use fixed naming like `120_120_150_ldc.of`

### Examples

**Generate short preset with empty domain:**
```bash
python EX1/lbm/generate.py synthetic \
  --out-dir EX1/lbm/input_data/short/input_gen \
  --count 1 --preset short --kind empty --fixed-name
```
Output: `120_120_150_ldc.of` (120×120×150, all fluid)

**Generate long preset with cavity pattern:**
```bash
python EX1/lbm/generate.py synthetic \
  --out-dir EX1/lbm/input_data/long/input_gen \
  --count 1 --preset long --kind cavity --fixed-name
```
Output: `240_240_250_ldc.of` (240×240×250, box walls)

**Generate mini:**
```bash
python EX1/lbm/generate.py synthetic \
  --out-dir EX1/lbm/input_data/mini/input_gen \
  --count 1 --preset mini --kind empty --fixed-name
```
Output: `60_60_80_ldc.of`

**Generate medium:**
```bash
python EX1/lbm/generate.py synthetic \
  --out-dir EX1/lbm/input_data/medium/input_gen \
  --count 1 --preset medium --kind empty --fixed-name
```
Output: `180_180_200_ldc.of`

**Generate extra-long:**
```bash
python EX1/lbm/generate.py synthetic \
  --out-dir EX1/lbm/input_data/extra-long/input_gen \
  --count 1 --preset extra-long --kind empty --fixed-name
```
Output: `300_300_300_ldc.of`

**Custom size with random obstacles:**
```bash
python EX1/lbm/generate.py synthetic \
  --out-dir EX1/lbm/input_data/custom/input \
  --size-x 200 --size-y 200 --size-z 180 \
  --kind random --density 0.1 --seed 42 --fixed-name
```

**Multiple files with indexed naming:**
```bash
python EX1/lbm/generate.py synthetic \
  --out-dir EX1/lbm/input_data/short/input_multi \
  --count 3 --preset short --kind channel \
  --start-index 0 --digits 4
```
Output: `lbm_120x120x150_0000.of`, `lbm_120x120x150_0001.of`, `lbm_120x120x150_0002.of`

---

## Subcommand: `from-sources`
Copy and validate existing `.of` files.

### Options
- `--src-glob STR` (required): glob for input .of files
- `--out-dir PATH` (required): output directory (created recursively)
- `--start-index INT` (default: 0)
- `--digits INT` (default: 4)
- `--overwrite`: allow overwriting
- `--fixed-name`: use fixed naming

### Examples

**Copy existing file:**
```bash
python EX1/lbm/generate.py from-sources \
  --src-glob "EX1/lbm/input_data/short/input/*.of" \
  --out-dir EX1/lbm/input_data/short/input_copy \
  --fixed-name --overwrite
```

---

## Running the LBM Benchmark
After generating obstacle fields:
```bash
./lbm_binary -i input_data/short/input_gen/120_120_150_ldc.of -- 100
```
Where the argument after `--` is the number of timesteps.

Refer to `DESCRIPTION` files in existing `input_data/` subdirectories:
- `short`: 100 timesteps
- `long`: 3000 timesteps

---

## Rules and Notes
- Output directories are created recursively.
- By default, existing files are **not** overwritten; use `--overwrite` to replace.
- All dimensions are validated before writing.
- A `.meta.json` sidecar is written next to each output.
- The `--fixed-name` option produces filenames like `SIZE_X_SIZE_Y_SIZE_Z_ldc.of` matching existing structure.
- For reproducibility, specify `--seed INT`.

---

## Preset Comparison

| Preset     | Dimensions (X×Y×Z) | Total Cells | File Size (approx) |
|------------|--------------------|-----------  |--------------------|
| mini       | 60×60×80           | 288,000     | ~350 KB            |
| short      | 120×120×150        | 2,160,000   | ~2.6 MB            |
| medium     | 180×180×200        | 6,480,000   | ~7.8 MB            |
| long       | 240×240×250        | 14,400,000  | ~17.3 MB           |
| extra-long | 300×300×300        | 27,000,000  | ~32.4 MB           |

Use larger presets for stress testing; smaller presets for quick validation.

