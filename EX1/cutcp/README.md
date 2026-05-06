## CUTCP Dataset Generator (PQR)

### Overview
`generate.py` produces PQR files compatible with the CUTCP benchmark. It supports:
- `synthetic`: random atoms within a cubic box, reproducible via seed
- `from-sources`: validate and copy existing PQRs into a unified naming scheme

### Quick Start
```bash
python datasets_parboil/cutcp/generate.py synthetic \
  --out-dir datasets_parboil/cutcp/small/input_gen \
  --count 2 --preset small --seed 0 --start-index 0 --digits 4

python datasets_parboil/cutcp/generate.py from-sources \
  --src-glob "datasets_parboil/cutcp/small/input/*.pqr" \
  --out-dir datasets_parboil/cutcp/small/input_copy \
  --start-index 0 --digits 4
```

### CLI
Common behavior:
- Output directories are created recursively.
- Indexing is zero-padded to `--digits` width.
- Errors are concise and in English.

Subcommand `synthetic`:
- `--out-dir PATH` (required): output directory
- `--count INT` (default: 1): number of files
- `--preset {small,large}` (default: small): sets default atoms and box
- `--seed INT` (default: 0): base RNG seed
- `--start-index INT` (default: 0): numbering start
- `--digits INT` (default: 4): zero-padding width
- `--overwrite`: allow overwriting existing files
- `--num-atoms INT`: override atoms per file
- `--box-half FLOAT`: override half box size (uniform range [-box_half, box_half])

Subcommand `from-sources`:
- `--src-glob GLOB` (required): input PQR files
- `--out-dir PATH` (required): output directory
- `--start-index INT` (default: 0)
- `--digits INT` (default: 4)
- `--overwrite`

### Format
- Plain-text PQR with a `CRYST1` line followed by `ATOM` lines.
- Minimal validation: presence of `CRYST1` and at least one parseable `ATOM` line.

### Notes
- This generator follows the general data rule: match consumer format, be reproducible, safe directory handling, and validate before writing.

