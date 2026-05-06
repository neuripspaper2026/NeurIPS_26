# SGEMM Input Generator

This `generate.py` script produces input data files for the SGEMM (Single-precision General Matrix Multiply) benchmark in the expected text format.

## File Format

Text format for matrices (column-major layout):
- **First line**: `rows cols` (two integers)
- **Following lines**: Space-separated float32 values in column-major order

For matrix multiplication `C = A × B`, the generator creates **four files**:
- `matrix1.txt`: Matrix A (M × K)
- `matrix1t.txt`: Matrix A transposed (K × M) - for optimization
- `matrix2.txt`: Matrix B (K × N)
- `matrix2t.txt`: Matrix B transposed (N × K) - for optimization

## Presets

The generator includes several size presets for `C[M×N] = A[M×K] × B[K×N]`:

| Preset | M | K | N | Total Floats | Description |
|--------|---|---|---|--------------|-------------|
| mini | 32 | 32 | 32 | ~8K | Tiny matrices for testing |
| small | 128 | 96 | 160 | ~58K | Reference small (matches parboil) |
| medium | 1024 | 992 | 1056 | ~4.1M | Medium scale (matches parboil) |
| large | 2048 | 1984 | 2112 | ~16.4M | Large scale |
| extra-large | 4096 | 3968 | 4224 | ~65.6M | Extra large scale |

Formula: Total floats = `M*K + K*M + K*N + N*K` (matrix1 + matrix1t + matrix2 + matrix2t)

**Note**: The `small` and `medium` presets match the original parboil dataset dimensions (non-square matrices).

You can also override individual parameters with explicit arguments.

## Usage

### Synthetic Generation

Generate random matrix data using predefined distributions:

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
- `--M INT`: Override M (rows of A, rows of C)
- `--K INT`: Override K (cols of A, rows of B)
- `--N INT`: Override N (cols of B, cols of C)
- `--kind {uniform,normal,identity,diagonal}`: Matrix generation pattern (default: uniform)
  - `uniform`: Random uniform distribution [0, 1]
  - `normal`: Random distribution [-1, 1]
  - `identity`: Identity matrix (requires M == K == N)
  - `diagonal`: Diagonal matrix (requires M == K == N)
- `--seed INT`: Random seed (default: 0)
- `--start-index INT`: Starting index for filenames (default: 0)
- `--digits INT`: Zero-padding width (default: 4)
- `--overwrite`: Overwrite existing files
- `--fixed-name`: Use fixed naming (no index prefix)

#### Examples:

**Generate 1 tiny dataset:**
```bash
python EX1/sgemm/generate.py synthetic \
  --out-dir EX1/sgemm/tiny/input \
  --count 1 --preset tiny --seed 0 --fixed-name
```

**Generate 1 small dataset (reference):**
```bash
python EX1/sgemm/generate.py synthetic \
  --out-dir EX1/sgemm/small_gen/input \
  --count 1 --preset small --seed 42 --fixed-name
```
This creates: `matrix1.txt`, `matrix2.txt`, `matrix2t.txt` (1024×1024)

**Generate 1 medium dataset:**
```bash
python EX1/sgemm/generate.py synthetic \
  --out-dir EX1/sgemm/medium_gen/input \
  --count 1 --preset medium --seed 123 --fixed-name
```
This creates matrices for 2048×2048 multiplication

**Generate 1 large dataset:**
```bash
python EX1/sgemm/generate.py synthetic \
  --out-dir EX1/sgemm/large_gen/input \
  --count 1 --preset large --seed 0 --fixed-name
```
This creates matrices for 4096×4096 multiplication

**Generate 1 extra-large dataset:**
```bash
python EX1/sgemm/generate.py synthetic \
  --out-dir EX1/sgemm/extra_large/input \
  --count 1 --preset extra-large --seed 999 --fixed-name
```
This creates matrices for 8192×8192 multiplication

**Generate with normal distribution:**
```bash
python EX1/sgemm/generate.py synthetic \
  --out-dir EX1/sgemm/normal_test/input \
  --count 1 --preset medium --kind normal --seed 456 --fixed-name
```

**Generate identity matrices:**
```bash
python EX1/sgemm/generate.py synthetic \
  --out-dir EX1/sgemm/identity/input \
  --count 1 --preset small --kind identity --fixed-name
```

**Custom rectangular matrices:**
```bash
python EX1/sgemm/generate.py synthetic \
  --out-dir EX1/sgemm/custom/input \
  --count 1 --M 512 --K 1024 --N 2048 \
  --kind uniform --seed 777 --fixed-name
```

**Batch generation (5 files with different seeds):**
```bash
python EX1/sgemm/generate.py synthetic \
  --out-dir EX1/sgemm/batch/input \
  --count 5 --preset tiny --seed 100 \
  --start-index 0 --digits 4
```
This creates: `0000_matrix1.txt`, `0001_matrix1.txt`, ..., `0004_matrix1.txt` (and corresponding matrix2/matrix2t files)

### From Sources

Copy and normalize existing matrix files:

```bash
python generate.py from-sources \
  --src-glob "<GLOB_PATTERN>" \
  --out-dir <OUTPUT_DIR>
```

#### Parameters:

- `--src-glob PATTERN`: Glob pattern for source `matrix1.txt` files
- `--out-dir PATH`: Output directory
- `--start-index INT`: Starting index for output filenames
- `--digits INT`: Zero-padding width
- `--overwrite`: Overwrite existing files
- `--fixed-name`: Keep original filenames

#### Examples:

**Copy existing small dataset:**
```bash
python EX1/sgemm/generate.py from-sources \
  --src-glob "EX1/sgemm/input_data/small/input/matrix1.txt" \
  --out-dir EX1/sgemm/small_copy/input \
  --fixed-name
```

**Batch copy and rename:**
```bash
python EX1/sgemm/generate.py from-sources \
  --src-glob "raw_data/*matrix1.txt" \
  --out-dir EX1/sgemm/normalized/input \
  --start-index 0 --digits 4
```

## Running the Benchmark

After generating data, run the benchmark:

```bash
cd EX1/sgemm
make
./EX1_optimized_codes/sgemm small/input/matrix1.txt small/input/matrix2t.txt small/input/matrix2t.txt
```

Or with specific variants:

```bash
./EX1_optimized_codes/sgemm_gcc_claude_v10 medium/input/matrix1.txt medium/input/matrix2t.txt medium/input/matrix2t.txt
```

## Metadata

Each generated dataset includes a `.meta.json` sidecar file for debugging:
- `M`: Number of rows in matrix A and C
- `K`: Number of cols in matrix A, rows in matrix B
- `N`: Number of cols in matrix B and C
- `kind`: Generation pattern (uniform/normal/identity/diagonal)
- `seed`: Random seed used

## Notes

- All matrices use **column-major** storage order (Fortran-style)
- Matrix multiplication: `C[M×N] = A[M×K] × B[K×N]`
- The `matrix2t.txt` file is the transpose of `matrix2.txt`, often used for cache-friendly access patterns
- File sizes grow with `O(M*K + K*N)` complexity
- For square matrices (M=K=N): Total floats = `3*N²`

## Size Comparisons

| Preset | M=K=N | Matrix A | Matrix B | Total Size (approx) |
|--------|-------|----------|----------|---------------------|
| tiny | 64 | 16 KB | 16 KB | ~50 KB |
| small | 1024 | 4 MB | 4 MB | ~12 MB |
| medium | 2048 | 16 MB | 16 MB | ~50 MB |
| large | 4096 | 64 MB | 64 MB | ~200 MB |
| extra-large | 8192 | 256 MB | 256 MB | ~800 MB |

Formula: Matrix size (bytes) ≈ `rows × cols × 4` (for float32)

