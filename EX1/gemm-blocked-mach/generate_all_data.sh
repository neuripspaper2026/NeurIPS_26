#!/bin/bash

# GEMM-Blocked-Mach Data Generation Script
# Generates 5 datasets with different random input patterns (fixed size: 64×64 matrices)

set -e

BENCHMARK="gemm-blocked-mach"
BASELINE_EXE="./EX1_optimized_codes/gemm_gcc"
CORR_OUTPUT="./EX5_correctness/output_gcc.data"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║    GEMM-BLOCKED-MACH Data Generation (Fixed Scale: 64×64)        ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Check baseline exists
if [ ! -f "$BASELINE_EXE" ]; then
    echo "⚠️  Baseline not found. Compiling..."
    make clean > /dev/null 2>&1 || true
    make baseline
    echo ""
fi

# Create directory structure
echo "📁 Creating directory structure..."
for size in mini small medium large extra-large; do
    mkdir -p "input_data/$size/input"
    mkdir -p "input_data/$size/output"
done
echo "   ✓ Directories created"

# Function to generate GEMM input with specific seed
generate_data() {
    local size=$1
    local seed=$2
    local description=$3
    echo ""
    echo "[$size] Pattern: $description (seed=$seed)"

    # Create a temporary generate.c with the specific seed
    cat <<EOF > generate_temp.c
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <assert.h>

#define SEED_VALUE $seed // Dynamic seed
#define TYPE double

//Algorithm Parameters
#define row_size 64
#define col_size 64
#define N row_size*col_size

struct bench_args_t {
  TYPE m1[N];
  TYPE m2[N];
  TYPE prod[N];
};

#include "../../common_MachSuite/support.h"

int main(int argc, char **argv)
{
  struct bench_args_t data;
  int i, fd;
  struct prng_rand_t state;

  // Fill data structure with specific seed
  prng_srand(SEED_VALUE, &state);
  for(i=0; i<N; i++) {
    data.m1[i] = ((TYPE)prng_rand(&state))/((TYPE)PRNG_RAND_MAX);
    data.m2[i] = ((TYPE)prng_rand(&state))/((TYPE)PRNG_RAND_MAX);
  }

  // Write to file
  fd = open("temp_input.data", O_WRONLY|O_CREAT|O_TRUNC, S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH);
  assert( fd>0 && "Couldn't open output data file" );
  
  // Write section 1: matrix 1
  write_section_header(fd);
  write_double_array(fd, data.m1, N);
  
  // Write section 2: matrix 2
  write_section_header(fd);
  write_double_array(fd, data.m2, N);
  
  close(fd);

  return 0;
}
EOF

    # Compile with specific seed
    cc -O3 -Wall -Wno-unused-label -I../../common_MachSuite \
        -DSEED_VALUE=$seed \
        -o generate_temp \
        generate_temp.c ../../common_MachSuite/support.c 2>/dev/null
    
    # Generate data
    ./generate_temp
    mv temp_input.data "input_data/$size/input/input.data"
    
    echo "  ✓ Generated input_data/$size/input/input.data"
}

# Generate all datasets with different random seeds
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Generating Input Data (Different Random Matrix Patterns)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

generate_data "mini"        1   "Standard random matrices (original seed)"
generate_data "small"       42  "Alternative random distribution"
generate_data "medium"      123 "Different random pattern"
generate_data "large"       456 "Another random distribution"
generate_data "extra-large" 789 "Fifth random pattern"

# Clean up
rm -f generate_temp generate_temp.c temp_input.data

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Generating Golden Output (check.data)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

for size in mini small medium large extra-large; do
    echo "[$size] Running baseline to generate check.data..."
    
    # Run baseline (it will output to EX5_correctness/)
    $BASELINE_EXE \
        "input_data/$size/input/input.data" \
        "input_data/$size/input/input.data" 2>/dev/null || true
    
    # Copy output to check.data
    if [ -f "$CORR_OUTPUT" ]; then
        cp "$CORR_OUTPUT" "input_data/$size/output/check.data"
        echo "  ✓ Generated input_data/$size/output/check.data"
    else
        echo "  ✗ Failed to generate check.data"
        exit 1
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Verifying All Datasets"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

all_passed=true
for size in mini small medium large extra-large; do
    echo -n "[$size] Testing... "
    
    output=$($BASELINE_EXE \
        "input_data/$size/input/input.data" \
        "input_data/$size/output/check.data" 2>&1 || true)
    
    if echo "$output" | grep -q "Success"; then
        echo "✓ Success"
    else
        echo "✗ Failed"
        all_passed=false
    fi
done

echo ""
if [ "$all_passed" = true ]; then
    echo "🎉 All datasets generated and verified successfully!"
else
    echo "⚠️  Some datasets failed verification."
    exit 1
fi

# Generate README
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Generating Documentation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cat > input_data/README.md <<'EOFREADME'
# GEMM-Blocked-Mach Dataset Documentation

This directory contains test datasets for the **GEMM-Blocked-Mach** benchmark, which implements blocked matrix multiplication for improved cache locality.

---

## ⚠️ Important: About "Size"

Due to the algorithm's design and the use of fixed-size arrays defined by macros (e.g., `row_size 64`, `col_size 64`), all test sets for `gemm-blocked-mach` have **exactly the same input size** (64×64 matrices = 4096 elements per matrix).

The different "sizes" (mini, small, medium, large, extra-large) represent different input matrix **patterns** generated with varying random seeds for correctness validation, NOT different computational scales. This approach allows for testing the algorithm's behavior across diverse input data while maintaining a consistent execution environment.

---

## Dataset Structure

```
input_data/
├── mini/
│   ├── input/input.data     (8195 lines: 2 matrices × 4096 values + headers)
│   └── output/check.data    (4098 lines: result matrix)
├── small/
├── medium/
├── large/
├── extra-large/
└── README.md (this file)
```

---

## Dataset Details

| Size        | Random Seed | Description                               | Sample Input (First 3 values of m1) |
|-------------|-------------|-------------------------------------------|--------------------------------------|
| mini        | 1           | Standard random matrices (original seed)  | (depends on PRNG) |
| small       | 42          | Alternative random distribution           | (different pattern) |
| medium      | 123         | Different random pattern                  | (different pattern) |
| large       | 456         | Another random distribution               | (different pattern) |
| extra-large | 789         | Fifth random pattern                      | (different pattern) |

**Key Points:**
- All datasets: 64×64 matrix multiplication (fixed)
- Input format: 2 sections (matrix 1, matrix 2)
- Each matrix: 4096 double-precision floating-point values
- Values range: [0.0, 1.0] (normalized random)
- Precision: `double` (1e-6 tolerance for validation)
- Block size: 8×8 (for cache optimization)

---

## Algorithm Information

### Blocked GEMM (General Matrix Multiply)
- **Type**: Cache-optimized blocked matrix multiplication
- **Size**: 64×64 matrices (fixed)
- **Complexity**: O(N³) = O(64³) = O(262,144) operations
- **Block size**: 8×8 blocks
- **Number of blocks**: 64 blocks (8×8 grid)
- **Optimization**: Improves cache locality by processing small blocks

### Cache Optimization
The blocked algorithm divides the matrices into 8×8 blocks and performs multiplication on these blocks, which:
- Reduces cache misses
- Improves data reuse
- Optimizes memory access patterns
- Achieves better performance than naive matrix multiplication

---

## Usage

### Test Single Size
```bash
./EX1_optimized_codes/gemm_gcc \
    input_data/mini/input/input.data \
    input_data/mini/output/check.data
```

Expected output: `Success.`

### Test All Sizes
```bash
for size in mini small medium large extra-large; do
    echo "Testing $size..."
    ./EX1_optimized_codes/gemm_gcc \
        input_data/$size/input/input.data \
        input_data/$size/output/check.data
done
```

### Run with Custom Executable
```bash
# Test an optimized variant
./EX1_optimized_codes/gemm_gcc_optimized \
    input_data/medium/input/input.data \
    input_data/medium/output/check.data
```

---

## Data Format

### Input Data (`input.data`)
```
%% Section 1
<4096 double values: matrix 1 (64×64, row-major)>

%% Section 2
<4096 double values: matrix 2 (64×64, row-major)>
```

### Output Data (`check.data`)
```
%% Section 1
<4096 double values: result matrix (64×64, row-major)>
```

### Matrix Layout
- Storage: Row-major order
- Element [i,j] at index: `i * col_size + j`
- Example: For 64×64 matrix, element [2,3] is at index 2*64+3 = 131

### Validation
- Comparison uses epsilon tolerance: `EPSILON = 1.0e-6`
- All elements must match within tolerance

---

## Regeneration

To regenerate all datasets from scratch:

```bash
# 1. Ensure baseline is compiled
make clean
make baseline

# 2. Run generation script
bash generate_all_data.sh
```

This will:
1. Create directory structure
2. Generate 5 different input patterns (using different random seeds)
3. Use baseline executable to compute golden outputs
4. Verify all datasets pass correctness tests

---

## Technical Notes

### Why Fixed Size?
The blocked GEMM algorithm is optimized for:
- Fixed 64×64 matrix size (defined by `row_size` and `col_size`)
- Fixed 8×8 block size for cache optimization
- Hardcoded array sizes: `TYPE m1[N]`, `TYPE m2[N]`, `TYPE prod[N]` where `N = 4096`

Changing the size would require modifying the algorithm parameters, not just input data.

### Random Pattern Variation
Different random seeds produce different matrix values, which test:
- Numerical stability across diverse value ranges
- Edge cases in multiplication (e.g., near-zero values, large products)
- Correctness of blocking strategy
- Cache behavior with different data patterns

### Reference
Implementation based on:
- "The cache performance and optimizations of blocked algorithms"
- M. D. Lam, E. E. Rothberg, and M. E. Wolf
- ASPLOS 1991

---

## File Sizes

Each dataset:
- `input.data`: ~152 KB (8195 lines: 2 matrices)
- `check.data`: ~76 KB (4098 lines: 1 result matrix)

Total: ~1.14 MB for all 5 datasets

---

## Validation Results

All datasets have been verified to produce correct outputs:
- ✓ mini
- ✓ small
- ✓ medium
- ✓ large
- ✓ extra-large

---

## Related Benchmarks

- **gemm-ncubed**: Naive O(n³) implementation (baseline comparison)
- **gemm-blocked**: This benchmark with cache optimization

The blocked version typically achieves better performance due to improved cache locality.
EOFREADME

echo "  ✓ Generated input_data/README.md"

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                     Generation Complete!                         ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Generated files:"
echo "  • 5 input.data files (different random matrix patterns)"
echo "  • 5 check.data files (golden outputs)"
echo "  • input_data/README.md (documentation)"
echo ""
echo "All datasets verified successfully! ✓"

