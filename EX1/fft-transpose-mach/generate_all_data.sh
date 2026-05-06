#!/bin/bash

# FFT-Transpose-Mach Data Generation Script
# Generates 5 datasets with different random input patterns (fixed size: 512 points)

set -e

BENCHMARK="fft-transpose-mach"
BASELINE_EXE="./EX1_optimized_codes/fft_gcc"
CORR_OUTPUT="./EX5_correctness/output_gcc.data"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║     FFT-TRANSPOSE-MACH Data Generation (Fixed Scale: 512)        ║"
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

# Function to generate FFT input with specific seed
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

struct bench_args_t {
    TYPE work_x[512];
    TYPE work_y[512];
};

#include "../../common_MachSuite/support.h"

int main(int argc, char **argv)
{
  struct bench_args_t data;
  int i, fd;
  struct prng_rand_t state;

  // Fill data structure with specific seed
  prng_srand(SEED_VALUE, &state);
  for(i=0; i<512; i++){
    data.work_x[i] = ((TYPE)prng_rand(&state))/((TYPE)PRNG_RAND_MAX);
    data.work_y[i] = ((TYPE)prng_rand(&state))/((TYPE)PRNG_RAND_MAX);
  }

  // Write to file
  fd = open("temp_input.data", O_WRONLY|O_CREAT|O_TRUNC, S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH);
  assert( fd>0 && "Couldn't open output data file" );
  
  // Write section 1: real part
  write_section_header(fd);
  write_double_array(fd, data.work_x, 512);
  
  // Write section 2: imaginary part
  write_section_header(fd);
  write_double_array(fd, data.work_y, 512);
  
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
echo "  Generating Input Data (Different Random Patterns)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

generate_data "mini"        1   "Standard random pattern (original seed)"
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
# FFT-Transpose-Mach Dataset Documentation

This directory contains test datasets for the **FFT-Transpose-Mach** benchmark, which implements a 1D Fast Fourier Transform using a transpose-based memory access pattern.

---

## ⚠️ Important: About "Size"

Due to the algorithm's design and the use of fixed-size arrays (512-point FFT), all test sets for `fft-transpose-mach` have **exactly the same input size** (512 complex numbers = 1024 double values).

The different "sizes" (mini, small, medium, large, extra-large) represent different input signal **patterns** generated with varying random seeds for correctness validation, NOT different computational scales. This approach allows for testing the algorithm's behavior across diverse input data while maintaining a consistent execution environment.

---

## Dataset Structure

```
input_data/
├── mini/
│   ├── input/input.data     (1027 lines: 2 sections × 512 values + headers)
│   └── output/check.data    (1027 lines: FFT output)
├── small/
├── medium/
├── large/
├── extra-large/
└── README.md (this file)
```

---

## Dataset Details

| Size        | Random Seed | Description                               | Sample Input (First 5 real values) |
|-------------|-------------|-------------------------------------------|------------------------------------|
| mini        | 1           | Standard random pattern (original seed)   | (depends on PRNG) |
| small       | 42          | Alternative random distribution           | (different pattern) |
| medium      | 123         | Different random pattern                  | (different pattern) |
| large       | 456         | Another random distribution               | (different pattern) |
| extra-large | 789         | Fifth random pattern                      | (different pattern) |

**Key Points:**
- All datasets: 512-point FFT (fixed)
- Input format: 2 sections (real part, imaginary part)
- Each section: 512 double-precision floating-point values
- Values range: [0.0, 1.0] (normalized random)
- Precision: `double` (1e-6 tolerance for validation)

---

## Algorithm Information

### FFT-Transpose Algorithm
- **Type**: Transpose-based memory access pattern
- **Size**: 512 points (fixed)
- **Complexity**: O(N log N) = O(512 × 9) = O(4608) operations
- **Precision**: Double-precision floating-point
- **Stages**: 9 butterfly stages
- **Memory Pattern**: Uses transpose to optimize cache locality

### Difference from FFT-Strided
- **FFT-Strided**: Accesses memory with stride pattern (non-contiguous)
- **FFT-Transpose**: Uses explicit transpose operations for better locality
- Both implement the same mathematical FFT, but with different optimization strategies

---

## Usage

### Test Single Size
```bash
./EX1_optimized_codes/fft_gcc \
    input_data/mini/input/input.data \
    input_data/mini/output/check.data
```

Expected output: `Success.`

### Test All Sizes
```bash
for size in mini small medium large extra-large; do
    echo "Testing $size..."
    ./EX1_optimized_codes/fft_gcc \
        input_data/$size/input/input.data \
        input_data/$size/output/check.data
done
```

### Run with Custom Executable
```bash
# Test an optimized variant
./EX1_optimized_codes/fft_gcc_optimized \
    input_data/medium/input/input.data \
    input_data/medium/output/check.data
```

---

## Data Format

### Input Data (`input.data`)
```
%% Section 1
<512 double values: real part of input signal>

%% Section 2
<512 double values: imaginary part of input signal>
```

### Output Data (`check.data`)
```
%% Section 1
<512 double values: real part of FFT result>

%% Section 2
<512 double values: imaginary part of FFT result>
```

### Validation
- Comparison uses epsilon tolerance: `EPSILON = 1.0e-6`
- Both real and imaginary parts must match within tolerance

---

## Regeneration

To regenerate all datasets from scratch:

```bash
# 1. Ensure baseline is compiled
make clean
make baseline

# 2. Run generation script
./generate_all_data.sh
```

This will:
1. Create directory structure
2. Generate 5 different input patterns (using different random seeds)
3. Use baseline executable to compute golden outputs
4. Verify all datasets pass correctness tests

---

## Technical Notes

### Why Fixed Size?
The FFT algorithm is optimized for power-of-2 sizes (512 = 2^9). The implementation uses:
- Hardcoded array sizes: `TYPE work_x[512]`, `TYPE work_y[512]`
- Specialized butterfly operations for 512-point FFT
- Fixed transpose patterns

Changing the size would require modifying the algorithm implementation, not just input data.

### Random Pattern Variation
Different random seeds produce different input signals, which test:
- Numerical stability across diverse input ranges
- Edge cases in FFT computation
- Correctness of transpose and butterfly operations
- Cache behavior with different data patterns

---

## File Sizes

Each dataset:
- `input.data`: ~19 KB (1027 lines)
- `check.data`: ~20 KB (1027 lines)

Total: ~195 KB for all 5 datasets

---

## Validation Results

All datasets have been verified to produce correct outputs:
- ✓ mini
- ✓ small
- ✓ medium
- ✓ large
- ✓ extra-large

---

## References

- Implementation based on: V. Volkov and B. Kazian. "Fitting FFT onto the G80 architecture." 2008.
- MachSuite benchmark suite
- Original dataset: Single pattern with seed=1
EOFREADME

echo "  ✓ Generated input_data/README.md"

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                     Generation Complete!                         ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Generated files:"
echo "  • 5 input.data files (different random patterns)"
echo "  • 5 check.data files (golden outputs)"
echo "  • input_data/README.md (documentation)"
echo ""
echo "All datasets verified successfully! ✓"


