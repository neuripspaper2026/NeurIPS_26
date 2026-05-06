#!/bin/bash
# Generate all 5 sizes of FFT-Strided test data with different random inputs
# All datasets have the same size (FFT_SIZE=1024) but different input patterns

set -e

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BENCH_DIR"

echo "=== Generating FFT-Strided test data with different input patterns ==="

# Function to generate FFT data with specific seed
generate_data() {
    local size=$1
    local seed=$2
    local pattern=$3
    
    echo ""
    echo "[$size] Pattern: $pattern (seed=$seed)"
    
    # Create modified generator
    cat > generate_temp.c << 'EOF'
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <assert.h>
#include <math.h>

#include "fft.h"

#ifndef SEED_VALUE
#define SEED_VALUE 1
#endif

int main(int argc, char **argv)
{
  struct bench_args_t data;
  int i, n, fd;
  double typed;
  struct prng_rand_t state;

  // Fill data structure with specific seed
  prng_srand(SEED_VALUE, &state);
  for(i=0; i<FFT_SIZE; i++){
    data.real[i] = ((double)prng_rand(&state))/((double)PRNG_RAND_MAX);
    data.img[i] = ((double)prng_rand(&state))/((double)PRNG_RAND_MAX);
  }

  // Pre-calc twiddles (same for all datasets)
  for(n=0; n<(FFT_SIZE>>1); n++){
      typed = (double)(twoPI*n/FFT_SIZE);
      data.real_twid[n] = cos(typed);
      data.img_twid[n] = (-1.0)*sin(typed);
  }

  // Write to file
  fd = open("temp_input.data", O_WRONLY|O_CREAT|O_TRUNC, S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH);
  assert( fd>0 && "Couldn't open output data file" );
  data_to_input(fd, (void *)(&data));
  close(fd);

  return 0;
}
EOF

    # Compile with specific seed
    cc -O3 -Wall -Wno-unused-label -I../../common_MachSuite \
        -DSEED_VALUE=$seed \
        -o generate_temp \
        generate_temp.c fft.c local_support.c ../../common_MachSuite/support.c -lm 2>/dev/null
    
    # Generate data
    ./generate_temp
    mv temp_input.data "input_data/$size/input/input.data"
    
    echo "  ✓ Generated input_data/$size/input/input.data"
}

# Generate different patterns with different seeds
generate_data "mini"        1       "Standard random pattern (original seed)"
generate_data "small"       42      "Alternative random pattern"
generate_data "medium"      123     "Different random distribution"
generate_data "large"       456     "Another random pattern"
generate_data "extra-large" 789     "Yet another random pattern"

# Clean up
rm -f generate_temp generate_temp.c

echo ""
echo "=== Compiling baseline program ==="
make clean > /dev/null 2>&1
make all

if [ ! -f "EX1_optimized_codes/fft_gcc" ]; then
    echo "ERROR: Baseline binary not found!"
    exit 1
fi

echo ""
echo "=== Generating golden outputs (check.data) ==="

for size in mini small medium large extra-large; do
    echo "[$size] Running baseline to generate check.data..."
    
    # Run baseline with input, redirect output to temporary location
    ./EX1_optimized_codes/fft_gcc \
        "input_data/$size/input/input.data" \
        "input_data/$size/input/input.data" 2>/dev/null || true
    
    # Copy the generated output as golden reference
    if [ -f "EX5_correctness/output_gcc.data" ]; then
        cp "EX5_correctness/output_gcc.data" "input_data/$size/output/check.data"
        echo "  ✓ Generated input_data/$size/output/check.data"
    else
        echo "  ✗ ERROR: Failed to generate output for $size"
        exit 1
    fi
done

echo ""
echo "=== Verifying all datasets ==="

all_passed=true
for size in mini small medium large extra-large; do
    echo -n "[$size] "
    if ./EX1_optimized_codes/fft_gcc \
        "input_data/$size/input/input.data" \
        "input_data/$size/output/check.data" 2>&1 | grep -q "Success"; then
        echo "✓ PASS"
    else
        echo "✗ FAIL"
        all_passed=false
    fi
done

echo ""
if [ "$all_passed" = true ]; then
    echo "=== All datasets generated and verified successfully! ==="
else
    echo "=== ERROR: Some datasets failed verification ==="
    exit 1
fi

# Show file sizes
echo ""
echo "=== Dataset sizes ==="
for size in mini small medium large extra-large; do
    input_size=$(wc -l < "input_data/$size/input/input.data")
    output_size=$(wc -l < "input_data/$size/output/check.data")
    echo "[$size] input: $input_size lines, output: $output_size lines"
done

echo ""
echo "=== Generation complete! ==="

