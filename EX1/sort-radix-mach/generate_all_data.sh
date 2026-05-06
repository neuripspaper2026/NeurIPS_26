#!/bin/bash

# Generate all 5 datasets for sort-radix-mach benchmark
# This benchmark has FIXED size (2048 elements), so we generate different patterns

set -e

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║     SORT-RADIX-MACH 数据生成脚本                                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Benchmark: Radix Sort"
echo "Array Size: 2048 int32_t elements (FIXED)"
echo "Strategy: Generate different random patterns with varying seeds"
echo ""

# Function to generate data with a specific seed
generate_data() {
    local size=$1
    local seed=$2
    local description=$3
    echo ""
    echo "[$size] Seed: $seed - $description"
    
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

#include "sort.h"
#include "../../common_MachSuite/support.h" // Include support.h for data_to_input

int main(int argc, char **argv)
{
  struct bench_args_t data;
  int i, fd;
  struct prng_rand_t state;

  // Fill data structure
  prng_srand(SEED_VALUE, &state); // Use dynamic seed
  for(i=0; i<SIZE; i++)
    data.a[i] = prng_rand(&state) & TYPE_MAX;

  // Open and write
  fd = open("temp_input.data", O_WRONLY|O_CREAT|O_TRUNC, S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH);
  assert( fd>0 && "Couldn't open input data file" );
  data_to_input(fd, (void *)(&data));
  close(fd);

  return 0;
}
EOF

    # Compile with specific parameters
    cc -O3 -Wall -Wno-unused-label -I../../common_MachSuite \
        -DSEED_VALUE=$seed \
        -o generate_temp \
        generate_temp.c sort.c local_support.c ../../common_MachSuite/support.c 2>/dev/null
    
    # Generate data
    ./generate_temp
    mv temp_input.data "input_data/$size/input/input.data"
    
    echo "  ✓ Generated input_data/$size/input/input.data"
}

# Generate all 5 datasets
echo "════════════════════════════════════════════════════════════════════"
echo "Phase 1: Generating input.data files"
echo "════════════════════════════════════════════════════════════════════"

generate_data "mini"        1     "Standard random distribution (original seed)"
generate_data "small"       42    "Alternative random distribution"
generate_data "medium"      123   "Different random pattern"
generate_data "large"       456   "Another random distribution"
generate_data "extra-large" 789   "Fifth random distribution"

# Clean up temporary files
rm -f generate_temp generate_temp.c temp_input.data

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "Phase 2: Generating check.data (Golden Output)"
echo "════════════════════════════════════════════════════════════════════"

# Ensure baseline is compiled
if [ ! -f "EX1_optimized_codes/sort_gcc" ]; then
    echo "Building baseline executable..."
    make clean && make baseline
fi

# Generate check.data for all sizes
for size in mini small medium large extra-large; do
    echo ""
    echo "[$size] Generating check.data..."
    
    # Run baseline to generate output
    ./EX1_optimized_codes/sort_gcc \
        input_data/$size/input/input.data \
        input_data/$size/input/input.data 2>/dev/null || true
    
    # Copy to check.data
    cp EX5_correctness/output_gcc.data input_data/$size/output/check.data
    
    echo "  ✓ Generated input_data/$size/output/check.data"
done

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "Phase 3: Verification"
echo "════════════════════════════════════════════════════════════════════"

# Verify all datasets
all_passed=true
for size in mini small medium large extra-large; do
    echo -n "[$size] Verifying... "
    
    if ./EX1_optimized_codes/sort_gcc \
        input_data/$size/input/input.data \
        input_data/$size/output/check.data 2>&1 | grep -q "Success"; then
        echo "✓ Success"
    else
        echo "✗ Failed"
        all_passed=false
    fi
done

echo ""
if [ "$all_passed" = true ]; then
    echo "════════════════════════════════════════════════════════════════════"
    echo "✅ All datasets generated and verified successfully!"
    echo "════════════════════════════════════════════════════════════════════"
else
    echo "════════════════════════════════════════════════════════════════════"
    echo "❌ Some datasets failed verification. Please check."
    echo "════════════════════════════════════════════════════════════════════"
    exit 1
fi

echo ""
echo "📊 Summary:"
echo "  • Total sizes: 5 (mini, small, medium, large, extra-large)"
echo "  • Array size: 2048 int32_t elements (FIXED)"
echo "  • Pattern: Different random distributions"
echo "  • Files per size: input.data + check.data"
echo ""
echo "Usage:"
echo "  ./EX1_optimized_codes/sort_gcc input_data/<size>/input/input.data input_data/<size>/output/check.data"
echo ""

