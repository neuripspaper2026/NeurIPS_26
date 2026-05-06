#!/bin/bash

# Generate all 5 datasets for stencil-3d-mach benchmark
# This benchmark has FIXED size, so we generate different input patterns with different seeds

set -e

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║   STENCIL-3D-MACH 数据生成脚本                                    ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Benchmark: 3D Stencil Computation (7-point stencil)"
echo "Grid Size: 32×32×16 = 16,384 elements (FIXED)"
echo "Coefficients: C[0]=6, C[1]=-1 (3D Laplacian)"
echo "Strategy: Generate different input patterns with varying seeds"
echo ""

# Function to generate data with a specific seed
generate_data() {
    local size=$1
    local seed=$2
    local description=$3
    echo ""
    echo "[$size] Seed: $seed - $description"
    
    # Create a temporary generate.c with the specific seed
    cat <<'EOF_GENERATE' > generate_temp.c
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <assert.h>

#include "stencil.h"
#include "../../common_MachSuite/support.h"

// SEED_VALUE will be defined via -D flag during compilation

int main(int argc, char **argv)
{
  struct bench_args_t data;
  int i, fd;
  struct prng_rand_t state;

  // 3D discrete Laplacian coefficients (fixed)
  data.C[0] = 6;
  data.C[1] = -1;
  
  // Random matrix with dynamic seed
  prng_srand(SEED_VALUE, &state); // Use dynamic seed
  for(i=0; i<SIZE; i++)
    data.orig[i] = prng_rand(&state)%(MAX-MIN) + MIN;

  // Open and write
  fd = open("temp_input.data", O_WRONLY|O_CREAT|O_TRUNC, S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH);
  assert( fd>0 && "Couldn't open input data file" );
  data_to_input(fd, (void *)(&data));
  close(fd);

  return 0;
}
EOF_GENERATE

    # Compile with specific seed
    cc -O3 -Wall -Wno-unused-label -I../../common_MachSuite \
        -DSEED_VALUE=$seed \
        -o generate_temp \
        generate_temp.c stencil.c local_support.c ../../common_MachSuite/support.c 2>/dev/null
    
    # Generate data
    ./generate_temp
    mv temp_input.data "input_data/$size/input/input.data"
    
    echo "  ✓ Generated input_data/$size/input/input.data"
}

# Generate all 5 datasets
echo "════════════════════════════════════════════════════════════════════"
echo "Phase 1: Generating input.data files"
echo "════════════════════════════════════════════════════════════════════"

generate_data "mini"        1     "Standard random pattern (original seed)"
generate_data "small"       42    "Alternative random pattern"
generate_data "medium"      123   "Different random distribution"
generate_data "large"       456   "Another random pattern"
generate_data "extra-large" 789   "Fifth random pattern"

# Clean up temporary files
rm -f generate_temp generate_temp.c temp_input.data

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "Phase 2: Generating check.data (Golden Output)"
echo "════════════════════════════════════════════════════════════════════"

# Ensure baseline is compiled
if [ ! -f "EX1_optimized_codes/stencil_gcc" ]; then
    echo "Building baseline executable..."
    make clean && make baseline
fi

# Generate check.data for all sizes
for size in mini small medium large extra-large; do
    echo ""
    echo "[$size] Generating check.data..."
    
    # Run baseline to generate output
    ./EX1_optimized_codes/stencil_gcc \
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
    
    if ./EX1_optimized_codes/stencil_gcc \
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
echo "  • Grid: 32×32×16 (16,384 elements, FIXED)"
echo "  • Stencil: 7-point (center + 6 face neighbors)"
echo "  • Variation: Different input patterns (random seeds)"
echo "  • Files per size: input.data + check.data"
echo ""
echo "Usage:"
echo "  ./EX1_optimized_codes/stencil_gcc input_data/<size>/input/input.data input_data/<size>/output/check.data"
echo ""

