#!/bin/bash

# Generate all 5 datasets for spmv-crs-mach benchmark
# This benchmark has FIXED matrix structure (IEEE 494 bus), so we generate different input vectors

set -e

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║     SPMV-CRS-MACH 数据生成脚本                                    ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Benchmark: Sparse Matrix-Vector Multiplication (CRS format)"
echo "Matrix: IEEE 494 bus (494×494, 1666 nonzeros, FIXED)"
echo "Strategy: Generate different input vectors with varying seeds"
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

#include "spmv.h"
#include "../../common_MachSuite/support.h"

#define ROW 0
#define COL 1

// SEED_VALUE will be defined via -D flag during compilation

// Row first, then column comparison
int compar(const void *v_lhs, const void *v_rhs)
{
  int row_lhs = ((int *)v_lhs)[ROW];
  int row_rhs = ((int *)v_rhs)[ROW];
  int col_lhs = ((int *)v_lhs)[COL];
  int col_rhs = ((int *)v_rhs)[COL];

  if( row_lhs==row_rhs ) {
    if( col_lhs==col_rhs )
      return 0;
    else if( col_lhs<col_rhs )
      return -1;
    else
      return 1;
  } else if( row_lhs<row_rhs ) {
    return -1;
  } else {
    return 1;
  }
}

int main(int argc, char **argv)
{
  struct bench_args_t data;
  struct stat file_info;
  char *current, *next, *buffer;
  int status, i, fd, nbytes;
  int coords[NNZ][2]; // row, col
  struct prng_rand_t state;

  // Load matrix file
  fd = open("494_bus_full.mtx", O_RDONLY);
  assert( fd>=0 && "couldn't open matrix" );
  status = fstat( fd, &file_info );
  assert( status==0 && "couldn't get filesize of matrix" );
  buffer = malloc(file_info.st_size+1);
  buffer[file_info.st_size]=(char)0;
  nbytes = 0;
  do {
    status = read(fd, buffer, file_info.st_size-nbytes);
    assert(status>=0 && "Couldn't read from matrix file");
    nbytes+=status;
  } while( nbytes<file_info.st_size );
  close(fd);

  // Parse matrix file
  current = buffer;
  next = strchr(current, '\n');// skip first two lines
  *next = (char)0;
  current = next+1;
  for(i=0; i<NNZ; i++) {
    next = strchr(current, '\n');
    *next = (char)0;
    current = next+1;
    status = sscanf(current, "%d %d %lf", &coords[i][ROW], &coords[i][COL], &data.val[i]);
    assert(status==3 && "Parse error in matrix file");
  }

  // Sort by row
  qsort(coords, NNZ, 2*sizeof(int), &compar);

  // Fill data structure
  for(i=0; i<NNZ; i++)
    data.cols[i] = coords[i][COL]-1;
  memset(data.rowDelimiters, 0, (N+1)*sizeof(int));
  for(i=0; i<NNZ; i++)
    data.rowDelimiters[coords[i][ROW]-1+1] += 1; // count
    // (-1 because matrix is 1-indexed, +1 because it's counting cells before it)
  for(i=1; i<N+1; i++)
    data.rowDelimiters[i] += data.rowDelimiters[i-1]; // scan

  // Set vector with dynamic seed
  prng_srand(SEED_VALUE, &state); // Use dynamic seed
  for( i=0; i<N; i++ )
    data.vec[i] = ((double)prng_rand(&state))/((double)PRNG_RAND_MAX);

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
        generate_temp.c spmv.c local_support.c ../../common_MachSuite/support.c 2>/dev/null
    
    # Generate data
    ./generate_temp
    mv temp_input.data "input_data/$size/input/input.data"
    
    echo "  ✓ Generated input_data/$size/input/input.data"
}

# Generate all 5 datasets
echo "════════════════════════════════════════════════════════════════════"
echo "Phase 1: Generating input.data files"
echo "════════════════════════════════════════════════════════════════════"

generate_data "mini"        1     "Standard input vector (original seed)"
generate_data "small"       42    "Alternative input vector"
generate_data "medium"      123   "Different input pattern"
generate_data "large"       456   "Another input vector"
generate_data "extra-large" 789   "Fifth input vector"

# Clean up temporary files
rm -f generate_temp generate_temp.c temp_input.data

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "Phase 2: Generating check.data (Golden Output)"
echo "════════════════════════════════════════════════════════════════════"

# Ensure baseline is compiled
if [ ! -f "EX1_optimized_codes/spmv_gcc" ]; then
    echo "Building baseline executable..."
    make clean && make baseline
fi

# Generate check.data for all sizes
for size in mini small medium large extra-large; do
    echo ""
    echo "[$size] Generating check.data..."
    
    # Run baseline to generate output
    ./EX1_optimized_codes/spmv_gcc \
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
    
    if ./EX1_optimized_codes/spmv_gcc \
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
echo "  • Matrix: IEEE 494 bus (494×494, 1666 nonzeros, FIXED)"
echo "  • Variation: Different input vectors (random seeds)"
echo "  • Files per size: input.data + check.data"
echo ""
echo "Usage:"
echo "  ./EX1_optimized_codes/spmv_gcc input_data/<size>/input/input.data input_data/<size>/output/check.data"
echo ""

