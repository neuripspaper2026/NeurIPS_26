#!/bin/bash

set -e

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║       MD-GRID-MACH Dataset Generation (Fixed Scale)              ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Benchmark: Molecular Dynamics with Grid (N-body simulation)"
echo "Type: Fixed Scale (256 atoms, 4x4x4 grid, domain=20.0)"
echo "Strategy: Generate different random atom distributions"
echo ""

# Clean up any previous files
rm -f generate_temp generate_temp.c temp_input.data

# Function to generate molecular dynamics data
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

#include "md.h"
#include "../../common_MachSuite/support.h"

// LJ1=4e(s^12), LJ2=4e(s^6) -> s=1.049
#define van_der_Waals_thresh (1.049*1.049)

static inline TYPE dist_sq( dvector_t p, dvector_t q ) {
  TYPE dx, dy, dz;
  dx = p.x - q.x;
  dy = p.y - q.y;
  dz = p.z - q.z;
  return dx*dx + dy*dy + dz*dz;
}

int main(int argc, char **argv)
{
  struct bench_args_t data;
  int fd, reject;
  int32_t i;
  dvector_t p, q;
  ivector_t b;
  dvector_t points[nAtoms];
  int idx, entry;
  struct prng_rand_t state;

  // Create random positions in the box [0,domainEdge]^3
  prng_srand(SEED_VALUE, &state); // Use dynamic seed
  i=0;
  while( i<nAtoms ) {
    // Generate a new point
    p.x = domainEdge*(((TYPE)prng_rand(&state))/((TYPE)PRNG_RAND_MAX));
    p.y = domainEdge*(((TYPE)prng_rand(&state))/((TYPE)PRNG_RAND_MAX));
    p.z = domainEdge*(((TYPE)prng_rand(&state))/((TYPE)PRNG_RAND_MAX));
    // Assure that it's not directly on top of another atom
    reject = 0;
    for( idx=0; idx<nAtoms; idx++ ) {
      q = points[idx];
      if( dist_sq(p,q)<van_der_Waals_thresh ) {
        reject=1;
        break;
      }
    }
    if(!reject) {
      points[i] = p;
      ++i;
    }
  }

  // Insert points into blocks
  memset(data.n_points, 0, blockSide*blockSide*blockSide*sizeof(int32_t));
  memset(data.position, 0, 3*blockSide*blockSide*blockSide*densityFactor*sizeof(TYPE));
  for( idx=0; idx<nAtoms; idx++ ) {
    b.x = (int32_t) (points[idx].x / blockEdge);
    b.y = (int32_t) (points[idx].y / blockEdge);
    b.z = (int32_t) (points[idx].z / blockEdge);
    entry = data.n_points[b.x][b.y][b.z];
    data.position[b.x][b.y][b.z][entry].x = points[idx].x;
    data.position[b.x][b.y][b.z][entry].y = points[idx].y;
    data.position[b.x][b.y][b.z][entry].z = points[idx].z;
    ++data.n_points[b.x][b.y][b.z];
    assert(data.n_points[b.x][b.y][b.z]<densityFactor && "block overflow");
  }

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
        generate_temp.c md.c local_support.c ../../common_MachSuite/support.c 2>/dev/null
    
    # Generate data
    ./generate_temp
    mv temp_input.data "input_data/$size/input/input.data"
    
    echo "  ✓ Generated input_data/$size/input/input.data"
}

# Generate all datasets with different random seeds
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Generating Input Data (Different Random Atom Distributions)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

generate_data "mini"        1    "Standard random distribution (original seed)"
generate_data "small"       42   "Alternative random distribution"
generate_data "medium"      123  "Different random pattern"
generate_data "large"       456  "Another random distribution"
generate_data "extra-large" 789  "Fifth random distribution"

# Clean up
rm -f generate_temp generate_temp.c temp_input.data

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Generating Golden Outputs (check.data)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Compiling baseline..."
make clean >/dev/null 2>&1 || true
make baseline 2>&1 | grep -E "(gcc|clang|md_)" || true

if [ ! -f "EX1_optimized_codes/md_gcc" ]; then
    echo "❌ Error: Baseline executable not found!"
    exit 1
fi

echo ""
for size in mini small medium large extra-large; do
    echo -n "[$size] Generating check.data... "
    
    # Run baseline to generate output
    ./EX1_optimized_codes/md_gcc \
        input_data/$size/input/input.data \
        input_data/$size/input/input.data 2>/dev/null || true
    
    # Copy output to check.data
    if [ -f "EX5_correctness/output_gcc.data" ]; then
        cp EX5_correctness/output_gcc.data input_data/$size/output/check.data
        echo "✓"
    else
        echo "❌ Failed"
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
    echo -n "[$size] Verifying... "
    if timeout 30 ./EX1_optimized_codes/md_gcc \
        input_data/$size/input/input.data \
        input_data/$size/output/check.data 2>&1 | grep -q "Success"; then
        echo "✓ Pass"
    else
        echo "❌ Failed"
        all_passed=false
    fi
done

echo ""
if [ "$all_passed" = true ]; then
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                    ✅ All Datasets Valid!                        ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
else
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                  ❌ Some Datasets Failed!                        ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    exit 1
fi

echo ""
echo "📊 Dataset Summary"
echo "─────────────────────────────────────────────────────────────────"
echo "  Benchmark: MD-Grid (Molecular Dynamics)"
echo "  Scale: Fixed (256 atoms, 4x4x4 grid)"
echo "  Datasets: 5 (different random distributions)"
echo "  Files generated: 10 (5 input + 5 check)"
echo ""
echo "✅ Generation complete!"
echo ""

