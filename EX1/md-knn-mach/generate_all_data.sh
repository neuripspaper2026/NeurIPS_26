#!/bin/bash

set -e

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║       MD-KNN-MACH Dataset Generation (Fixed Scale)               ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Benchmark: Molecular Dynamics with K-Nearest Neighbors"
echo "Type: Fixed Scale (256 atoms, K=16 neighbors)"
echo "Strategy: Generate different random atom distributions"
echo ""

# Clean up any previous files
rm -f generate_temp generate_temp.c temp_input.data

# Function to generate molecular dynamics data with KNN
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

#define domainEdge 20.0
// LJ1=4e(s^12), LJ2=4e(s^6) -> s=1.049
#define van_der_Waals_thresh (1.049*1.049)

static inline TYPE dist_sq(TYPE x1, TYPE y1, TYPE z1, TYPE x2, TYPE y2, TYPE z2) {
  TYPE dx, dy, dz;
  dx=x2-x1;
  dy=y2-y1;
  dz=z2-z1;
  return dx*dx + dy*dy + dz*dz;
}

typedef struct {
  int index;
  TYPE dist_sq;
} neighbor_t;

int neighbor_compar(const void *v_lhs, const void *v_rhs) {
  neighbor_t lhs = *((neighbor_t *)v_lhs);
  neighbor_t rhs = *((neighbor_t *)v_rhs);
  return lhs.dist_sq==rhs.dist_sq ? 0 : ( lhs.dist_sq<rhs.dist_sq ? -1 : 1 );
}

int main(int argc, char **argv)
{
  struct bench_args_t data;
  int i, j, reject, fd;
  neighbor_t neighbor_list[nAtoms];
  TYPE x, y, z;
  const TYPE infinity = (domainEdge*domainEdge*3.)*1000;//(max length)^2 * 1000
  struct prng_rand_t state;

  // Create random positions in the box [0,domainEdge]^3
  prng_srand(SEED_VALUE, &state); // Use dynamic seed
  i=0;
  while( i<nAtoms ) {
    // Generate a new point
    x = domainEdge*(((TYPE)prng_rand(&state))/((TYPE)PRNG_RAND_MAX));
    y = domainEdge*(((TYPE)prng_rand(&state))/((TYPE)PRNG_RAND_MAX));
    z = domainEdge*(((TYPE)prng_rand(&state))/((TYPE)PRNG_RAND_MAX));
    // Assure that it's not directly on top of another atom
    reject = 0;
    for( j=0; j<i; j++ ) {
      if( dist_sq(x,y,z, data.position_x[j], data.position_y[j], data.position_z[j])<van_der_Waals_thresh ) {
        reject=1;
        break;
      }
    }
    if(!reject) {
      data.position_x[i] = x;
      data.position_y[i] = y;
      data.position_z[i] = z;
      ++i;
    }
  }

  // Compute k-nearest neighbors
  memset(data.NL, 0, nAtoms*maxNeighbors*sizeof(int32_t));
  for( i=0; i<nAtoms; i++ ) {
    for( j=0; j<nAtoms; j++ ) {
      neighbor_list[j].index = j;
      if( i==j )
        neighbor_list[j].dist_sq = infinity;
      else
        neighbor_list[j].dist_sq = dist_sq(data.position_x[i], data.position_y[i], data.position_z[i], data.position_x[j], data.position_y[j], data.position_z[j]);
    }
    qsort(neighbor_list, nAtoms, sizeof(neighbor_t), neighbor_compar);
    for( j=0; j<maxNeighbors; j++ ) {
      data.NL[i*maxNeighbors +j] = neighbor_list[j].index;
    }
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
echo "  Benchmark: MD-KNN (Molecular Dynamics with K-Nearest Neighbors)"
echo "  Scale: Fixed (256 atoms, K=16 neighbors)"
echo "  Datasets: 5 (different random distributions)"
echo "  Files generated: 10 (5 input + 5 check)"
echo ""
echo "✅ Generation complete!"
echo ""

