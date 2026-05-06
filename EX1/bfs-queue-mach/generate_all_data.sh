#!/bin/bash
# Generate all 5 sizes of BFS-Queue test data with different graph patterns
# All graphs have the same size (256 nodes, 4096 edges) but different topologies

set -e

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BENCH_DIR"

# Compile the generator with different random seeds
echo "=== Generating BFS-Queue test data with different graph patterns ==="

# Function to generate graph data with specific seed
generate_graph() {
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

// R-MAT parameters (*100)
#ifndef RMAT_A
#define RMAT_A 57
#endif
#ifndef RMAT_B
#define RMAT_B 19
#endif
#ifndef RMAT_C
#define RMAT_C 19
#endif
#ifndef RMAT_D
#define RMAT_D 5
#endif

#ifndef SEED_VALUE
#define SEED_VALUE 1
#endif

#include "bfs.h"

int main(int argc, char **argv)
{
  struct bench_args_t data;
  int fd;
  node_index_t adjmat[N_NODES][N_NODES];
  node_index_t r,c,s,temp;
  edge_index_t e;
  int scale;
  long int rint;
  struct prng_rand_t state;

  // Generate dense R-MAT matrix
  memset(adjmat, 0, N_NODES*N_NODES*sizeof(node_index_t));
  prng_srand(SEED_VALUE, &state);

  e = 0;
  while( e<N_EDGES/2 ) {
    r = 0;
    c = 0;
    // Pick a random edge according to R-MAT parameters
    for( scale=SCALE; scale>0; scale-- ) {
      rint = prng_rand(&state)%100;
      if( rint>=(RMAT_A+RMAT_B) )
        r += 1<<(scale-1);
      if( (rint>=RMAT_A && rint<RMAT_A+RMAT_B) || (rint>=RMAT_A+RMAT_B+RMAT_C) )
        c += 1<<(scale-1);
    }
    if( adjmat[r][c]==0 && r!=c ) {
      adjmat[r][c]=1;
      adjmat[c][r]=1;
      ++e;
    }
  }

  // Shuffle matrix
  for( s=0; s<N_NODES; s++ ) {
    rint = prng_rand(&state)%N_NODES;
    for( r=0; r<N_NODES; r++ ) {
      for( c=0; c<N_NODES; c++ ) {
        temp = adjmat[r][c];
        adjmat[r][c] = adjmat[rint][c];
        adjmat[rint][c] = temp;
      }
    }
    for( c=0; c<N_NODES; c++ ) {
      for( r=0; r<N_NODES; r++ ) {
        temp = adjmat[r][c];
        adjmat[r][c] = adjmat[r][rint];
        adjmat[r][rint] = temp;
      }
    }
  }

  // Scan rows for edge list lengths
  e = 0;
  for( r=0; r<N_NODES; r++ ) {
    data.nodes[r].edge_begin = 0;
    data.nodes[r].edge_end = 0;
    for( c=0; c<N_NODES; c++ ) {
      if( adjmat[r][c] ) {
        ++data.nodes[r].edge_end;
        data.edges[e].dst = c;
        ++e;
      }
    }
  }

  for( r=1; r<N_NODES; r++ ) {
    data.nodes[r].edge_begin = data.nodes[r-1].edge_end;
    data.nodes[r].edge_end += data.nodes[r-1].edge_end;
  }

  // Pick starting node
  do {
    rint = prng_rand(&state)%N_NODES;
  } while( (data.nodes[rint].edge_end-data.nodes[rint].edge_begin)<2 );
  data.starting_node = rint;

  // Write to file
  fd = open("temp_input.data", O_WRONLY|O_CREAT|O_TRUNC, S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH);
  assert( fd>0 && "Couldn't open output data file" );
  data_to_input(fd, &data);
  close(fd);

  return 0;
}
EOF

    # Compile with specific parameters
    cc -O3 -Wall -Wno-unused-label -I../../common_MachSuite \
        -DSEED_VALUE=$seed \
        -o generate_temp \
        generate_temp.c bfs.c local_support.c ../../common_MachSuite/support.c 2>/dev/null
    
    # Generate data
    ./generate_temp
    mv temp_input.data "input_data/$size/input/input.data"
    
    echo "  ✓ Generated input_data/$size/input/input.data"
}

# Generate different patterns
generate_graph "mini"        1    "Standard scale-free (original seed)"
generate_graph "small"       42   "Alternative scale-free topology"
generate_graph "medium"      123  "Different random seed"
generate_graph "large"       456  "Another random topology"
generate_graph "extra-large" 789  "Yet another random topology"

# Clean up
rm -f generate_temp generate_temp.c

echo ""
echo "=== Compiling baseline program ==="
make clean > /dev/null 2>&1
make all

if [ ! -f "EX1_optimized_codes/bfs_gcc" ]; then
    echo "ERROR: Baseline binary not found!"
    exit 1
fi

echo ""
echo "=== Generating golden outputs (check.data) ==="

for size in mini small medium large extra-large; do
    echo "[$size] Running baseline to generate check.data..."
    
    # Run baseline with input, redirect output to temporary location
    ./EX1_optimized_codes/bfs_gcc \
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
    if ./EX1_optimized_codes/bfs_gcc \
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

