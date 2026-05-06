#!/bin/bash

set -e

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║       NW-MACH Dataset Generation (Fixed Scale)                   ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Benchmark: Needleman-Wunsch Sequence Alignment"
echo "Type: Fixed Scale (128x128 sequences)"
echo "Strategy: Generate different DNA sequence pairs"
echo ""

# Clean up any previous files
rm -f generate_temp generate_temp.c temp_input.data

# Function to generate Needleman-Wunsch data
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
#define ALEN 128
#define BLEN 128

struct bench_args_t {
  char seqA[ALEN];
  char seqB[BLEN];
  char alignedA[ALEN+BLEN];
  char alignedB[ALEN+BLEN];
  int M[(ALEN+1)*(BLEN+1)];
  char ptr[(ALEN+1)*(BLEN+1)];
};

#include "../../common_MachSuite/support.h"

// DNA nucleotides
const char nucleotides[] = {'a', 'c', 'g', 't'};

void generate_dna_sequence(char *seq, int len, struct prng_rand_t *state) {
    for (int i = 0; i < len; i++) {
        seq[i] = nucleotides[prng_rand(state) % 4];
    }
}

int main(int argc, char **argv)
{
  struct bench_args_t data;
  struct prng_rand_t state;
  int fd;

  // Initialize random seed
  prng_srand(SEED_VALUE, &state);

  // Generate random DNA sequences
  generate_dna_sequence(data.seqA, ALEN, &state);
  generate_dna_sequence(data.seqB, BLEN, &state);

  // Open and write
  fd = open("temp_input.data", O_WRONLY|O_CREAT|O_TRUNC, S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH);
  assert( fd>0 && "Couldn't open input data file" );
  
  // Write section 1: sequence A
  write_section_header(fd);
  write_string(fd, data.seqA, ALEN);
  
  // Write section 2: sequence B
  write_section_header(fd);
  write_string(fd, data.seqB, BLEN);
  
  // Write section 3 (empty terminator)
  write_section_header(fd);
  
  close(fd);

  return 0;
}
EOF
    
    # Compile with specific parameters
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
echo "  Generating Input Data (Different DNA Sequence Pairs)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

generate_data "mini"        1    "Standard DNA sequences (original seed)"
generate_data "small"       42   "Alternative sequence pair"
generate_data "medium"      123  "Different sequence pattern"
generate_data "large"       456  "Another sequence pair"
generate_data "extra-large" 789  "Fifth sequence pair"

# Clean up
rm -f generate_temp generate_temp.c temp_input.data

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Generating Golden Outputs (check.data)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Compiling baseline..."
make clean >/dev/null 2>&1 || true
make baseline 2>&1 | grep -E "(gcc|clang|nw_)" || true

if [ ! -f "EX1_optimized_codes/nw_gcc" ]; then
    echo "❌ Error: Baseline executable not found!"
    exit 1
fi

echo ""
for size in mini small medium large extra-large; do
    echo -n "[$size] Generating check.data... "
    
    # Run baseline to generate output
    ./EX1_optimized_codes/nw_gcc \
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
    if timeout 30 ./EX1_optimized_codes/nw_gcc \
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
echo "  Benchmark: Needleman-Wunsch (Sequence Alignment)"
echo "  Scale: Fixed (128x128 character sequences)"
echo "  Datasets: 5 (different DNA sequence pairs)"
echo "  Files generated: 10 (5 input + 5 check)"
echo ""
echo "✅ Generation complete!"
echo ""

