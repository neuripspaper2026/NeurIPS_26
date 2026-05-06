#!/bin/bash

# Generate all 5 datasets for viterbi-mach benchmark
# This benchmark has FIXED size, so we generate different HMM models with different seeds

set -e

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║   VITERBI-MACH 数据生成脚本                                       ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Benchmark: Viterbi Algorithm (Hidden Markov Model)"
echo "HMM Size: 64 states, 140 observations, 64 tokens (FIXED)"
echo "Strategy: Generate different HMM models with varying seeds"
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
#include <math.h>
#include "viterbi.h"
#include "../../common_MachSuite/support.h"

// SEED_VALUE will be defined via -D flag during compilation

int main(int argc, char **argv) {
  struct bench_args_t data;
  int fd;
  state_t s0, s1, s[N_OBS];
  tok_t o;
  step_t t;
  double P[N_STATES];
  double Q[N_TOKENS];
  struct prng_rand_t state;
  prob_t r;

  // Fill data structure with dynamic seed
  prng_srand(SEED_VALUE, &state);

  // Generate a random transition matrix P(S1|S0)
  // Invariant: SUM_S1 P(S1|S0) = 1
  for(s0=0; s0<N_STATES; s0++) {
    // Generate random weights
    double sum = 0;
    for(s1=0; s1<N_STATES; s1++) {
      P[s1] = ((double)prng_rand(&state)/(double)PRNG_RAND_MAX);
      if(s1==s0) P[s1] = N_STATES; // self-transitions are much more likely
      sum += P[s1];
    }
    // Normalize and convert to -log domain
    for(s1=0; s1<N_STATES; s1++) {
      data.transition[s0*N_STATES+s1] = -1*logf(P[s1]/sum);
    }
  }

  // Generate a random emission matrix P(O|S)
  // Invariant: SUM_O P(O|S) = 1
  for(s0=0; s0<N_STATES; s0++) {
    // Generate random weights
    double sum = 0;
    for(o=0; o<N_TOKENS; o++) {
      Q[o] = ((double)prng_rand(&state)/(double)PRNG_RAND_MAX);
      if( o==s0 ) Q[o] = N_TOKENS; // one token is much more likely
      sum += Q[o];
    }
    // Normalize and convert to -log domain
    for(o=0; o<N_TOKENS; o++) {
      data.emission[s0*N_TOKENS+o] = -1*logf(Q[o]/sum);
    }
  }

  // Generate a random starting distribution P(S_0)
  // Invariant: SUM P(S_0) = 1
  {
    // Generate random weights
    double sum = 0;
    for(s0=0; s0<N_STATES; s0++) {
      P[s0] = ((double)prng_rand(&state)/(double)PRNG_RAND_MAX);
      sum += P[s0];
    }
    // Normalize and convert to -log domain
    for(s0=0; s0<N_STATES; s0++) {
      data.init[s0] = -1*logf(P[s0]/sum);
    }
  }

  // To get observations, just run the HMM forwards N_OBS steps
  // Nondeterministic sampling uses the inverse transform method

  // Sample s_0 from init
  r = ((double)prng_rand(&state)/(double)PRNG_RAND_MAX);
  s[0]=0; do{r-=expf(-data.init[s[0]]);} while(r>0&&(++s[0]));

  // Sample o_0 from emission
  r = ((double)prng_rand(&state)/(double)PRNG_RAND_MAX);
  o=0; do{ r-=expf(-data.emission[s[0]*N_TOKENS+o]); }while(r>0 && ++o);
  data.obs[0] = o;

  for(t=1; t<N_OBS; t++) {
    // Sample s_t from transition
    r = ((double)prng_rand(&state)/(double)PRNG_RAND_MAX);
    s[t]=0; do{r-=expf(-data.transition[s[t-1]*N_STATES+s[t]]);} while(r>0&&++s[t]);

    // Sample o_t from emission
    r = ((double)prng_rand(&state)/(double)PRNG_RAND_MAX);
    o=0; do{ r-=expf(-data.emission[s[t]*N_TOKENS+o]); }while(r>0 && ++o);
    data.obs[t] = o;
  }

  // Open and write
  fd = open("temp_input.data", O_WRONLY|O_CREAT|O_TRUNC, S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH);
  assert( fd>0 && "Couldn't open input data file" );
  data_to_input(fd, (void*)(&data));
  close(fd);

  return 0;
}
EOF_GENERATE

    # Compile with specific seed
    cc -O3 -Wall -Wno-unused-label -I../../common_MachSuite \
        -DSEED_VALUE=$seed \
        -o generate_temp \
        generate_temp.c viterbi.c local_support.c ../../common_MachSuite/support.c -lm 2>/dev/null
    
    # Generate data
    ./generate_temp
    mv temp_input.data "input_data/$size/input/input.data"
    
    echo "  ✓ Generated input_data/$size/input/input.data"
}

# Generate all 5 datasets
echo "════════════════════════════════════════════════════════════════════"
echo "Phase 1: Generating input.data files"
echo "════════════════════════════════════════════════════════════════════"

generate_data "mini"        1     "Standard HMM model (original seed)"
generate_data "small"       42    "Alternative HMM model"
generate_data "medium"      123   "Different HMM parameters"
generate_data "large"       456   "Another HMM configuration"
generate_data "extra-large" 789   "Fifth HMM model"

# Clean up temporary files
rm -f generate_temp generate_temp.c temp_input.data

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "Phase 2: Generating check.data (Golden Output)"
echo "════════════════════════════════════════════════════════════════════"

# Ensure baseline is compiled
if [ ! -f "EX1_optimized_codes/viterbi_gcc" ]; then
    echo "Building baseline executable..."
    make clean && make baseline
fi

# Generate check.data for all sizes
for size in mini small medium large extra-large; do
    echo ""
    echo "[$size] Generating check.data..."
    
    # Run baseline to generate output
    ./EX1_optimized_codes/viterbi_gcc \
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
    
    if ./EX1_optimized_codes/viterbi_gcc \
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
echo "  • HMM: 64 states, 140 observations, 64 tokens (FIXED)"
echo "  • Algorithm: Viterbi (dynamic programming)"
echo "  • Variation: Different HMM models (random seeds)"
echo "  • Files per size: input.data + check.data"
echo ""
echo "Usage:"
echo "  ./EX1_optimized_codes/viterbi_gcc input_data/<size>/input/input.data input_data/<size>/output/check.data"
echo ""

