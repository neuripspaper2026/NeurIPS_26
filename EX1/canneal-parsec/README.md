# Canneal Benchmark - Simulated Annealing for Chip Routing

## Overview

Canneal uses simulated annealing to optimize the routing cost of a chip design netlist. The algorithm attempts to minimize wire length by swapping element locations on a 2D grid.

## Quick Start

```bash
# Build
make

# Run with mini configuration
./EX1_optimized_codes/canneal_gcc 1 1000 2000 input_data/mini/input/input_mini.nets 50
```

## Available Sizes

| Size          | Elements | Grid    | Swaps     | Steps | Est. Runtime |
|--------------|----------|---------|-----------|-------|--------------|
| mini         | 100      | 15x15   | 1,000     | 50    | ~1s          |
| small        | 500      | 30x30   | 50,000    | 150   | ~2s          |
| medium       | 2,000    | 60x60   | 200,000   | 400   | ~5s          |
| large        | 5,000    | 100x100 | 500,000   | 700   | ~15s         |
| extra-large  | 10,000   | 150x150 | 1,000,000 | 1000  | ~30s         |

## Usage

```bash
./canneal NTHREADS NSWAPS TEMP NETLIST [NSTEPS]
```

### Parameters

- **NTHREADS**: Number of threads (currently must be 1)
- **NSWAPS**: Number of swaps to attempt per temperature step
- **TEMP**: Starting temperature for annealing
- **NETLIST**: Path to netlist file (.nets format)
- **NSTEPS**: (Optional) Number of temperature steps before termination

### Example Commands

```bash
# Mini
./EX1_optimized_codes/canneal_gcc 1 1000 2000 input_data/mini/input/input_mini.nets 50

# Small  
./EX1_optimized_codes/canneal_gcc 1 50000 2500 input_data/small/input/input_small.nets 150

# Medium
./EX1_optimized_codes/canneal_gcc 1 200000 3000 input_data/medium/input/input_medium.nets 400

# Large
./EX1_optimized_codes/canneal_gcc 1 500000 3500 input_data/large/input/input_large.nets 700

# Extra-large
./EX1_optimized_codes/canneal_gcc 1 1000000 4000 input_data/extra-large/input/input_extra-large.nets 1000
```

## Generating Input Data

Use the provided Python script to generate netlist files:

```bash
# Generate all sizes
python3 generate.py

# Generate specific size
python3 generate.py --size mini

# Generate custom netlist
python3 generate.py --custom --elements 1000 --grid-width 40 --grid-height 40 --output my_netlist.nets
```

## Netlist Format

The netlist file format:
```
NUM_ELEMENTS  GRID_WIDTH  GRID_HEIGHT
element_name  type  connection1  connection2  ...  END
...
```

Example:
```
10	5	5
a	2	g	a	h	d	h	END
b	2	c	j	g	j	a	END
...
```

## Output

The program outputs:
- Initial routing cost
- Final routing cost after optimization
- Improvement (absolute and percentage)

Example output:
```
Initial routing cost: 3500

========================================
Routing Optimization Results:
========================================
Initial cost: 3500
Final cost:   2232
Improvement:  1268 (36.23%)
========================================
```

## OpenMP Parallelization (Future)

Currently, the code runs in single-threaded mode. To enable OpenMP parallelization:

### 1. Modify Makefile

Add `-fopenmp` to CXXFLAGS:
```makefile
CXXFLAGS ?= -O3 -std=c++11 -fopenmp
```

### 2. Include OpenMP Header

In `main.cpp` and `annealer_thread.cpp`:
```cpp
#include <omp.h>
```

### 3. Remove Thread Check

In `main.cpp`, remove or modify the single-thread enforcement (lines 63-66).

### 4. Add Parallel Directives

In `annealer_thread.cpp`, add pragmas around the swap loop:

```cpp
#pragma omp parallel for schedule(dynamic)
for (int i = 0; i < _moves_per_thread_temp; i++){
    // ... swap logic
}
```

**Note**: Parallel version will require careful handling of:
- Random number generation (thread-local RNG)
- Netlist access synchronization
- Cost calculation atomicity

## Files

- **`main.cpp`** - Entry point, parameter parsing
- **`annealer_thread.cpp`** - Simulated annealing algorithm
- **`netlist.cpp`** - Netlist data structure
- **`generate.py`** - Input data generator
- **`Makefile`** - Build configuration

## Algorithm

Simulated annealing optimization:
1. Start with random element placement
2. At each temperature step:
   - Attempt N random swaps
   - Accept swaps that improve cost
   - Accept some bad swaps based on Boltzmann probability
3. Decrease temperature gradually
4. Stop when converged or max steps reached

Temperature update: `T = T / 1.5`

Acceptance probability: `exp(-ΔCost / T)`

## Performance Tips

- Larger grid sizes allow more optimization potential
- More swaps per temperature improve solution quality
- Higher starting temperature allows more exploration
- More temperature steps provide better convergence

## Correctness Verification

### ⚠️ Important: Canneal uses Random Optimization

Canneal uses **simulated annealing** (随机优化算法), so:
- ✅ **Normal**: Final cost will be slightly different each run
- ✅ **Expected**: Different variants may produce different (but similar) results
- ❌ **Wrong**: Final cost > Initial cost (optimization failed)

### Python Tools for Verification

#### 1. Verify Single Output

```bash
# Run and save output
./EX1_optimized_codes/canneal_gcc 1 1000 2000 input_data/mini/input/input_mini.nets 50 > output.txt

# Verify correctness
python3 verify_correctness.py output.txt
```

#### 2. Compare All Variants

```bash
# Test all variants with mini size
python3 compare_variants.py mini

# Results saved in correctness_test/mini/
```

### Correctness Criteria

A variant is **correct** if:
1. ✅ Program runs without crash
2. ✅ Final cost < Initial cost (optimization works)
3. ✅ Improvement > 0
4. ✅ Output format is valid

A variant is **acceptable** if:
- Final cost differs from baseline by **< 10%** (due to randomness)

See **`CORRECTNESS_README.md`** for detailed usage and examples.

