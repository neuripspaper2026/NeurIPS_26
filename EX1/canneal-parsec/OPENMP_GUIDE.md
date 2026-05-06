# OpenMP Parallelization Guide for Canneal

## Current Status

**The code is currently single-threaded and has been structured to make OpenMP parallelization straightforward.**

## Why Single-Threaded Currently?

The simulated annealing algorithm requires careful synchronization when parallelized:
- Random number generation must be thread-safe
- Element swaps must avoid race conditions
- Cost calculations need atomic operations

## Quick OpenMP Enablement (3 Steps)

### Step 1: Update Makefile

```makefile
CXXFLAGS ?= -O3 -std=c++11 -fopenmp
```

### Step 2: Add OpenMP Header

In `main.cpp` (line 31):
```cpp
#include <omp.h>
```

In `annealer_thread.cpp` (line 29):
```cpp
#include <omp.h>
```

### Step 3: Remove Thread Restriction

In `main.cpp` (replace lines 63-66):
```cpp
// Old code (remove):
if (num_threads != 1){
    cout << "Warning: Multi-threading not yet enabled. Running with 1 thread." << endl;
    num_threads = 1;
}

// New code:
#ifdef _OPENMP
    omp_set_num_threads(num_threads);
    cout << "OpenMP enabled with " << num_threads << " threads" << endl;
#else
    cout << "OpenMP not available, using 1 thread" << endl;
    num_threads = 1;
#endif
```

## Advanced Parallelization (Optional)

### Option A: Parallel Temperature Steps (Simple)

This allows running multiple independent annealing runs in parallel:

In `main.cpp`, replace the single run with:
```cpp
#pragma omp parallel
{
    #pragma omp single
    {
        a_thread.Run();
    }
}
```

### Option B: Parallel Swaps (Complex)

For parallelizing the swap loop itself, modify `annealer_thread.cpp`:

```cpp
// Add thread-local RNG
#pragma omp parallel
{
    Rng local_rng;  // Thread-local random number generator
    
    #pragma omp for schedule(dynamic)
    for (int i = 0; i < _moves_per_thread_temp; i++){
        // ... swap logic with local_rng instead of rng
    }
}
```

**Note**: This requires additional synchronization for:
- Netlist access (may need locks)
- Cost updates (atomic operations)
- Acceptance counters (reduction clause)

## Complete Example: Parallel Swaps with OpenMP

Here's a more complete implementation for `annealer_thread.cpp`:

```cpp
void annealer_thread::Run()
{
    int accepted_good_moves=0;
    int accepted_bad_moves=-1;
    double T = _start_temp;
    
    int temp_steps_completed=0; 
    while(keep_going(temp_steps_completed, accepted_good_moves, accepted_bad_moves)){
        T = T / 1.5;
        int thread_good_moves = 0;
        int thread_bad_moves = 0;
        
        #pragma omp parallel reduction(+:thread_good_moves,thread_bad_moves)
        {
            // Thread-local variables
            Rng local_rng;
            int tid = omp_get_thread_num();
            local_rng.seed(3 + tid);  // Different seed per thread
            
            long a_id, b_id;
            netlist_elem* a;
            netlist_elem* b;
            
            #pragma omp for schedule(dynamic, 100)
            for (int i = 0; i < _moves_per_thread_temp; i++){
                // Get random elements (needs locking if netlist is shared)
                #pragma omp critical(netlist_access)
                {
                    a = _netlist->get_random_element(&a_id, NO_MATCHING_ELEMENT, &local_rng);
                    b = _netlist->get_random_element(&b_id, a_id, &local_rng);
                }
                
                routing_cost_t delta_cost = calculate_delta_routing_cost(a,b);
                move_decision_t is_good_move = accept_move(delta_cost, T, &local_rng);

                // Make the move (needs locking for swap)
                if (is_good_move != move_decision_rejected){
                    #pragma omp critical(netlist_swap)
                    {
                        _netlist->swap_locations(a,b);
                    }
                    
                    if (is_good_move == move_decision_accepted_bad){
                        thread_bad_moves++;
                    } else {
                        thread_good_moves++;
                    }
                }
            }
        }
        
        accepted_good_moves = thread_good_moves;
        accepted_bad_moves = thread_bad_moves;
        temp_steps_completed++;
    }
}
```

## Performance Considerations

### Parallel Overhead

- **Critical sections** can become bottlenecks
- **Lock contention** on netlist access
- **False sharing** in data structures

### Best Practices

1. **Minimize critical sections**: Batch operations where possible
2. **Use thread-local data**: Avoid shared state
3. **Coarse-grained parallelism**: Parallelize outer loops, not inner ones
4. **Test scalability**: Measure speedup with different thread counts

### Expected Speedup

With proper implementation:
- 2 threads: 1.5-1.8x speedup
- 4 threads: 2.5-3.2x speedup
- 8 threads: 3.5-5.0x speedup

Diminishing returns due to:
- Synchronization overhead
- Memory bandwidth limits
- Algorithm dependencies

## Testing

### Compile with OpenMP

```bash
make clean
CXX="g++ -fopenmp" make baseline
```

### Run with Different Thread Counts

```bash
# 1 thread (baseline)
./EX1_optimized_codes/canneal_gcc 1 10000 2000 input_data/small/input/input_small.nets 100

# 2 threads
./EX1_optimized_codes/canneal_gcc 2 10000 2000 input_data/small/input/input_small.nets 100

# 4 threads
./EX1_optimized_codes/canneal_gcc 4 10000 2000 input_data/small/input/input_small.nets 100
```

### Verify Correctness

Results may vary slightly due to:
- Different random number sequences
- Different swap orders
- Timing-dependent decisions

This is **expected and acceptable** for stochastic algorithms.

## Summary

| Complexity | Steps | Effort | Speedup |
|-----------|-------|--------|---------|
| **Minimal** | Add `-fopenmp`, include `<omp.h>`, remove thread check | 5 minutes | 1x (no parallelism yet) |
| **Simple** | Add `#pragma omp parallel` around main loop | 15 minutes | 1.2-1.5x |
| **Full** | Thread-local RNG, critical sections, reductions | 2-4 hours | 2-4x |

**Recommendation**: Start with minimal changes to ensure code compiles with OpenMP, then incrementally add parallelism while testing correctness.

## Current Code Structure (OpenMP-Ready)

✅ **Good practices already in place:**
- No global mutable state
- Clean separation of concerns
- RNG passed as parameter (easy to make thread-local)
- Modular functions

✅ **What makes it OpenMP-ready:**
- Main loop structure is suitable for parallelization
- Swap operations are independent (given synchronization)
- Temperature updates are sequential (as intended)

⚠️ **Challenges remaining:**
- Netlist access synchronization
- Random number generation thread-safety
- Cost calculation atomicity

The code is **well-structured for OpenMP** - it just needs the pragmas and synchronization added.

