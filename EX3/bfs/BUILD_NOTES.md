# BFS Multi-Variant Build System

## Overview

BFS (Breadth-First Search) has been modified to support:
1. **Time measurement** for kernel execution and total runtime
2. **Multi-variant compilation** for different kernel implementations

## Special Characteristics

### Two-Kernel Architecture

BFS uses **two cooperating kernels** that execute alternately:

```
do {
    Kernel<<<>>>()   // Explore current layer
    Kernel2<<<>>>()  // Update for next layer
} while (not_done);
```

**Kernel (kernel.cu):**
- Processes nodes in current BFS layer (`g_graph_mask == true`)
- Traverses edges and marks unvisited neighbors
- Updates cost/distance values
- Marks nodes for next layer in `g_updating_graph_mask`

**Kernel2 (kernel2.cu):**
- Processes nodes marked by Kernel
- Officially adds them to next layer (`g_graph_mask = true`)
- Marks nodes as visited (`g_graph_visited = true`)
- Sets continuation flag if new nodes found

### Why Two Kernels?

**Purpose:** Avoid data races and maintain BFS level correctness

Without separation:
- ❌ Threads could see updates from same iteration
- ❌ BFS levels would mix (violating algorithm correctness)

With two-phase approach:
- ✅ Kernel only writes to `g_updating_graph_mask` (read-only for others)
- ✅ Kernel2 synchronizes updates to `g_graph_mask`
- ✅ Each iteration processes exactly one BFS level

## Time Measurement

### Timing Strategy

```c
double total_kernel_time = 0.0;

do {
    clock_gettime(CLOCK_MONOTONIC, &kernel_start);
    
    Kernel<<<>>>();
    Kernel2<<<>>>();
    cudaDeviceSynchronize();
    
    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    total_kernel_time += kernel_time_iter;
    
} while (stop);
```

**Key Points:**
- Both kernels timed together (they're a unit operation)
- Time accumulated across all BFS iterations
- `KERNEL_TIME` = sum of all iterations
- `TOTAL_TIME` = entire program (I/O + computation)

### Output Format

```
KERNEL_TIME: 0.123456789
TOTAL_TIME: 1.234567890
```

Written to:
- `stderr` by default
- File specified by `TIMING_LOG_FILE` environment variable

## Multi-Variant Makefile

### File Naming Convention

```
EX3_optimized_codes/
├── kernel.cu              # Baseline kernel 1
├── kernel2.cu             # Baseline kernel 2
├── kernel_gpt4_v1.cu      # Variant kernel 1
├── kernel2_gpt4_v1.cu     # Variant kernel 2 (must match!)
├── kernel_claude_v2.cu    # Another variant kernel 1
└── kernel2_claude_v2.cu   # Another variant kernel 2 (must match!)
```

**CRITICAL:** Both `kernel_<id>.cu` and `kernel2_<id>.cu` must exist for a variant to build!

### How It Works

1. **Detection Phase:**
   ```makefile
   KERNEL_1_IDS = gpt4_v1 claude_v2
   KERNEL_2_IDS = gpt4_v1 claude_v2
   VALID_VARIANTS = $(filter KERNEL_1_IDS, KERNEL_2_IDS)  # Both must exist
   ```

2. **Dynamic Include Rewriting:**
   ```makefile
   # For baseline:
   sed 's|#include "EX3_optimized_codes/kernel.cu"|#include "../EX3_optimized_codes/kernel.cu"|g' \
       bfs.cu > .build/bfs_baseline.cu
   
   # For variant gpt4_v1:
   sed 's|#include "EX3_optimized_codes/kernel.cu"|#include "../EX3_optimized_codes/kernel_gpt4_v1.cu"|g' \
       bfs.cu > .build/bfs_gpt4_v1.cu
   ```

3. **Compilation:**
   ```makefile
   EX3_optimized_codes/bfs: .build/bfs_baseline.cu
       nvcc -o $@ $<
   
   EX3_optimized_codes/bfs_gpt4_v1: .build/bfs_gpt4_v1.cu
       nvcc -o $@ $<
   ```

### Build Targets

```bash
make                # Build all variants
make clean          # Remove executables (safe - preserves .cu files)
make list-variants  # Show detected variants
make help           # Show help
```

### Build Output

```
==========================================
Building: EX3_optimized_codes/bfs (baseline)
==========================================
✓ Successfully built EX3_optimized_codes/bfs

==========================================
Building: EX3_optimized_codes/bfs_gpt4_v1 (variant gpt4_v1)
==========================================
✓ Successfully built EX3_optimized_codes/bfs_gpt4_v1

==========================================
Build Summary:
==========================================
✓ EX3_optimized_codes/bfs - SUCCESS
✓ EX3_optimized_codes/bfs_gpt4_v1 - SUCCESS
==========================================
```

## File Structure

```
EX3/bfs/
├── bfs.cu                      # Main program (modified includes)
├── Makefile                    # Multi-variant build system
├── kernel.cu                   # Original kernel 1 (kept for reference)
├── kernel2.cu                  # Original kernel 2 (kept for reference)
├── EX3_optimized_codes/
│   ├── kernel.cu               # Baseline kernel 1
│   ├── kernel2.cu              # Baseline kernel 2
│   ├── kernel_*.cu             # Variant kernels 1
│   ├── kernel2_*.cu            # Variant kernels 2
│   ├── bfs                     # Baseline executable
│   └── bfs_*                   # Variant executables
└── .build/                     # Temporary files (auto-generated)
    ├── bfs_baseline.cu
    ├── bfs_gpt4_v1.cu
    └── ...
```

## Important Notes

### Clean Target Safety

The `make clean` rule uses filtering to avoid deleting source files:

```makefile
cd $(OUTPUT_DIR) && rm -f bfs $(shell ls bfs_* | grep -v '\.cu$$' | grep -v '\.h$$')
```

This ensures:
- ✅ Deletes: `bfs`, `bfs_gpt4_v1` (executables)
- ✅ Preserves: `kernel.cu`, `kernel_gpt4_v1.cu` (source files)

### Error Tolerance

The `-@` prefix on compilation commands allows the build to continue even if one variant fails:

```makefile
-@$(NVCC) ... -o $@ $<
```

### Include Path Management

Original `bfs.cu` includes:
```c
#include "EX3_optimized_codes/kernel.cu"
#include "EX3_optimized_codes/kernel2.cu"
```

Temporary `.build/bfs_*.cu` files have rewritten includes:
```c
#include "../EX3_optimized_codes/kernel_gpt4_v1.cu"
#include "../EX3_optimized_codes/kernel2_gpt4_v1.cu"
```

This allows each build to link with the correct kernel pair.

## Verification

After building, verify executables exist:

```bash
ls -la EX3_optimized_codes/bfs*
```

Expected output:
```
-rwxr-xr-x bfs
-rwxr-xr-x bfs_gpt4_v1
-rwxr-xr-x bfs_claude_v2
-rw-r--r-- kernel.cu
-rw-r--r-- kernel2.cu
-rw-r--r-- kernel_gpt4_v1.cu
-rw-r--r-- kernel2_gpt4_v1.cu
```

Run a variant:
```bash
cd EX3/bfs
./EX3_optimized_codes/bfs_gpt4_v1 input_data/mini/input/graph1M.txt -o output.txt
```

Check timing output:
```bash
TIMING_LOG_FILE=time.log ./EX3_optimized_codes/bfs_gpt4_v1 input_data/mini/input/graph1M.txt -o output.txt
cat time.log
```

## Troubleshooting

### Missing Kernel Pair

**Error:** Variant not building even though files exist

**Solution:** Ensure BOTH `kernel_<id>.cu` AND `kernel2_<id>.cu` exist

```bash
make list-variants  # Check detected pairs
```

### Include Path Errors

**Error:** `fatal error: kernel.cu: No such file or directory`

**Cause:** Temporary file generation issue

**Solution:** Check `.build/` directory and sed commands in Makefile

### Compilation Warnings

**Warning:** `__CUDA_DEPRECATED`

**Solution:** Ensure `cudaDeviceSynchronize()` is used (not `cudaThreadSynchronize()`)

## Algorithm Reference

BFS implementation based on:
- **Paper:** "Accelerating Large Graph Algorithms on the GPU using CUDA"
- **Conference:** HiPC'07
- **Authors:** Pawan Harish et al.
- **Institution:** International Institute of Information Technology - Hyderabad
