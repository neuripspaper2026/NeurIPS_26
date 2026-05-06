# BFS Two-Kernel Architecture Explanation

## Why BFS Has Two Kernels

Unlike most benchmarks that have a single computational kernel, BFS uses **two cooperating kernels** that execute alternately. This is a fundamental design pattern for parallel graph algorithms.

## The Two Kernels

### Kernel 1: `Kernel()` in kernel.cu

**Purpose:** Exploration Phase - Find neighbors to visit next

```cuda
__global__ void Kernel(Node *g_graph_nodes, int *g_graph_edges,
                       bool *g_graph_mask, bool *g_updating_graph_mask,
                       bool *g_graph_visited, int *g_cost, int no_of_nodes)
```

**What it does:**
1. Processes all nodes in current BFS layer (`g_graph_mask[tid] == true`)
2. For each active node:
   - Traverses all its edges
   - For unvisited neighbors:
     - Updates distance: `g_cost[neighbor] = g_cost[node] + 1`
     - Marks for next layer: `g_updating_graph_mask[neighbor] = true`
   - Deactivates current node: `g_graph_mask[tid] = false`

**Input:** Current layer nodes
**Output:** Marked nodes for next layer

### Kernel 2: `Kernel2()` in kernel2.cu

**Purpose:** Update Phase - Officially add nodes to next layer

```cuda
__global__ void Kernel2(bool *g_graph_mask, bool *g_updating_graph_mask,
                        bool *g_graph_visited, bool *g_over, int no_of_nodes)
```

**What it does:**
1. Processes all marked nodes (`g_updating_graph_mask[tid] == true`)
2. For each marked node:
   - Activate for next iteration: `g_graph_mask[tid] = true`
   - Mark as visited: `g_graph_visited[tid] = true`
   - Signal continuation: `*g_over = true`
   - Clear update flag: `g_updating_graph_mask[tid] = false`

**Input:** Marked nodes from Kernel
**Output:** Updated masks for next iteration

## Execution Flow

```
Initialize: Set source node in g_graph_mask

do {
    stop = false
    
    ┌─────────────────────────────────────┐
    │ Kernel: Explore current layer       │
    │  - Read g_graph_mask                │
    │  - Write g_updating_graph_mask      │
    │  - Update g_cost                    │
    └─────────────────────────────────────┘
    
    ┌─────────────────────────────────────┐
    │ Kernel2: Update for next layer      │
    │  - Read g_updating_graph_mask       │
    │  - Write g_graph_mask               │
    │  - Write g_graph_visited            │
    │  - Set g_over flag                  │
    └─────────────────────────────────────┘
    
    Copy g_over back to host
    
} while (stop == true);  // Continue if new nodes found
```

## Why Two Phases Are Necessary

### Problem with Single-Kernel Approach

If we tried to do everything in one kernel:

```cuda
// WRONG - Data race!
if (g_graph_mask[tid]) {
    g_graph_mask[tid] = false;  // Deactivate
    for (neighbor : neighbors) {
        if (!g_graph_visited[neighbor]) {
            g_graph_mask[neighbor] = true;  // Activate neighbor - RACE!
        }
    }
}
```

**Issues:**
- ❌ Thread A deactivates itself, Thread B might activate it immediately
- ❌ Nodes from different BFS levels mix in same iteration
- ❌ BFS correctness violated (distance guarantees broken)
- ❌ Non-deterministic behavior

### Solution with Two-Phase Approach

**Phase 1 (Kernel):** Only WRITE to `g_updating_graph_mask`
```cuda
if (g_graph_mask[tid]) {
    g_graph_mask[tid] = false;
    for (neighbor : neighbors) {
        g_updating_graph_mask[neighbor] = true;  // No race - only marking
    }
}
```

**Phase 2 (Kernel2):** Only READ from `g_updating_graph_mask`
```cuda
if (g_updating_graph_mask[tid]) {
    g_graph_mask[tid] = true;  // Safe - all Phase 1 threads done
    g_updating_graph_mask[tid] = false;
}
```

**Benefits:**
- ✅ No data races (reads and writes separated)
- ✅ BFS levels maintained correctly
- ✅ Deterministic behavior
- ✅ Algorithm correctness guaranteed

## Memory Access Pattern

```
Iteration N:
  Kernel:  READ  g_graph_mask[layer N]
           WRITE g_updating_graph_mask[layer N+1]
  
  Kernel2: READ  g_updating_graph_mask[layer N+1]
           WRITE g_graph_mask[layer N+1]

Iteration N+1:
  Kernel:  READ  g_graph_mask[layer N+1]  ← Safe!
           WRITE g_updating_graph_mask[layer N+2]
  ...
```

## Timing Implications

Because the two kernels work as a **unit operation**, we time them together:

```c
do {
    start_timer();
    
    Kernel<<<>>>();   // These two together
    Kernel2<<<>>>();  // are ONE BFS iteration
    
    synchronize();
    stop_timer();
    
    total_kernel_time += iteration_time;
    
} while (more_nodes);
```

**KERNEL_TIME = Sum of all (Kernel + Kernel2) iterations**

## Variant Implications

When creating optimized variants, you MUST provide BOTH kernels:

```
EX3_optimized_codes/
├── kernel_gpt4_v1.cu    ← Optimized Kernel
├── kernel2_gpt4_v1.cu   ← Matching Kernel2
```

**Why?**
- The two kernels share data structures
- They must work together correctly
- Optimizations in Kernel may require changes in Kernel2
- Mixing baseline + variant would likely fail

## Academic Reference

This two-phase approach is from:

**Paper:** "Accelerating Large Graph Algorithms on the GPU using CUDA"
**Authors:** Pawan Harish, P. J. Narayanan
**Conference:** HiPC'07 (14th International Conference on High Performance Computing)
**Year:** 2007

The paper introduced this pattern for parallel BFS and it became a standard technique for GPU graph algorithms.

## Comparison with Other Patterns

| Pattern | Example | Kernel Count |
|---------|---------|--------------|
| Simple Computation | Matrix Multiply | 1 kernel |
| Multi-stage Pipeline | Backprop | 2+ kernels (sequential stages) |
| **Iterative Two-Phase** | **BFS** | **2 kernels (cooperating loop)** |
| Reduction | Sum | 1 kernel (possibly recursive) |

BFS is unique in that its two kernels are:
- Tightly coupled (share all data structures)
- Execute alternately (not sequentially)
- Repeat until convergence (not fixed iterations)
- Required for correctness (not just optimization)

## Summary

**Key Takeaways:**

1. BFS needs TWO kernels for algorithmic correctness
2. Kernel explores, Kernel2 updates
3. Two phases prevent data races and maintain BFS levels
4. Both kernels timed together as a unit
5. Variants must provide both kernel files
6. This is a standard pattern for parallel graph algorithms

**For Makefile/Build:**
- Detect `kernel_*.cu` AND `kernel2_*.cu` together
- Only build if BOTH exist
- Include path rewriting must handle BOTH files
