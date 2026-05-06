# Benchmark Suites

HPC-Bench draws 92 workloads from seven well-established benchmark suites, covering 17 computational motifs. Below is the catalog with original references.

| Suite | Count | Domain | Reference |
|---|---|---|---|
| **PolyBench** | 30 | Dense linear algebra, stencils, dynamic programming | Pouchet, *PolyBench/C: The Polyhedral Benchmark Suite*, 2012. https://www.cs.colostate.edu/~pouchet/software/polybench/ |
| **Rodinia** | 18 | HPC C/C++ benchmarks (BFS, CFD, hotspot, etc.) | Che et al., *Rodinia: A benchmark suite for heterogeneous computing*, IISWC 2009. https://github.com/yuhc/gpu-rodinia |
| **Parboil** | 8 | GPU-heritage benchmarks adapted for CPU | Stratton et al., *Parboil: A Revised Benchmark Suite for Scientific and Commercial Throughput Computing*, IMPACT TR 2012. http://impact.crhc.illinois.edu/parboil/parboil.aspx |
| **MachSuite** | ~15 | Algorithm-specific C kernels (FFT, GEMM, BFS, MD, …) | Reagen et al., *MachSuite: Benchmarks for Accelerator Design and Customized Architectures*, IISWC 2014. https://github.com/breagen/MachSuite |
| **MiBench** | ~7 | Embedded benchmarks (dijkstra, qsort, patricia, susan, …) | Guthaus et al., *MiBench: A Free, Commercially Representative Embedded Benchmark Suite*, IISWC 2001. http://vhosts.eecs.umich.edu/mibench/ |
| **PARSEC** | 2 (canneal, cholesky) | Parallel workloads | Bienia et al., *The PARSEC Benchmark Suite: Characterization and Architectural Implications*, PACT 2008. https://parsec.cs.princeton.edu/ |
| **CODEE** | 5 | Small pedagogical kernels (matmul, coulomb, pi, …) | https://www.codee.com/ |

## Computational motifs

Each benchmark is annotated with a "computational motif" label. The 17 motifs (after Berkeley's *13 Dwarves* taxonomy, extended for HPC):

* Dense linear algebra
* Sparse linear algebra
* Stencil computations
* Spectral methods
* Graph traversal
* Dynamic programming
* Backtrack / branch-and-bound
* N-body methods
* Map-reduce
* Combinational logic
* Finite-state machines
* Regular expressions / string matching
* Image / signal processing
* Sorting
* Reductions
* Histogram
* Particle filters

The exact motif labels per benchmark are recorded in `scripts/benchmark_catalog.py` (`motif` field).

## Difficulty levels

Each benchmark is also classified into a difficulty tier `d1`–`d4` based on:
* Lines of code (LOC) of the kernel
* Number of functions
* Number of branches
* Loop nest depth

Computed by `scripts/analysis/difficulty.py`.

## Licensing notes

The bundled benchmark sources retain the licenses of their respective upstream suites. Do **not** assume the MIT license on this repository covers them — see the per-suite `COPYING` / `README` files in `EX{1,2,3}/<bench>/` for the upstream license.
