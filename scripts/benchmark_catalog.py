"""Centralize benchmark metadata for shared use across multiple classes."""

from __future__ import annotations

from typing import Any, Dict

BenchmarkMeta = Dict[str, Dict[str, Any]]

BENCHMARK_POLYBENCH: BenchmarkMeta = {
    "correlation": {"target_file": "correlation.c", "threshold": 0.002, "start_line": 67, "end_line": 126},
    "covariance": {"target_file": "covariance.c", "threshold": 0.002, "start_line": 65, "end_line": 98},
    "gemm": {"target_file": "gemm.c", "threshold": 0.002, "start_line": 73, "end_line": 93},
    "gemver": {"target_file": "gemver.c", "threshold": 0.002, "start_line": 85, "end_line": 119},
    "gesummv": {"target_file": "gesummv.c", "threshold": 0.002, "start_line": 72, "end_line": 98},
    "symm": {"target_file": "symm.c", "threshold": 0.002, "start_line": 75, "end_line": 98},
    "syr2k": {"target_file": "syr2k.c", "threshold": 0.002, "start_line": 73, "end_line": 94},
    "syrk": {"target_file": "syrk.c", "threshold": 0.002, "start_line": 69, "end_line": 88},
    "trmm": {"target_file": "trmm.c", "threshold": 0.002, "start_line": 71, "end_line": 88},
    "2mm": {"target_file": "2mm.c", "threshold": 0.002, "start_line": 77, "end_line": 107},
    "3mm": {"target_file": "3mm.c", "threshold": 0.002, "start_line": 73, "end_line": 112},
    "atax": {"target_file": "atax.c", "threshold": 0.002, "start_line": 66, "end_line": 88},
    "bicg": {"target_file": "bicg.c", "threshold": 0.0005, "start_line": 74, "end_line": 98},
    "doitgen": {"target_file": "doitgen.c", "threshold": 0.002, "start_line": 67, "end_line": 87},
    "mvt": {"target_file": "mvt.c", "threshold": 0.002, "start_line": 79, "end_line": 98},
    "cholesky": {"target_file": "cholesky.c", "threshold": 0.002, "start_line": 84, "end_line": 108},
    "durbin": {"target_file": "durbin.c", "threshold": 0.002, "start_line": 62, "end_line": 97},
    "gramschmidt": {"target_file": "gramschmidt.c", "threshold": 0.002, "start_line": 80, "end_line": 110},
    "lu": {"target_file": "lu.c", "threshold": 0.002, "start_line": 85, "end_line": 106},
    "ludcmp": {"target_file": "ludcmp.c", "threshold": 0.002, "start_line": 95, "end_line": 139},
    "trisolv": {"target_file": "trisolv.c", "threshold": 0.002, "start_line": 67, "end_line": 85},
    "deriche": {"target_file": "deriche.c", "threshold": 0.002, "start_line": 68, "end_line": 157},
    "floyd-warshall": {"target_file": "floyd-warshall.c", "threshold": 0.002, "start_line": 53, "end_line": 69},
    "nussinov": {"target_file": "nussinov.c", "threshold": 0.002, "start_line": 81, "end_line": 111},
    "adi": {"target_file": "adi.c", "threshold": 0.002, "start_line": 68, "end_line": 130},
    "fdtd-2d": {"target_file": "fdtd-2d.c", "threshold": 0.002, "start_line": 91, "end_line": 121},
    "heat-3d": {"target_file": "heat-3d.c", "threshold": 0.002, "start_line": 65, "end_line": 98},
    "jacobi-1d": {"target_file": "jacobi-1d.c", "threshold": 0.002, "start_line": 65, "end_line": 83},
    "jacobi-2d": {"target_file": "jacobi-2d.c", "threshold": 0.002, "start_line": 66, "end_line": 86},
    "seidel-2d": {"target_file": "seidel-2d.c", "threshold": 0.002, "start_line": 62, "end_line": 78},
}

BENCHMARK_RODINIA: BenchmarkMeta = {
    "streamcluster": {"target_file": "streamcluster.cpp", "threshold": 0.002, "start_line": 354, "end_line": 597},
    "srad_v1": {"target_file": "srad.c", "threshold": 0.002, "start_line": 1, "end_line": 343},
    "srad_v2": {"target_file": "srad.cpp", "threshold": 0.002, "start_line": 1, "end_line": 248},
    "pathfinder": {"target_file": "pathfinder.cpp", "threshold": 0.002, "start_line": 104, "end_line": 172},
    "particlefilter": {"target_file": "particle-filter.c", "threshold": 0.002, "start_line": 361, "end_line": 581},
    "nw": {"target_file": "needle.cpp", "threshold": 0.002, "start_line": 133, "end_line": 307},
    "myocyte": {"target_file": "master.c", "threshold": 0.002, "start_line": 1, "end_line": 85},
    "lud": {"target_file": "lud-omp.c", "threshold": 0.002, "start_line": 1, "end_line": 23},
    "leukocyte": {"target_file": "track-ellipse.c", "threshold": 0.002, "start_line": 4, "end_line": 205},
    "lavaMD": {"target_file": "kernel_cpu.c", "threshold": 0.002, "start_line": 1, "end_line": 140},
    "kmeans": {"target_file": "kmeans-clustering.c", "threshold": 0.002, "start_line": 1, "end_line": 119},
    "hotspot3D": {"target_file": "3D.c", "threshold": 0.002, "start_line": 127, "end_line": 173},
    "hotspot": {"target_file": "hotspot.cpp", "threshold": 0.002, "start_line": 47, "end_line": 157},
    "heartwall": {"target_file": "kernel.c", "threshold": 0.002, "start_line": 1, "end_line": 596},
    "cfd": {"target_file": "euler3d-cpu.cpp", "threshold": 0.002, "start_line": 194, "end_line": 403},
    "bfs": {"target_file": "bfs.cpp", "threshold": 0.002, "start_line": 1, "end_line": 190},
    "backprop": {"target_file": "backprop.c", "threshold": 0.002, "start_line": 220, "end_line": 256},
    "b+tree": {"target_file": ["kernel-cpu.c", "kernel-cpu-2.c"], "threshold": 0.002, "start_line": [1, 1], "end_line": [77, 101]},
    "bilateral": {"target_file": "bilateralFilter-cpu.cpp", "threshold": 0.002, "start_line": 95, "end_line": 160},
}

BENCHMARK_CODEE: BenchmarkMeta = {
    "matmul": {"target_file": "matmul.c", "threshold": 0.1, "start_line": 8, "end_line": 30},
    "coulomb": {"target_file": "coulomb.c", "threshold": 0.001, "start_line": 16, "end_line": 40},
    "pi": {"target_file": "pi.c", "threshold": 0.001, "start_line": 1, "end_line": 103},
    "atmux": {"target_file": "atmux.c", "threshold": 0.001, "start_line": 17, "end_line": 27},
    "haccmk": {"target_file": "Step10_orig.c", "threshold": 0.001, "start_line": 1, "end_line": 48},
}

BENCHMARK_PARBOIL: BenchmarkMeta = {
    "sad": {"target_file": "sad_cpu.c", "threshold": "TBD", "start_line": 1, "end_line": 202},
    "cutcp": {"target_file": "cutcpu.c", "threshold": "TBD", "start_line": 1, "end_line": 221},
    "histo": {"target_file": "main.c", "threshold": 0.001, "start_line": 1, "end_line": 142},
    "lbm": {"target_file": "lbm.c", "threshold": "TBD", "start_line": 1, "end_line": 626},
    "mri-gridding": {"target_file": "CPU_kernels.c", "threshold": "TBD", "start_line": 1, "end_line": 197},
    "mri-q": {"target_file": "computeQ.cc", "threshold": "TBD", "start_line": 1, "end_line": 65},
    "sgemm": {"target_file": "sgemm_kernel.cc", "threshold": 0.001, "start_line": 1, "end_line": 24},
    "spmv": {"target_file": "main.c", "threshold": "TBD", "start_line": 1, "end_line": 134},
    "stencil": {"target_file": "kernels.c", "threshold": "TBD", "start_line": 1, "end_line": 25},
    "tpacf": {"target_file": "model_compute_cpu.c", "threshold": 0, "start_line": 1, "end_line": 60},
}

BENCHMARK_MIBENCH: BenchmarkMeta = {
    "basicmath-mibench": {"target_file": "cubic.c", "threshold": "TBD", "start_line": 10, "end_line": 47},
    "bitcount-mibench": {"target_file": "bitcnts.c", "threshold": "TBD", "start_line": 120, "end_line": 134},
    "qsort-mibench": {"target_file": "qsort_large.c", "threshold": 0, "start_line": 1, "end_line": 114},
    "susan-s-mibench": {"target_file": "susan.c", "threshold": "TBD", "start_line": 391, "end_line": 529},
    "susan-e-mibench": {"target_file": "susan.c", "threshold": "TBD", "start_line": 783, "end_line": 1023},
    "susan-c-mibench": {"target_file": "susan.c", "threshold": "TBD", "start_line": 1184, "end_line": 1469},
    "dijkstra-mibench": {"target_file": "dijkstra_parameterized.c", "threshold": 0, "start_line": 103, "end_line": 153},
    "patricia-mibench": {"target_file": "patricia.c", "threshold": "TBD", "start_line": 63, "end_line": 175},
    "stringsearch-mibench": {"target_file": "pbmsrch_parameterized.c", "threshold": "TBD", "start_line": 26, "end_line": 57},
}

BENCHMARK_PARSEC: BenchmarkMeta = {
    "canneal-parsec": {"target_file": "annealer_thread.cpp", "threshold": "TBD", "start_line": 19, "end_line": 64},
}

BENCHMARK_MACHSUITE: BenchmarkMeta = {
    "aes": {"target_file": "aes.c", "threshold": "None", "start_line": 179, "end_line": 212},
    "bfs-bulk-mach": {"target_file": "bfs.c", "threshold": "None", "start_line": 1, "end_line": 49},
    "bfs-queue-mach": {"target_file": "bfs.c", "threshold": "None", "start_line": 1, "end_line": 56},
    "fft-strided-mach": {"target_file": "fft.c", "threshold": "None", "start_line": 1, "end_line": 44},
    "fft-transpose-mach": {"target_file": "fft.c", "threshold": "None", "start_line": 114, "end_line": 415},
    "gemm-blocked-mach": {"target_file": "gemm.c", "threshold": "None", "start_line": 1, "end_line": 36},
    "gemm-ncubed-mach": {"target_file": "gemm.c", "threshold": "None", "start_line": 1, "end_line": 33},
    "kmp-mach": {"target_file": "kmp.c", "threshold": "None", "start_line": 1, "end_line": 52},
    "md-grid-mach": {"target_file": "md.c", "threshold": "None", "start_line": 1, "end_line": 70},
    "md-knn-mach": {"target_file": "md.c", "threshold": "None", "start_line": 1, "end_line": 64},
    "nw-mach": {"target_file": "nw.c", "threshold": "None", "start_line": 1, "end_line": 104},
    "sort-merge-mach": {"target_file": "sort.c", "threshold": "None", "start_line": 1, "end_line": 64},
    "sort-radix-mach": {"target_file": "sort.c", "threshold": "None", "start_line": 77, "end_line": 110},
    "spmv-crs-mach": {"target_file": "spmv.c", "threshold": "None", "start_line": 1, "end_line": 30},
    "spmv-ellpack-mach": {"target_file": "spmv.c", "threshold": "None", "start_line": 1, "end_line": 29},
    "stencil-2d-mach": {"target_file": "stencil.c", "threshold": "None", "start_line": 1, "end_line": 32},
    "stencil-3d-mach": {"target_file": "stencil.c", "threshold": "None", "start_line": 1, "end_line": 58},
    "viterbi-mach": {"target_file": "viterbi.c", "threshold": "None", "start_line": 1, "end_line": 76},
}

__all__ = [
    "BenchmarkMeta",
    "BENCHMARK_POLYBENCH",
    "BENCHMARK_RODINIA",
    "BENCHMARK_CODEE",
    "BENCHMARK_PARBOIL",
    "BENCHMARK_MIBENCH",
    "BENCHMARK_PARSEC",
    "BENCHMARK_MACHSUITE",
]

