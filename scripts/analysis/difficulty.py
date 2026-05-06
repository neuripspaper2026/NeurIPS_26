from __future__ import annotations

from typing import Dict, List

DIFFICULTY_GROUPS: Dict[str, List[str]] = {
    "d1": [
        "2mm", "3mm", "adi", "aes", "atax", "atmux", "backprop", "basicmath-mibench",
        "bfs-bulk-mach", "bicg", "bilateral", "bitcount-mibench", "canneal-parsec",
        "cholesky", "correlation", "coulomb", "covariance", "deriche", "dijkstra-mibench",
        "doitgen", "durbin", "fdtd-2d", "fft-strided-mach", "floyd-warshall", "gemm",
        "gemm-blocked-mach", "gemm-ncubed-mach", "gemver", "gesummv", "gramschmidt",
        "haccmk", "heat-3d", "jacobi-1d", "jacobi-2d", "lavaMD", "leukocyte", "lu",
        "lud", "ludcmp", "matmul", "md-knn-mach", "mri-q", "mvt", "myocyte",
        "seidel-2d", "sgemm", "sort-merge-mach", "sort-radix-mach", "spmv",
        "spmv-crs-mach", "spmv-ellpack-mach", "stencil", "stencil-2d-mach",
        "stencil-3d-mach", "stringsearch-mibench", "symm", "syr2k", "syrk",
        "trisolv", "trmm", "viterbi-mach",
    ],
    "d2": [
        "bfs-queue-mach", "cfd", "fft-transpose-mach", "histo", "hotspot3D",
        "kmeans", "kmp-mach", "md-grid-mach", "nussinov", "nw-mach", "particlefilter",
        "pathfinder", "patricia-mibench", "pi", "srad_v1", "susan-s-mibench", "tpacf",
    ],
    "d3": [
        "b+tree", "bfs", "hotspot", "mri-gridding", "nw", "qsort-mibench",
        "sad", "srad_v2", "streamcluster",
    ],
    "d4": [
        "cutcp", "heartwall", "lbm", "susan-c-mibench", "susan-e-mibench",
    ],
}

BENCHMARK_TO_DIFFICULTY: Dict[str, str] = {}
for difficulty_name, benchmarks in DIFFICULTY_GROUPS.items():
    for benchmark in benchmarks:
        BENCHMARK_TO_DIFFICULTY[benchmark] = difficulty_name


def difficulty_of(benchmark: str) -> str:
    return BENCHMARK_TO_DIFFICULTY.get(benchmark, "unknown")


def total_benchmarks() -> int:
    return len(BENCHMARK_TO_DIFFICULTY)


if __name__ == "__main__":
    print(total_benchmarks())