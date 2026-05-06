from __future__ import annotations

from typing import Dict, List

MOTIF_GROUPS: Dict[str, List[str]] = {
    "dense_linear_algebra": [
        "2mm",
        "3mm",
        "gemm",
        "sgemm",
        "gemm-blocked-mach",
        "gemm-ncubed-mach",
        "matmul",
        "trmm",
        "symm",
        "syr2k",
        "syrk",
        "gemver",
        "lu",
        "lud",
        "ludcmp",
        "cholesky",
        "gramschmidt",
        "trisolv",
    ],
    "sparse_linear_algebra": [
        "spmv",
        "spmv-crs-mach",
        "spmv-ellpack-mach",
        "bicg",
        "atax",
        "mvt",
        "gesummv",
    ],
    "spectral_methods": [
        "fft-strided-mach",
        "fft-transpose-mach",
    ],
    "stencil_computations": [
        "jacobi-1d",
        "jacobi-2d",
        "seidel-2d",
        "adi",
        "heat-3d",
        "fdtd-2d",
        "hotspot",
        "hotspot3D",
        "stencil",
        "stencil-2d-mach",
        "stencil-3d-mach",
        "srad_v1",
        "srad_v2",
    ],
    "dynamic_programming": [
        "nw",
        "nw-mach",
        "nussinov",
        "floyd-warshall",
        "pathfinder",
    ],
    "n_body_methods": [
        "lavaMD",
        "md-grid-mach",
        "md-knn-mach",
        "coulomb",
        "cutcp",
        "haccmk",
    ],
    "graph_algorithms": [
        "bfs",
        "bfs-bulk-mach",
        "bfs-queue-mach",
        "dijkstra-mibench",
        "b+tree",
        "patricia-mibench",
    ],
    "clustering_and_ml": [
        "kmeans",
        "backprop",
        "streamcluster",
        "canneal-parsec",
    ],
    "image_and_video_processing": [
        "bilateral",
        "deriche",
        "sad",
        "susan-c-mibench",
        "susan-e-mibench",
        "susan-s-mibench",
        "leukocyte",
    ],
    "computational_fluid_dynamics": [
        "cfd",
        "lbm",
    ],
    "medical_and_scientific": [
        "myocyte",
        "heartwall",
        "mri-q",
        "mri-gridding",
    ],
    "sorting_and_searching": [
        "sort-radix-mach",
        "sort-merge-mach",
        "qsort-mibench",
        "kmp-mach",
        "stringsearch-mibench",
    ],
    "statistical_computations": [
        "correlation",
        "covariance",
        "durbin",
        "doitgen",
    ],
    "monte_carlo_methods": [
        "pi",
        "particlefilter",
        "tpacf",
    ],
    "cryptography_and_encoding": [
        "aes",
        "viterbi-mach",
    ],
    "utility": [
        "bitcount-mibench",
        "basicmath-mibench",
        "histo",
        "atmux",
    ],
}

BENCHMARK_TO_MOTIF: Dict[str, str] = {}
for motif_name, benchmarks in MOTIF_GROUPS.items():
    for benchmark in benchmarks:
        BENCHMARK_TO_MOTIF[benchmark] = motif_name


def motif_of(benchmark: str) -> str:
    return BENCHMARK_TO_MOTIF.get(benchmark, "unknown")


def total_benchmarks() -> int:
    return len(BENCHMARK_TO_MOTIF)


if __name__ == "__main__":
    print(total_benchmarks())