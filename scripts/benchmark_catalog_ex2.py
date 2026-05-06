"""EX2-specific benchmark metadata allowing independent line ranges."""
from __future__ import annotations

from typing import Any, Dict, List

from .benchmark_catalog import (
    BENCHMARK_CODEE,
    BENCHMARK_MACHSUITE,
    BENCHMARK_MIBENCH,
    BENCHMARK_PARBOIL,
    BENCHMARK_PARSEC,
    BENCHMARK_POLYBENCH,
    BENCHMARK_RODINIA,
)

BenchmarkMeta = Dict[str, Dict[str, Any]]


def _clone(meta: BenchmarkMeta) -> BenchmarkMeta:
    """Create a shallow copy of the benchmark mapping so EX2 can tweak values."""
    return {name: dict(spec) for name, spec in meta.items()}


BENCHMARK_POLYBENCH_EX2: BenchmarkMeta = _clone(BENCHMARK_POLYBENCH)
BENCHMARK_RODINIA_EX2: BenchmarkMeta = _clone(BENCHMARK_RODINIA)
BENCHMARK_CODEE_EX2: BenchmarkMeta = _clone(BENCHMARK_CODEE)
BENCHMARK_PARBOIL_EX2: BenchmarkMeta = _clone(BENCHMARK_PARBOIL)
BENCHMARK_MIBENCH_EX2: BenchmarkMeta = _clone(BENCHMARK_MIBENCH)
BENCHMARK_PARSEC_EX2: BenchmarkMeta = _clone(BENCHMARK_PARSEC)
BENCHMARK_MACHSUITE_EX2: BenchmarkMeta = _clone(BENCHMARK_MACHSUITE)

# Placeholder for future EX2-specific overrides (e.g., adjusted start/end lines).
# Example:
# EX2_OVERRIDES = {
#     "2mm": {"start_line": 79, "end_line": 109},
# }
EX2_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "2mm": {"start_line": 78, "end_line": 108},
    "3mm": {"start_line": 74, "end_line": 113},
    "adi": {"start_line": 69, "end_line": 131},
    "atax": {"start_line": 67, "end_line": 89},
    "bicg": {"start_line": 75, "end_line": 99},
    "cholesky": {"start_line": 85, "end_line": 109},
    "correlation": {"start_line": 68, "end_line": 127},
    "covariance": {"start_line": 66, "end_line": 99},
    "deriche": {"start_line": 69, "end_line": 158},
    "doitgen": {"start_line": 68, "end_line": 88},
    "durbin": {"start_line": 63, "end_line": 98},
    "fdtd-2d": {"start_line": 92, "end_line": 122},
    "gemm": {"start_line": 74, "end_line": 94},
    "gemver": {"start_line": 86, "end_line": 120},
    "gesummv": {"start_line": 73, "end_line": 99},
    "gramschmidt": {"start_line": 81, "end_line": 111},
    "heat-3d": {"start_line": 66, "end_line": 99},
    "jacobi-1d": {"start_line": 66, "end_line": 84},
    "jacobi-2d": {"start_line": 67, "end_line": 87},
    "lu": {"start_line": 86, "end_line": 107},
    "ludcmp": {"start_line": 96, "end_line": 140},
    "mvt": {"start_line": 80, "end_line": 99},
    "nussinov": {"start_line": 82, "end_line": 112},
    "seidel-2d": {"start_line": 63, "end_line": 79},
    "symm": {"start_line": 76, "end_line": 99},
    "syr2k": {"start_line": 74, "end_line": 95},
    "syrk": {"start_line": 70, "end_line": 89},
    "trisolv": {"start_line": 68, "end_line": 86},
    "trmm": {"start_line": 72, "end_line": 89},
    "floyd-warshall": {"start_line": 54, "end_line": 70},
    # parboil
    "cutcp": {"start_line": 1, "end_line": 222},
    "sad": {"start_line": 1, "end_line": 203},
    "histo": {"start_line": 1, "end_line": 143},
    # lbm same as previous version, no need to write
    "mri-gridding": {"start_line": 1, "end_line": 198},
    "mri-q": {"start_line": 1, "end_line": 68},
    "sgemm": {"start_line": 1, "end_line": 29},
    "spmv": {"start_line": 1, "end_line": 137},
    "stencil": {"start_line": 1, "end_line": 28},
    "tpacf": {"start_line": 1, "end_line": 63},
    # rodinia
    "streamcluster": {"start_line": 357, "end_line": 600},
    "srad_v1": {"start_line": 1, "end_line": 346},
    "srad_v2": {"start_line": 1, "end_line": 251},
    "pathfinder": {"start_line": 107, "end_line": 175},
    "particlefilter": {"start_line": 364, "end_line": 584},
    "nw": {"start_line": 136, "end_line": 310},
    "myocyte": {"start_line": 1, "end_line": 88},
    "lud": {"start_line": 1, "end_line": 26},
    "leukocyte": {"start_line": 7, "end_line": 191},
    "kmeans": {"start_line": 1, "end_line": 122},
    "lavaMD": {"start_line": 1, "end_line": 143},
    "hotspot3D": {"start_line": 130, "end_line": 176},
    "hotspot": {"start_line": 50, "end_line": 160},
    "heartwall": {"start_line": 1, "end_line": 599},
    "cfd": {"start_line": 196, "end_line": 405},
    "bfs": {"start_line": 1, "end_line": 193},
    "backprop": {"start_line": 223, "end_line": 259},
    "b+tree": {
        "target_file": ["kernel-cpu.c", "kernel-cpu-2.c"],
        "start_line": [1, 1],
        "end_line": [80, 104],
    },
    "bilateral": {"start_line": 98, "end_line": 163},
    # parsec
    "canneal-parsec": {"start_line": 22, "end_line": 67},
    # codee
    "matmul": {"start_line": 9, "end_line": 31},
    "coulomb": {"start_line": 19, "end_line": 43},
    # pi is the same as previous version, no need to write
    "atmux": {"start_line": 20, "end_line": 30},
    "haccmk": {"start_line": 1, "end_line": 51},
    # mibench
    "basicmath-mibench": {"start_line": 13, "end_line": 50},
    "bitcount-mibench": {"start_line": 123, "end_line": 137},
    "dijkstra-mibench": {"start_line": 106, "end_line": 156},
    "patricia-mibench": {"start_line": 66, "end_line": 178},
    "stringsearch-mibench": {"start_line": 29, "end_line": 62},
    "qsort-mibench": {"start_line": 1, "end_line": 117},
    "susan-s-mibench": {"start_line": 394, "end_line": 532},
    "susan-e-mibench": {"start_line": 786, "end_line": 1026},
    "susan-c-mibench": {"start_line": 1187, "end_line": 1472},
    # machsuite
    "aes": {"start_line": 182, "end_line": 215},
    "bfs-bulk-mach": {"start_line": 1, "end_line": 52},
    "bfs-queue-mach": {"start_line": 1, "end_line": 59},
    "fft-strided-mach": {"start_line": 1, "end_line": 47},
    "fft-transpose-mach": {"start_line": 117, "end_line": 418},
    "gemm-blocked-mach": {"start_line": 1, "end_line": 39},
    "gemm-ncubed-mach": {"start_line": 1, "end_line": 36},
    "kmp-mach": {"start_line": 1, "end_line": 55},
    "md-grid-mach": {"start_line": 1, "end_line": 73},
    "md-knn-mach": {"start_line": 1, "end_line": 67},
    "nw-mach": {"start_line": 1, "end_line": 107},
    "sort-merge-mach": {"start_line": 1, "end_line": 67},
    "sort-radix-mach": {"start_line": 80, "end_line": 113},
    "spmv-crs-mach": {"start_line": 1, "end_line": 33},
    "spmv-ellpack-mach": {"start_line": 1, "end_line": 32},
    "stencil-2d-mach": {"start_line": 1, "end_line": 35},
    "stencil-3d-mach": {"start_line": 1, "end_line": 61},
    "viterbi-mach": {"start_line": 1, "end_line": 79},
}


def _apply_overrides(overrides: Dict[str, Dict[str, Any]], registries: List[BenchmarkMeta]) -> None:
    if not overrides:
        return
    for bench, patch in overrides.items():
        for registry in registries:
            if bench in registry:
                registry[bench].update(patch)
                break


_apply_overrides(
    EX2_OVERRIDES,
    [
        BENCHMARK_POLYBENCH_EX2,
        BENCHMARK_RODINIA_EX2,
        BENCHMARK_CODEE_EX2,
        BENCHMARK_PARBOIL_EX2,
        BENCHMARK_MIBENCH_EX2,
        BENCHMARK_PARSEC_EX2,
        BENCHMARK_MACHSUITE_EX2,
    ],
)

__all__ = [
    "BenchmarkMeta",
    "BENCHMARK_POLYBENCH_EX2",
    "BENCHMARK_RODINIA_EX2",
    "BENCHMARK_CODEE_EX2",
    "BENCHMARK_PARBOIL_EX2",
    "BENCHMARK_MIBENCH_EX2",
    "BENCHMARK_PARSEC_EX2",
    "BENCHMARK_MACHSUITE_EX2",
]


