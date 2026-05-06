"""EX3 benchmark metadata: start_line, end_line, and target_file information.

This file contains the code location information for EX3 benchmarks.
Note: EX3 benchmarks have different kernel locations compared to EX1/EX2.
"""

from __future__ import annotations

from typing import Any, Dict

BenchmarkMeta = Dict[str, Dict[str, Any]]

# EX3 Rodinia benchmarks - CUDA kernel locations
BENCHMARK_RODINIA_EX3: BenchmarkMeta = {
    "b+tree": {
        "target_file": ["kernel_gpu_cuda.cu", "kernel_gpu_cuda_2.cu"],
        "threshold": 0.002,
        "start_line": [1, 1],
        "end_line": [44, 62]
    },
    "backprop": {
        "target_file": "backprop_cuda_kernel.cu",
        "threshold": 0.002,
        "start_line": 1,
        "end_line": 89
    },
    "bfs": {
        "target_file": ["kernel.cu", "kernel2.cu"],
        "threshold": 0.002,
        "start_line": [1, 1],
        "end_line": [22, 16]
    },
    "bilateral": {
        "target_file": "bilateral_kernel.cu",
        "threshold": 0.002,
        "start_line": 80,
        "end_line": 106
    },
    "cfd": {
        "target_file": "euler3d.cu",
        "threshold": 0.002,
        "start_line": 231,
        "end_line": 417
    },
    "dwt2d": {
        "target_file": "fdwt97.cu",
        "threshold": 0.002,
        "start_line": 1,
        "end_line": 350
    },
    "gaussian": {
        "target_file": "gaussian.cu",
        "threshold": 0.002,
        "start_line": 351,
        "end_line": 387
    },
    "heartwall": {
        "target_file": "kernel.cu",
        "threshold": 0.002,
        "start_line": 1,
        "end_line": 1397
    },
    "hotspot": {
        "target_file": "hotspot.cu",
        "threshold": 0.002,
        "start_line": 96,
        "end_line": 217
    },
    "hotspot3D": {
        "target_file": "opt1.cu",
        "threshold": 0.002,
        "start_line": 7,
        "end_line": 54
    },
    "huffman": {
        "target_file": "vlc_kernel_sm64huff.cu",
        "threshold": 0.002,
        "start_line": 1,
        "end_line": 145
    },
    "hybridsort": {
        "target_file": ["bucketsort_kernel.cu", "mergesort_kernel.cu"],
        "threshold": 0.002,
        "start_line": [1, 1],
        "end_line": [115, 155]
    },
    "kmeans": {
        "target_file": "kmeans_cuda_kernel.cu",
        "threshold": 0.002,
        "start_line": 1,
        "end_line": 184
    },
    "lavaMD": {
        "target_file": "kernel_gpu_cuda.cu",
        "threshold": 0.002,
        "start_line": 1,
        "end_line": 193
    },
    "leukocyte": {
        "target_file": ["find_ellipse_kernel.cu", "track_ellipse_kernel.cu"],
        "threshold": 0.002,
        "start_line": [41, 48],
        "end_line": [147, 247]
    },
    "lud": {
        "target_file": "lud_kernel.cu",
        "threshold": 0.002,
        "start_line": 21,
        "end_line": 162
    },
    "myocyte": {
        "target_file": ["kernel_cam.cu", "kernel_ecc.cu", "kernel_fin.cu"],
        "threshold": 0.002,
        "start_line": [1, 1, 1],
        "end_line": [349, 1080, 131]
    },
    "nw": {
        "target_file": "needle_kernel.cu",
        "threshold": 0.002,
        "start_line": 1,
        "end_line": 176
    },
    "particlefilter": {
        "target_file": "particlefilter_naive.cu",
        "threshold": 0.002,
        "start_line": 119,
        "end_line": 142
    },
    "pathfinder": {
        "target_file": "pathfinder.cu",
        "threshold": 0.002,
        "start_line": 75,
        "end_line": 149
    },
    "srad_v1": {
        "target_file": ["srad_kernel.cu", "srad2_kernel.cu"],
        "threshold": 0.002,
        "start_line": [1, 1],
        "end_line": [77, 42]
    },
    "srad_v2": {
        "target_file": "srad_kernel.cu",
        "threshold": 0.002,
        "start_line": 1,
        "end_line": 237
    },
    "streamcluster": {
        "target_file": "streamcluster_cuda.cu",
        "threshold": 0.002,
        "start_line": 69,
        "end_line": 95
    },
}

__all__ = [
    "BenchmarkMeta",
    "BENCHMARK_RODINIA_EX3",
]
