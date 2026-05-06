"""Header file dependencies for EX3 benchmarks.

This file specifies which header files and support files should be included
for each EX3 benchmark. EX3 benchmarks use CUDA kernels in EX3_optimized_codes/
directory, which is different from EX1/EX2 structure.

Usage:
    from configs.EX3_configs.head_code_ex3 import resolve_head_code_paths_ex3
    
    paths = resolve_head_code_paths_ex3(manager, "bfs")
"""

from __future__ import annotations

from pathlib import Path
from typing import List


def _abs_path(root: Path, *parts: str) -> Path:
    """Helper to construct absolute paths."""
    return root.joinpath(*parts)


def resolve_head_code_paths_ex3(manager, benchmark_name: str) -> List[Path]:
    """
    Compute the list of header/support files for EX3 benchmarks.
    
    Args:
        manager: Benchmark manager instance with absolute_path attribute
        benchmark_name: Name of the benchmark
        
    Returns:
        List of Path objects pointing to header files to include
        
    Note:
        EX3 benchmarks are all Rodinia benchmarks with CUDA kernels.
        The kernel files are typically in EX3/<benchmark>/EX3_optimized_codes/
    """
    head_code_paths: List[Path] = []
    root = manager.absolute_path
    name = benchmark_name
    
    # EX3-specific header dependencies
    # TODO: Add actual header file paths for each benchmark
    extra_map = {
        "b+tree": [
            # "EX3/b+tree/TODO.h",
        ],
        "backprop": [
            # "EX3/backprop/backprop.h",
        ],
        "bfs": [
            # "EX3/bfs/TODO.h",
        ],
        "bilateral": [
            "EX3/bilateral/add_info.txt",
            # Add other headers as needed
        ],
        "cfd": [
            "EX3/cfd/add_info.txt",
        ],
        "dwt2d": [
            # "EX3/dwt2d/TODO.h",
        ],
        "gaussian": [
            # "EX3/gaussian/TODO.h",
        ],
        "heartwall": [
            # "EX3/heartwall/TODO.h",
        ],
        "hotspot": [
            "EX3/hotspot/add_info.txt",
        ],
        "hotspot3D": [
            "EX3/hotspot3D/add_info.txt",
        ],
        "huffman": [
            # "EX3/huffman/TODO.h",
        ],
        "hybridsort": [
            # "EX3/hybridsort/TODO.h",
        ],
        "kmeans": [
            # "EX3/kmeans/TODO.h",
        ],
        "lavaMD": [
            "EX3/lavaMD/lavaMD.h",
            # Add other headers as needed
        ],
        "leukocyte": [
            "EX3/leukocyte/track-ellipse.h",
            # Add other headers as needed
        ],
        "lud": [
            "EX3/lud/add_info.txt",
        ],
        "myocyte": [
            "EX3/myocyte/kernel.cu",
            # Add other headers as needed
        ],
        "nw": [
            # "EX3/nw/TODO.h",
        ],
        "particlefilter": [
            # "EX3/particlefilter/TODO.h",
        ],
        "pathfinder": [
            "EX3/pathfinder/add_info.txt",
        ],
        "srad_v1": [
            # "EX3/srad_v1/TODO.h",
        ],
        "srad_v2": [
            # "EX3/srad_v2/TODO.h",
        ],
        "streamcluster": [
            # "EX3/streamcluster/TODO.h",
        ],
    }
    
    extra = extra_map.get(name, [])
    head_code_paths.extend(_abs_path(root, *path.split("/")) for path in extra)
    
    return head_code_paths


__all__ = ["resolve_head_code_paths_ex3"]
