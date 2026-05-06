"""Utilities for resolving benchmark-specific header dependencies."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence


def _abs_path(root: Path, *parts: str) -> Path:
    return root.joinpath(*parts)


def resolve_head_code_paths(manager, benchmark_name: str) -> List[Path]:
    """
    Compute the list of header/support files that should be injected into the LLM
    prompt for the given benchmark. The logic mirrors the legacy implementation in
    Benchmarks.__get_head_code, but is centralized for reuse and easier maintenance.
    """
    head_code_paths: List[Path] = []
    root = manager.absolute_path
    name = benchmark_name

    if name in manager.polybench_list:
        head_code_paths.append(_abs_path(root, "utilities", "polybench.h"))
        extra_map = {
            "correlation": ["EX1/correlation/correlation.h"],
            "covariance": ["EX1/covariance/covariance.h"],
            "gemm": ["EX1/gemm/gemm.h"],
            "gemver": ["EX1/gemver/gemver.h"],
            "gesummv": ["EX1/gesummv/gesummv.h"],
            "symm": ["EX1/symm/symm.h"],
            "syr2k": ["EX1/syr2k/syr2k.h"],
            "trmm": ["EX1/trmm/trmm.h"],
            "2mm": ["EX1/2mm/2mm.h"],
            "3mm": ["EX1/3mm/3mm.h"],
            "atax": ["EX1/atax/atax.h"],
            "bicg": ["EX1/bicg/bicg.h"],
            "doitgen": ["EX1/doitgen/doitgen.h"],
            "mvt": ["EX1/mvt/mvt.h"],
            "cholesky": ["EX1/cholesky/cholesky.h"],
            "durbin": ["EX1/durbin/durbin.h"],
            "gramschmidt": ["EX1/gramschmidt/gramschmidt.h"],
            "lu": ["EX1/lu/lu.h"],
            "ludcmp": ["EX1/ludcmp/ludcmp.h"],
            "trisolv": ["EX1/trisolv/trisolv.h"],
            "deriche": ["EX1/deriche/deriche.h"],
            "floyd-warshall": ["EX1/floyd-warshall/floyd-warshall.h"],
            "nussinov": ["EX1/nussinov/nussinov.h"],
            "adi": ["EX1/adi/adi.h"],
            "fdtd-2d": ["EX1/fdtd-2d/fdtd-2d.h"],
            "heat-3d": ["EX1/heat-3d/heat-3d.h"],
            "jacobi-1d": ["EX1/jacobi-1d/jacobi-1d.h"],
            "jacobi-2d": ["EX1/jacobi-2d/jacobi-2d.h"],
            "seidel-2d": ["EX1/seidel-2d/seidel-2d.h"],
        }
        extra = extra_map.get(name, [])
        head_code_paths.extend(_abs_path(root, *path.split("/")) for path in extra)

    elif name in manager.parboil_list:
        head_code_paths.extend([
            _abs_path(root, "common_parboil", "src", "parboil.c"),
            _abs_path(root, "common_parboil", "include", "parboil.h"),
        ])
        extra = {
            "sad": ["EX1/sad/sad.h"],
            "cutcp": ["EX1/cutcp/atom.h", "EX1/cutcp/cutoff.h"],
            "histo": ["EX1/histo/util.h"],
            "lbm": ["EX1/lbm/lbm.h", "EX1/lbm/lbm_1d_array.h", "EX1/lbm/config.h"],
            "mri-gridding": ["EX1/mri-gridding/CPU_kernels.h", "EX1/mri-gridding/UDTypes.h"],
            "stencil": ["EX1/stencil/common.h"],
            "tpacf": ["EX1/tpacf/model.h"],
        }.get(name, [])
        head_code_paths.extend(_abs_path(root, *path.split("/")) for path in extra)

    elif name in manager.rodinia_list:
        extra = {
            "myocyte": ["EX1/myocyte/cam.c", "EX1/myocyte/fin.c"],
            "leukocyte": ["EX1/leukocyte/track-ellipse.h"],
            "lavaMD": ["EX1/lavaMD/lavaMD.h"],
            "bilateral": ["EX1/bilateral/add_info.txt"],
        }.get(name, [])
        head_code_paths.extend(_abs_path(root, *path.split("/")) for path in extra)

    elif name in manager.codee_list:
        extra = {
            "haccmk": ["EX1/haccmk/haccmk.c"],
        }.get(name, [])
        head_code_paths.extend(_abs_path(root, *path.split("/")) for path in extra)

    elif name in manager.machsuite_list:
        extra_map = {
            "aes": ["EX1/aes/aes.h", "EX1/aes/add_info.txt"],
            "bfs-bulk-mach": ["EX1/bfs-bulk-mach/bfs.h"],
            "bfs-queue-mach": ["EX1/bfs-queue-mach/bfs.h"],
            "fft-strided-mach": ["EX1/fft-strided-mach/fft.h"],
            "fft-transpose-mach": ["EX1/fft-transpose-mach/fft.h", "EX1/fft-transpose-mach/add_info.txt"],
            "gemm-blocked-mach": ["EX1/gemm-blocked-mach/gemm.h"],
            "gemm-ncubed-mach": ["EX1/gemm-ncubed-mach/gemm.h"],
            "kmp-mach": ["EX1/kmp-mach/kmp.h", "EX1/kmp-mach/add_info.txt"],
            "md-grid-mach": ["EX1/md-grid-mach/md.h", "EX1/md-grid-mach/add_info.txt"],
            "md-knn-mach": ["EX1/md-knn-mach/md.h"],
            "nw-mach": ["EX1/nw-mach/nw.h", "EX1/nw-mach/add_info.txt"],
            "sort-merge-mach": ["EX1/sort-merge-mach/sort.h", "EX1/sort-merge-mach/add_info.txt"],
            "sort-radix-mach": ["EX1/sort-radix-mach/sort.h", "EX1/sort-radix-mach/add_info.txt"],
            "spmv-crs-mach": ["EX1/spmv-crs-mach/spmv.h"],
            "spmv-ellpack-mach": ["EX1/spmv-ellpack-mach/spmv.h"],
            "stencil-2d-mach": ["EX1/stencil-2d-mach/stencil.h"],
            "stencil-3d-mach": ["EX1/stencil-3d-mach/stencil.h"],
            "viterbi-mach": ["EX1/viterbi-mach/viterbi.h"],
        }
        extra = extra_map.get(name, [])
        head_code_paths.extend(_abs_path(root, *path.split("/")) for path in extra)

    elif name in manager.mibench_list:
        extra_map = {
            "basicmath-mibench": ["EX1/basicmath-mibench/snipmath.h"],
            "dijkstra-mibench": ["EX1/dijkstra-mibench/add_info.txt"],
            "stringsearch-mibench": ["EX1/stringsearch-mibench/add_info.txt"],
        }
        extra = extra_map.get(name, [])
        head_code_paths.extend(_abs_path(root, *path.split("/")) for path in extra)

    elif name in manager.parsec_list:
        if name == "canneal-parsec":
            head_code_paths.extend([
                _abs_path(root, "EX1", "canneal-parsec", "add_info.txt"),
                _abs_path(root, "EX1", "canneal-parsec", "annealer_thread.h"),
                _abs_path(root, "EX1", "canneal-parsec", "annealer_types.h"),
                _abs_path(root, "EX1", "canneal-parsec", "location_t.h"),
                _abs_path(root, "EX1", "canneal-parsec", "netlist_elem.h"),
                _abs_path(root, "EX1", "canneal-parsec", "rng.h"),
            ])

    else:
        raise ValueError(f"[ERROR] <auto_run> Unknown benchmark: {benchmark_name}. Please define metadata.")

    return head_code_paths

