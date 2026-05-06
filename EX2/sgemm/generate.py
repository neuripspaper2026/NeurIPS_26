#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import json
import argparse
from typing import Tuple, Optional, Dict
import random


# -------------------------
# Utilities
# -------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_join(out_dir: str, filename: str, overwrite: bool) -> str:
    p = os.path.join(out_dir, filename)
    if (not overwrite) and os.path.exists(p):
        raise FileExistsError(f"Output exists: {p}. Use --overwrite to replace.")
    return p


def write_meta(path_noext: str, meta: Dict) -> None:
    with open(path_noext + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


# -------------------------
# Format: SGEMM matrix text files
#   Text format for matrix:
#     Line 1: rows cols
#     Following lines: space-separated float values (column-major layout)
#   
#   For SGEMM: C = A * B
#   Three files needed:
#     - matrix1.txt: Matrix A (M x K)
#     - matrix2.txt: Matrix B (K x N) 
#     - matrix2t.txt: Matrix B transposed (N x K) for optimization
# -------------------------

def write_matrix_file(path: str, rows: int, cols: int, data: list) -> None:
    """Write matrix in column-major text format"""
    with open(path, 'w') as f:
        f.write(f"{rows} {cols} ")
        for val in data:
            f.write(f"{val} ")
        f.write("\n")


def read_matrix_file(path: str) -> Tuple[int, int, list]:
    """Read matrix from column-major text format"""
    with open(path, 'r') as f:
        content = f.read().strip().split()
    rows = int(content[0])
    cols = int(content[1])
    data = [float(x) for x in content[2:]]
    return rows, cols, data


def transpose_matrix(rows: int, cols: int, data: list) -> Tuple[int, int, list]:
    """Transpose a column-major matrix"""
    transposed = []
    for i in range(cols):
        for j in range(rows):
            transposed.append(data[j * cols + i])
    return cols, rows, transposed


def validate_dims(M: int, K: int, N: int) -> None:
    if M <= 0 or K <= 0 or N <= 0:
        raise ValueError("Matrix dimensions must be positive")
    if M > 100000 or K > 100000 or N > 100000:
        raise ValueError("Matrix dimensions too large (max 100k per dimension)")


# -------------------------
# Presets
# -------------------------

PRESETS: Dict[str, Tuple[int, int, int]] = {
    # name: (M, K, N) for C[M×N] = A[M×K] * B[K×N]
    # Based on actual parboil sgemm dataset dimensions
    "mini":   (32, 32, 32),          # Tiny matrices for testing
    "small":  (128, 96, 160),        # Reference small (matches parboil small)
    "medium": (1024, 992, 1056),     # Medium scale (matches parboil medium)
    "large":  (2048, 1984, 2112),    # Large scale (scaled from medium)
    "extra-large": (4096, 3968, 4224),  # Extra large (scaled from medium)
}


# -------------------------
# Synthetic generators
# -------------------------

def gen_random_matrix(rows: int, cols: int, seed: int, min_val: float = 0.0, max_val: float = 1.0) -> list:
    """Generate random matrix values in column-major order"""
    rng = random.Random(seed)
    return [rng.uniform(min_val, max_val) for _ in range(rows * cols)]


def gen_identity_matrix(size: int) -> list:
    """Generate identity matrix in column-major order"""
    data = []
    for col in range(size):
        for row in range(size):
            data.append(1.0 if row == col else 0.0)
    return data


def gen_diagonal_matrix(size: int, seed: int) -> list:
    """Generate diagonal matrix in column-major order"""
    rng = random.Random(seed)
    data = []
    for col in range(size):
        for row in range(size):
            if row == col:
                data.append(rng.uniform(0.5, 2.0))
            else:
                data.append(0.0)
    return data


def write_one(out_dir: str, name_prefix: str, M: int, K: int, N: int,
              matrix_a: list, matrix_b: list, overwrite: bool) -> Tuple[str, str, str, str]:
    """Write matrix1.txt, matrix1t.txt, matrix2.txt, and matrix2t.txt"""
    ensure_dir(out_dir)
    
    # matrix1.txt: A (M x K)
    matrix1_path = safe_join(out_dir, f"{name_prefix}matrix1.txt", overwrite)
    write_matrix_file(matrix1_path, M, K, matrix_a)
    
    # matrix1t.txt: A^T (K x M)
    K_t1, M_t1, matrix_a_t = transpose_matrix(M, K, matrix_a)
    matrix1t_path = safe_join(out_dir, f"{name_prefix}matrix1t.txt", overwrite)
    write_matrix_file(matrix1t_path, K_t1, M_t1, matrix_a_t)
    
    # matrix2.txt: B (K x N)
    matrix2_path = safe_join(out_dir, f"{name_prefix}matrix2.txt", overwrite)
    write_matrix_file(matrix2_path, K, N, matrix_b)
    
    # matrix2t.txt: B^T (N x K)
    N_t, K_t, matrix_b_t = transpose_matrix(K, N, matrix_b)
    matrix2t_path = safe_join(out_dir, f"{name_prefix}matrix2t.txt", overwrite)
    write_matrix_file(matrix2t_path, N_t, K_t, matrix_b_t)
    
    return matrix1_path, matrix1t_path, matrix2_path, matrix2t_path


def generate_synthetic(out_dir: str,
                       count: int,
                       preset: Optional[str],
                       seed: int,
                       M: Optional[int],
                       K: Optional[int],
                       N: Optional[int],
                       kind: str,
                       start_index: int,
                       digits: int,
                       overwrite: bool,
                       fixed_name: bool) -> None:
    if preset is not None:
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset: {preset}")
        p_M, p_K, p_N = PRESETS[preset]
    else:
        p_M = None
        p_K = None
        p_N = None

    M = M if M is not None else p_M
    K = K if K is not None else p_K
    N = N if N is not None else p_N

    if M is None or K is None or N is None:
        raise ValueError("Must provide preset or explicit --M, --K, --N")

    validate_dims(M, K, N)

    for i in range(start_index, start_index + count):
        if kind == 'uniform':
            matrix_a = gen_random_matrix(M, K, seed + i * 2, 0.0, 1.0)
            matrix_b = gen_random_matrix(K, N, seed + i * 2 + 1, 0.0, 1.0)
        elif kind == 'normal':
            # Use uniform as approximation (for compatibility)
            matrix_a = gen_random_matrix(M, K, seed + i * 2, -1.0, 1.0)
            matrix_b = gen_random_matrix(K, N, seed + i * 2 + 1, -1.0, 1.0)
        elif kind == 'identity':
            if M != K or K != N:
                raise ValueError("Identity matrix requires M == K == N")
            matrix_a = gen_identity_matrix(M)
            matrix_b = gen_identity_matrix(M)
        elif kind == 'diagonal':
            if M != K or K != N:
                raise ValueError("Diagonal matrix requires M == K == N")
            matrix_a = gen_diagonal_matrix(M, seed + i * 2)
            matrix_b = gen_diagonal_matrix(M, seed + i * 2 + 1)
        else:
            raise ValueError("Unknown kind. Use: uniform|normal|identity|diagonal")

        if fixed_name:
            name_prefix = ""
        else:
            idx_str = f"{i:0{digits}d}"
            name_prefix = f"{idx_str}_"

        m1_path, m1t_path, m2_path, m2t_path = write_one(
            out_dir, name_prefix, M, K, N, matrix_a, matrix_b, overwrite
        )
        
        base_path = m1_path.replace("matrix1.txt", "")
        write_meta(base_path.rstrip("_"), {
            "M": M,
            "K": K,
            "N": N,
            "kind": kind,
            "seed": seed + i,
        })


def from_sources(src_glob: str,
                 out_dir: str,
                 start_index: int,
                 digits: int,
                 overwrite: bool,
                 fixed_name: bool) -> None:
    # Find matrix1.txt files
    if "*" not in src_glob:
        src_glob = os.path.join(src_glob, "*matrix1.txt")
    
    paths = sorted(glob.glob(src_glob))
    if not paths:
        raise FileNotFoundError(f"No matrix1.txt files matched: {src_glob}")
    
    for j, m1_src in enumerate(paths, start=start_index):
        # Read matrix1.txt
        M, K, matrix_a = read_matrix_file(m1_src)
        
        # Find corresponding matrix2.txt
        base = m1_src.replace("matrix1.txt", "")
        m2_src = base + "matrix2.txt"
        if not os.path.exists(m2_src):
            raise FileNotFoundError(f"Missing matrix2.txt: {m2_src}")
        
        K2, N, matrix_b = read_matrix_file(m2_src)
        if K != K2:
            raise ValueError(f"Dimension mismatch: matrix1 K={K}, matrix2 K={K2}")
        
        ensure_dir(out_dir)
        if fixed_name:
            name_prefix = ""
        else:
            idx_str = f"{j:0{digits}d}"
            name_prefix = f"{idx_str}_"
        
        m1_dst, m1t_dst, m2_dst, m2t_dst = write_one(
            out_dir, name_prefix, M, K, N, matrix_a, matrix_b, overwrite
        )
        
        base_path = m1_dst.replace("matrix1.txt", "")
        write_meta(base_path.rstrip("_"), {
            "M": M,
            "K": K,
            "N": N,
            "source": os.path.abspath(m1_src),
        })


# -------------------------
# CLI
# -------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SGEMM input generator (matrix text files)")
    sub = p.add_subparsers(dest='cmd', required=True)

    sp_syn = sub.add_parser('synthetic', help='Generate synthetic matrix data')
    sp_syn.add_argument('--out-dir', required=True, help='Output directory (created recursively)')
    sp_syn.add_argument('--count', type=int, default=1, help='Number of datasets to generate')
    sp_syn.add_argument('--preset', choices=list(PRESETS.keys()), default=None, help='Preset size')
    sp_syn.add_argument('--M', type=int, default=None, help='Override M (rows of A, rows of C)')
    sp_syn.add_argument('--K', type=int, default=None, help='Override K (cols of A, rows of B)')
    sp_syn.add_argument('--N', type=int, default=None, help='Override N (cols of B, cols of C)')
    sp_syn.add_argument('--kind', choices=['uniform', 'normal', 'identity', 'diagonal'], default='uniform',
                        help='Matrix generation pattern')
    sp_syn.add_argument('--seed', type=int, default=0, help='Base RNG seed')
    sp_syn.add_argument('--start-index', type=int, default=0, help='Starting index')
    sp_syn.add_argument('--digits', type=int, default=4, help='Zero-padding width')
    sp_syn.add_argument('--overwrite', action='store_true', help='Allow overwriting existing files')
    sp_syn.add_argument('--fixed-name', action='store_true',
                        help='Use fixed naming (no index prefix)')

    sp_src = sub.add_parser('from-sources', help='Copy/validate existing matrix files')
    sp_src.add_argument('--src-glob', required=True, help='Glob for existing matrix1.txt files')
    sp_src.add_argument('--out-dir', required=True, help='Output directory (created recursively)')
    sp_src.add_argument('--start-index', type=int, default=0)
    sp_src.add_argument('--digits', type=int, default=4)
    sp_src.add_argument('--overwrite', action='store_true')
    sp_src.add_argument('--fixed-name', action='store_true')

    return p


def main() -> None:
    p = build_argparser()
    args = p.parse_args()

    if args.cmd == 'synthetic':
        generate_synthetic(
            out_dir=args.out_dir,
            count=args.count,
            preset=args.preset,
            seed=args.seed,
            M=args.M,
            K=args.K,
            N=args.N,
            kind=args.kind,
            start_index=args.start_index,
            digits=args.digits,
            overwrite=args.overwrite,
            fixed_name=args.fixed_name,
        )
    elif args.cmd == 'from-sources':
        from_sources(
            src_glob=args.src_glob,
            out_dir=args.out_dir,
            start_index=args.start_index,
            digits=args.digits,
            overwrite=args.overwrite,
            fixed_name=args.fixed_name,
        )
    else:
        p.print_help()


if __name__ == '__main__':
    main()

