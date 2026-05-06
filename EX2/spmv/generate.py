#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import json
import argparse
import struct
import random
from typing import Tuple, Optional, Dict, List


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
# Format: Matrix Market (COO format) + Binary vector
#   Matrix Market format (.mtx):
#     Header: %%MatrixMarket matrix coordinate real [symmetric/general]
#     Size line: rows cols nonzeros
#     Data lines: row col value (1-indexed)
#   
#   Vector format (.bin):
#     Binary file with `dim` float32 values (little-endian)
# -------------------------

def write_mtx_file(path: str, rows: int, cols: int, entries: List[Tuple[int, int, float]], 
                    is_symmetric: bool = False, name: str = "generated") -> None:
    """Write sparse matrix in Matrix Market format"""
    with open(path, 'w') as f:
        # Header
        matrix_type = "symmetric" if is_symmetric else "general"
        f.write(f"%%MatrixMarket matrix coordinate real {matrix_type}\n")
        f.write(f"% Generated sparse matrix: {name}\n")
        f.write(f"{rows} {cols} {len(entries)}\n")
        
        # Data entries (1-indexed)
        for row, col, val in entries:
            f.write(f"{row+1} {col+1} {val}\n")


def read_mtx_file(path: str) -> Tuple[int, int, int, List[Tuple[int, int, float]], bool]:
    """Read Matrix Market format file, returns (rows, cols, nnz, entries, is_symmetric)"""
    with open(path, 'r') as f:
        # Read header
        header = f.readline().strip()
        if not header.startswith("%%MatrixMarket"):
            raise ValueError(f"Invalid Matrix Market file: {path}")
        
        is_symmetric = "symmetric" in header.lower()
        
        # Skip comments
        while True:
            line = f.readline().strip()
            if not line.startswith('%'):
                break
        
        # Size line
        rows, cols, nnz = map(int, line.split())
        
        # Read entries
        entries = []
        for _ in range(nnz):
            parts = f.readline().strip().split()
            row = int(parts[0]) - 1  # Convert to 0-indexed
            col = int(parts[1]) - 1
            val = float(parts[2])
            entries.append((row, col, val))
    
    return rows, cols, nnz, entries, is_symmetric


def write_vector_bin(path: str, vec: List[float]) -> None:
    """Write dense vector in binary format (float32, little-endian)"""
    with open(path, 'wb') as f:
        for val in vec:
            f.write(struct.pack('<f', val))


def read_vector_bin(path: str, dim: int) -> List[float]:
    """Read dense vector from binary format"""
    with open(path, 'rb') as f:
        vec = []
        for _ in range(dim):
            vec.append(struct.unpack('<f', f.read(4))[0])
    return vec


def validate_dims(rows: int, cols: int, nnz: int) -> None:
    if rows <= 0 or cols <= 0:
        raise ValueError("Matrix dimensions must be positive")
    if nnz < 0 or nnz > rows * cols:
        raise ValueError(f"Invalid nonzero count: {nnz} (max: {rows * cols})")


# -------------------------
# Presets
# -------------------------

PRESETS: Dict[str, Tuple[int, int, int, float]] = {
    # name: (rows, cols, nnz, density_ratio)
    "mini":        (128,    128,    512,    0.03),      # Tiny for testing
    "small":       (1138,   1138,   2596,   0.002),     # Reference small (similar to 1138_bus)
    "medium":      (11948,  11948,  80519,  0.0006),    # Medium (similar to bcsstk18)
    "large":       (146689, 146689, 1009977, 0.00005),  # Large (similar to Dubcova3)
    "extra-large": (300000, 300000, 3000000, 0.00003),  # Extra large
}


# -------------------------
# Synthetic generators
# -------------------------

def gen_random_sparse(rows: int, cols: int, nnz: int, seed: int, 
                      is_symmetric: bool = False, 
                      min_val: float = -10.0, max_val: float = 10.0) -> List[Tuple[int, int, float]]:
    """Generate random sparse matrix entries"""
    rng = random.Random(seed)
    entries = []
    
    if is_symmetric:
        # Generate upper triangular entries
        positions = set()
        while len(positions) < nnz:
            i = rng.randint(0, rows - 1)
            j = rng.randint(i, cols - 1)  # Upper triangular
            positions.add((i, j))
        
        for i, j in positions:
            val = rng.uniform(min_val, max_val)
            entries.append((i, j, val))
    else:
        # Generate random entries
        positions = set()
        while len(positions) < nnz:
            i = rng.randint(0, rows - 1)
            j = rng.randint(0, cols - 1)
            positions.add((i, j))
        
        for i, j in positions:
            val = rng.uniform(min_val, max_val)
            entries.append((i, j, val))
    
    # Sort by row, then col
    entries.sort()
    return entries


def gen_diagonal_sparse(rows: int, cols: int, seed: int, bandwidth: int = 1) -> List[Tuple[int, int, float]]:
    """Generate diagonal/banded sparse matrix"""
    rng = random.Random(seed)
    entries = []
    
    for i in range(rows):
        for offset in range(-bandwidth, bandwidth + 1):
            j = i + offset
            if 0 <= j < cols:
                val = rng.uniform(0.1, 10.0) if offset == 0 else rng.uniform(-5.0, 5.0)
                entries.append((i, j, val))
    
    return entries


def gen_random_vector(dim: int, seed: int, min_val: float = 0.0, max_val: float = 1.0) -> List[float]:
    """Generate random dense vector"""
    rng = random.Random(seed)
    return [rng.uniform(min_val, max_val) for _ in range(dim)]


def write_one(out_dir: str, name_prefix: str, rows: int, cols: int, 
              entries: List[Tuple[int, int, float]], vec: List[float],
              is_symmetric: bool, overwrite: bool) -> Tuple[str, str]:
    """Write matrix and vector files"""
    ensure_dir(out_dir)
    
    # Matrix file (.mtx)
    matrix_name = f"{name_prefix}matrix.mtx"
    matrix_path = safe_join(out_dir, matrix_name, overwrite)
    write_mtx_file(matrix_path, rows, cols, entries, is_symmetric, name_prefix)
    
    # Vector file (.bin)
    vec_name = "vector.bin" if not name_prefix else f"{name_prefix}vector.bin"
    vec_path = safe_join(out_dir, vec_name, overwrite)
    write_vector_bin(vec_path, vec)
    
    return matrix_path, vec_path


def generate_synthetic(out_dir: str,
                       count: int,
                       preset: Optional[str],
                       seed: int,
                       rows: Optional[int],
                       cols: Optional[int],
                       nnz: Optional[int],
                       kind: str,
                       start_index: int,
                       digits: int,
                       overwrite: bool,
                       fixed_name: bool,
                       symmetric: bool) -> None:
    if preset is not None:
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset: {preset}")
        p_rows, p_cols, p_nnz, _ = PRESETS[preset]
    else:
        p_rows = None
        p_cols = None
        p_nnz = None

    rows = rows if rows is not None else p_rows
    cols = cols if cols is not None else p_cols
    nnz = nnz if nnz is not None else p_nnz

    if rows is None or cols is None or nnz is None:
        raise ValueError("Must provide preset or explicit --rows, --cols, --nnz")

    validate_dims(rows, cols, nnz)

    for i in range(start_index, start_index + count):
        if kind == 'random':
            entries = gen_random_sparse(rows, cols, nnz, seed + i * 2, symmetric)
        elif kind == 'diagonal':
            bandwidth = max(1, int((nnz / rows) / 2))
            entries = gen_diagonal_sparse(rows, cols, seed + i * 2, bandwidth)
        elif kind == 'banded':
            bandwidth = max(5, int((nnz / rows) / 2))
            entries = gen_diagonal_sparse(rows, cols, seed + i * 2, bandwidth)
        else:
            raise ValueError("Unknown kind. Use: random|diagonal|banded")
        
        # Generate vector
        dim = cols  # Vector dimension matches matrix column count
        vec = gen_random_vector(dim, seed + i * 2 + 1)
        
        if fixed_name:
            name_prefix = ""
        else:
            idx_str = f"{i:0{digits}d}"
            name_prefix = f"{idx_str}_"

        mat_path, vec_path = write_one(
            out_dir, name_prefix, rows, cols, entries, vec, symmetric, overwrite
        )
        
        base_name = mat_path.replace(".mtx", "")
        write_meta(base_name, {
            "rows": rows,
            "cols": cols,
            "nnz": len(entries),
            "kind": kind,
            "symmetric": symmetric,
            "seed": seed + i,
        })


def from_sources(src_glob: str,
                 out_dir: str,
                 start_index: int,
                 digits: int,
                 overwrite: bool,
                 fixed_name: bool) -> None:
    # Find matrix files
    if "*" not in src_glob:
        src_glob = os.path.join(src_glob, "*.mtx")
    
    paths = sorted(glob.glob(src_glob))
    if not paths:
        raise FileNotFoundError(f"No .mtx files matched: {src_glob}")
    
    for j, mtx_src in enumerate(paths, start=start_index):
        # Read matrix
        rows, cols, nnz, entries, is_symmetric = read_mtx_file(mtx_src)
        
        # Try to find corresponding vector file
        base = os.path.splitext(mtx_src)[0]
        vec_src = os.path.join(os.path.dirname(mtx_src), "vector.bin")
        
        if os.path.exists(vec_src):
            vec = read_vector_bin(vec_src, cols)
        else:
            # Generate random vector if not found
            print(f"Warning: vector.bin not found for {mtx_src}, generating random vector")
            vec = gen_random_vector(cols, j * 1000)
        
        ensure_dir(out_dir)
        if fixed_name:
            name_prefix = ""
        else:
            idx_str = f"{j:0{digits}d}"
            name_prefix = f"{idx_str}_"
        
        mat_dst, vec_dst = write_one(
            out_dir, name_prefix, rows, cols, entries, vec, is_symmetric, overwrite
        )
        
        base_name = mat_dst.replace(".mtx", "")
        write_meta(base_name, {
            "rows": rows,
            "cols": cols,
            "nnz": nnz,
            "symmetric": is_symmetric,
            "source": os.path.abspath(mtx_src),
        })


# -------------------------
# CLI
# -------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SPMV input generator (Matrix Market + vector)")
    sub = p.add_subparsers(dest='cmd', required=True)

    sp_syn = sub.add_parser('synthetic', help='Generate synthetic sparse matrix data')
    sp_syn.add_argument('--out-dir', required=True, help='Output directory (created recursively)')
    sp_syn.add_argument('--count', type=int, default=1, help='Number of datasets to generate')
    sp_syn.add_argument('--preset', choices=list(PRESETS.keys()), default=None, help='Preset size')
    sp_syn.add_argument('--rows', type=int, default=None, help='Override number of rows')
    sp_syn.add_argument('--cols', type=int, default=None, help='Override number of columns')
    sp_syn.add_argument('--nnz', type=int, default=None, help='Override number of nonzeros')
    sp_syn.add_argument('--kind', choices=['random', 'diagonal', 'banded'], default='random',
                        help='Sparse matrix pattern')
    sp_syn.add_argument('--symmetric', action='store_true', help='Generate symmetric matrix')
    sp_syn.add_argument('--seed', type=int, default=0, help='Base RNG seed')
    sp_syn.add_argument('--start-index', type=int, default=0, help='Starting index')
    sp_syn.add_argument('--digits', type=int, default=4, help='Zero-padding width')
    sp_syn.add_argument('--overwrite', action='store_true', help='Allow overwriting existing files')
    sp_syn.add_argument('--fixed-name', action='store_true',
                        help='Use fixed naming (no index prefix)')

    sp_src = sub.add_parser('from-sources', help='Copy/convert existing Matrix Market files')
    sp_src.add_argument('--src-glob', required=True, help='Glob for existing .mtx files')
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
            rows=args.rows,
            cols=args.cols,
            nnz=args.nnz,
            kind=args.kind,
            start_index=args.start_index,
            digits=args.digits,
            overwrite=args.overwrite,
            fixed_name=args.fixed_name,
            symmetric=args.symmetric,
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

