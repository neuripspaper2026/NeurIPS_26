#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import json
import argparse
import struct
from typing import Tuple, Optional, Dict
import random
import numpy as np


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
# Format: MRI-Q binary input
#   Binary format (little-endian):
#     numK: int32 (number of K-space samples)
#     numX: int32 (number of X-space samples)
#     kx[numK]: float32 array
#     ky[numK]: float32 array
#     kz[numK]: float32 array
#     x[numX]: float32 array
#     y[numX]: float32 array
#     z[numX]: float32 array
#     phiR[numK]: float32 array (real part)
#     phiI[numK]: float32 array (imaginary part)
# -------------------------

def encode_mri_q_input(numK: int, numX: int,
                       kx: np.ndarray, ky: np.ndarray, kz: np.ndarray,
                       x: np.ndarray, y: np.ndarray, z: np.ndarray,
                       phiR: np.ndarray, phiI: np.ndarray) -> bytes:
    """Encode MRI-Q input data to binary format"""
    data = bytearray()
    # Write header: numK, numX
    data.extend(struct.pack('<i', numK))
    data.extend(struct.pack('<i', numX))
    # Write K-space coordinates
    data.extend(kx.astype(np.float32).tobytes())
    data.extend(ky.astype(np.float32).tobytes())
    data.extend(kz.astype(np.float32).tobytes())
    # Write X-space coordinates
    data.extend(x.astype(np.float32).tobytes())
    data.extend(y.astype(np.float32).tobytes())
    data.extend(z.astype(np.float32).tobytes())
    # Write phi values
    data.extend(phiR.astype(np.float32).tobytes())
    data.extend(phiI.astype(np.float32).tobytes())
    return bytes(data)


def decode_mri_q_input(data: bytes) -> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray,
                                               np.ndarray, np.ndarray, np.ndarray,
                                               np.ndarray, np.ndarray]:
    """Decode MRI-Q input data from binary format"""
    offset = 0
    numK = struct.unpack('<i', data[offset:offset+4])[0]
    offset += 4
    numX = struct.unpack('<i', data[offset:offset+4])[0]
    offset += 4
    
    kx = np.frombuffer(data[offset:offset+numK*4], dtype=np.float32)
    offset += numK * 4
    ky = np.frombuffer(data[offset:offset+numK*4], dtype=np.float32)
    offset += numK * 4
    kz = np.frombuffer(data[offset:offset+numK*4], dtype=np.float32)
    offset += numK * 4
    
    x = np.frombuffer(data[offset:offset+numX*4], dtype=np.float32)
    offset += numX * 4
    y = np.frombuffer(data[offset:offset+numX*4], dtype=np.float32)
    offset += numX * 4
    z = np.frombuffer(data[offset:offset+numX*4], dtype=np.float32)
    offset += numX * 4
    
    phiR = np.frombuffer(data[offset:offset+numK*4], dtype=np.float32)
    offset += numK * 4
    phiI = np.frombuffer(data[offset:offset+numK*4], dtype=np.float32)
    
    return numK, numX, kx, ky, kz, x, y, z, phiR, phiI


def validate_dims(numK: int, numX: int) -> None:
    if numK <= 0 or numX <= 0:
        raise ValueError("numK and numX must be positive")
    if numK > 100_000_000:
        raise ValueError("numK too large (max 100M)")
    if numX > 100_000_000:
        raise ValueError("numX too large (max 100M)")


# -------------------------
# Presets
# -------------------------

PRESETS: Dict[str, Tuple[int, int]] = {
    # name: (numK, numX)
    "mini":   (1024, 8192),          # 1K k-space, 8K x-space
    "small":  (3072, 32768),         # 3K k-space, 32K x-space (32^3)
    "medium": (8192, 131072),        # 8K k-space, 128K x-space (32x32x128)
    "large":  (24576, 262144),       # 24K k-space, 256K x-space (64^3)
    "extra-large": (49152, 524288),  # 48K k-space, 512K x-space (80^3)
}


# -------------------------
# Synthetic generators
# -------------------------

def gen_uniform_kspace(numK: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate uniformly random K-space samples"""
    rng = np.random.RandomState(seed)
    kx = rng.uniform(-0.5, 0.5, numK).astype(np.float32)
    ky = rng.uniform(-0.5, 0.5, numK).astype(np.float32)
    kz = rng.uniform(-0.5, 0.5, numK).astype(np.float32)
    return kx, ky, kz


def gen_grid_xspace(numX: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate grid-based X-space samples"""
    # Approximate cube root
    side = int(round(numX ** (1/3)))
    if side ** 3 != numX:
        # If not a perfect cube, use linear arrangement
        rng = np.random.RandomState(seed)
        x = rng.uniform(-0.5, 0.5, numX).astype(np.float32)
        y = rng.uniform(-0.5, 0.5, numX).astype(np.float32)
        z = rng.uniform(-0.5, 0.5, numX).astype(np.float32)
        return x, y, z
    
    # Generate 3D grid
    coords = np.linspace(-0.5, 0.5, side, dtype=np.float32)
    grid = np.meshgrid(coords, coords, coords, indexing='ij')
    x = grid[0].flatten()
    y = grid[1].flatten()
    z = grid[2].flatten()
    return x, y, z


def gen_random_phi(numK: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate random complex phi values"""
    rng = np.random.RandomState(seed)
    phiR = rng.randn(numK).astype(np.float32)
    phiI = rng.randn(numK).astype(np.float32)
    return phiR, phiI


def gen_spiral_kspace(numK: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate spiral trajectory in K-space"""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 4 * np.pi, numK, dtype=np.float32)
    r = np.linspace(0, 0.5, numK, dtype=np.float32)
    theta = t + rng.uniform(-0.1, 0.1, numK).astype(np.float32)
    phi = rng.uniform(0, 2*np.pi, numK).astype(np.float32)
    
    kx = r * np.cos(theta) * np.cos(phi)
    ky = r * np.sin(theta) * np.cos(phi)
    kz = r * np.sin(phi)
    
    return kx.astype(np.float32), ky.astype(np.float32), kz.astype(np.float32)


def write_one(out_dir: str, name: str, numK: int, numX: int,
              kx: np.ndarray, ky: np.ndarray, kz: np.ndarray,
              x: np.ndarray, y: np.ndarray, z: np.ndarray,
              phiR: np.ndarray, phiI: np.ndarray,
              overwrite: bool) -> str:
    ensure_dir(out_dir)
    bin_path = safe_join(out_dir, f"{name}.bin", overwrite)
    data = encode_mri_q_input(numK, numX, kx, ky, kz, x, y, z, phiR, phiI)
    with open(bin_path, 'wb') as f:
        f.write(data)
    return bin_path


def generate_synthetic(out_dir: str,
                       count: int,
                       preset: Optional[str],
                       seed: int,
                       numK: Optional[int],
                       numX: Optional[int],
                       kind: str,
                       start_index: int,
                       digits: int,
                       overwrite: bool,
                       fixed_name: bool) -> None:
    if preset is not None:
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset: {preset}")
        p_numK, p_numX = PRESETS[preset]
    else:
        p_numK = None
        p_numX = None

    numK = numK if numK is not None else p_numK
    numX = numX if numX is not None else p_numX

    if numK is None or numX is None:
        raise ValueError("Must provide preset or explicit --numK and --numX")

    validate_dims(numK, numX)

    for i in range(start_index, start_index + count):
        if kind == 'uniform':
            kx, ky, kz = gen_uniform_kspace(numK, seed + i * 3)
            x, y, z = gen_grid_xspace(numX, seed + i * 3 + 1)
        elif kind == 'spiral':
            kx, ky, kz = gen_spiral_kspace(numK, seed + i * 3)
            x, y, z = gen_grid_xspace(numX, seed + i * 3 + 1)
        else:
            raise ValueError("Unknown kind. Use: uniform|spiral")

        phiR, phiI = gen_random_phi(numK, seed + i * 3 + 2)

        if fixed_name:
            name = preset if preset else "dataset"
        else:
            idx_str = f"{i:0{digits}d}"
            # Infer grid size from numX
            side = int(round(numX ** (1/3)))
            name = f"{side}_{side}_{side}_dataset"

        bin_path = write_one(
            out_dir, name, numK, numX, kx, ky, kz, x, y, z, phiR, phiI, overwrite
        )
        write_meta(os.path.splitext(bin_path)[0], {
            "numK": numK,
            "numX": numX,
            "kind": kind,
            "seed": seed + i,
        })


def from_sources(src_glob: str,
                 out_dir: str,
                 start_index: int,
                 digits: int,
                 overwrite: bool,
                 fixed_name: bool) -> None:
    paths = sorted(glob.glob(src_glob))
    if not paths:
        raise FileNotFoundError(f"No files matched: {src_glob}")
    for j, src in enumerate(paths, start=start_index):
        with open(src, 'rb') as f:
            data = f.read()
        numK, numX, kx, ky, kz, x, y, z, phiR, phiI = decode_mri_q_input(data)
        
        ensure_dir(out_dir)
        if fixed_name:
            base = os.path.splitext(os.path.basename(src))[0]
            name = base
        else:
            idx_str = f"{j:0{digits}d}"
            side = int(round(numX ** (1/3)))
            name = f"{side}_{side}_{side}_dataset_{idx_str}"
        
        bin_path = write_one(
            out_dir, name, numK, numX, kx, ky, kz, x, y, z, phiR, phiI, overwrite
        )
        write_meta(os.path.splitext(bin_path)[0], {
            "numK": numK,
            "numX": numX,
            "source": os.path.abspath(src),
        })


# -------------------------
# CLI
# -------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MRI-Q input generator (.bin)")
    sub = p.add_subparsers(dest='cmd', required=True)

    sp_syn = sub.add_parser('synthetic', help='Generate synthetic MRI-Q data')
    sp_syn.add_argument('--out-dir', required=True, help='Output directory (created recursively)')
    sp_syn.add_argument('--count', type=int, default=1, help='Number of datasets to generate')
    sp_syn.add_argument('--preset', choices=list(PRESETS.keys()), default=None, help='Preset size')
    sp_syn.add_argument('--numK', type=int, default=None, help='Override numK (K-space samples)')
    sp_syn.add_argument('--numX', type=int, default=None, help='Override numX (X-space samples)')
    sp_syn.add_argument('--kind', choices=['uniform', 'spiral'], default='uniform',
                        help='K-space sampling pattern')
    sp_syn.add_argument('--seed', type=int, default=0, help='Base RNG seed')
    sp_syn.add_argument('--start-index', type=int, default=0, help='Starting index')
    sp_syn.add_argument('--digits', type=int, default=4, help='Zero-padding width')
    sp_syn.add_argument('--overwrite', action='store_true', help='Allow overwriting existing files')
    sp_syn.add_argument('--fixed-name', action='store_true',
                        help='Use fixed naming based on preset')

    sp_src = sub.add_parser('from-sources', help='Copy/validate existing .bin files')
    sp_src.add_argument('--src-glob', required=True, help='Glob for existing .bin files')
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
            numK=args.numK,
            numX=args.numX,
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


