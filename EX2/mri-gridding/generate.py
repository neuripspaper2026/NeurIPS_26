#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import json
import argparse
import struct
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
# Format: MRI-gridding .uks + .uks.data
#   .uks: text config with parameters
#   .uks.data: binary samples (struct ReconstructionSample):
#     kX, kY, kZ: float (3x4 bytes)
#     real, imag: float (2x4 bytes)
#     sdc: float (1x4 bytes)
#     Total: 24 bytes per sample
# -------------------------

def write_uks_config(path: str, num_samples: int, kmax: Tuple[float, float, float],
                     acq_size: Tuple[int, int, int], recon_size: Tuple[int, int, int],
                     grid_size: Tuple[int, int, int], oversample: float, kernel_width: float,
                     use_lut: int) -> None:
    lines = [
        f"aquisition.numsamples={num_samples}\n",
        f"aquisition.kmax={kmax[0]} {kmax[1]} {kmax[2]}\n",
        f"aquisition.matrixSize={acq_size[0]} {acq_size[1]} {acq_size[2]}\n",
        f"reconstruction.matrixSize={recon_size[0]} {recon_size[1]} {recon_size[2]}\n",
        f"gridding.matrixSize={grid_size[0]} {grid_size[1]} {grid_size[2]}\n",
        f"gridding.oversampling={oversample}\n",
        f"kernel.width={kernel_width}\n",
        f"kernel.useLUT={use_lut}\n",
    ]
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(lines)


def write_uks_data(path: str, samples: list) -> None:
    """
    samples: list of (kX, kY, kZ, real, imag, sdc) tuples (6 floats)
    """
    with open(path, 'wb') as f:
        for kx, ky, kz, r, im, sdc in samples:
            f.write(struct.pack('<ffffff', kx, ky, kz, r, im, sdc))


def validate_dims(num_samples: int, grid_size: Tuple[int, int, int]) -> None:
    if num_samples <= 0:
        raise ValueError("Number of samples must be positive")
    if any(d <= 0 for d in grid_size):
        raise ValueError("Grid size must have positive dimensions")
    if num_samples > 50_000_000:
        raise ValueError("Too many samples (max 50M)")


# -------------------------
# Presets
# -------------------------

PRESETS: Dict[str, Tuple[int, Tuple[int, int, int]]] = {
    # name: (num_samples, grid_size)
    "mini":   (100_000,   (128, 128, 128)),
    "small":  (2_655_910, (256, 256, 256)),
    "medium": (5_000_000, (384, 384, 384)),
    "large":  (10_000_000, (512, 512, 512)),
    "extra-large": (20_000_000, (640, 640, 640)),
}


# -------------------------
# Synthetic generators
# -------------------------

def gen_uniform_samples(num_samples: int, kmax: Tuple[float, float, float], seed: int) -> list:
    """Generate uniformly random samples in k-space"""
    rng = random.Random(seed)
    samples = []
    for _ in range(num_samples):
        kx = rng.uniform(-kmax[0], kmax[0])
        ky = rng.uniform(-kmax[1], kmax[1])
        kz = rng.uniform(-kmax[2], kmax[2])
        real = rng.uniform(-1.0, 1.0)
        imag = rng.uniform(-1.0, 1.0)
        sdc = rng.uniform(0.5, 1.5)
        samples.append((kx, ky, kz, real, imag, sdc))
    return samples


def gen_radial_samples(num_samples: int, kmax: Tuple[float, float, float], seed: int) -> list:
    """Generate radial trajectory samples"""
    rng = random.Random(seed)
    samples = []
    for _ in range(num_samples):
        # Radial: random angle, random radius
        theta = rng.uniform(0, 2 * 3.14159)
        phi = rng.uniform(0, 3.14159)
        r = rng.uniform(0, min(kmax))
        kx = r * rng.uniform(0.8, 1.2)
        ky = r * rng.uniform(0.8, 1.2)
        kz = r * rng.uniform(0.8, 1.2)
        # Clamp to kmax
        kx = max(-kmax[0], min(kmax[0], kx))
        ky = max(-kmax[1], min(kmax[1], ky))
        kz = max(-kmax[2], min(kmax[2], kz))
        real = rng.gauss(0, 0.3)
        imag = rng.gauss(0, 0.3)
        sdc = 1.0
        samples.append((kx, ky, kz, real, imag, sdc))
    return samples


def write_one(out_dir: str, name: str, num_samples: int, kmax: Tuple[float, float, float],
              acq_size: Tuple[int, int, int], recon_size: Tuple[int, int, int],
              grid_size: Tuple[int, int, int], oversample: float, kernel_width: float,
              use_lut: int, samples: list, overwrite: bool) -> Tuple[str, str]:
    ensure_dir(out_dir)
    uks_path = safe_join(out_dir, f"{name}.uks", overwrite)
    data_path = safe_join(out_dir, f"{name}.uks.data", overwrite)
    write_uks_config(uks_path, num_samples, kmax, acq_size, recon_size, grid_size,
                     oversample, kernel_width, use_lut)
    write_uks_data(data_path, samples)
    return uks_path, data_path


def generate_synthetic(out_dir: str,
                       count: int,
                       preset: Optional[str],
                       seed: int,
                       num_samples: Optional[int],
                       grid_size_x: Optional[int], grid_size_y: Optional[int], grid_size_z: Optional[int],
                       kind: str,
                       start_index: int,
                       digits: int,
                       overwrite: bool,
                       fixed_name: bool) -> None:
    if preset is not None:
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset: {preset}")
        p_num, p_grid = PRESETS[preset]
    else:
        p_num = None
        p_grid = (256, 256, 256)  # default

    num_samples = num_samples if num_samples is not None else p_num
    grid_x = grid_size_x if grid_size_x is not None else p_grid[0]
    grid_y = grid_size_y if grid_size_y is not None else p_grid[1]
    grid_z = grid_size_z if grid_size_z is not None else p_grid[2]

    if num_samples is None:
        raise ValueError("Must provide preset or explicit --num-samples")

    grid_size = (grid_x, grid_y, grid_z)
    validate_dims(num_samples, grid_size)

    # Fixed parameters (can be made customizable if needed)
    kmax = (150.0, 150.0, 150.0)
    acq_size = (60, 60, 60)
    recon_size = (60, 60, 60)
    oversample = 5.0
    kernel_width = 5.0
    use_lut = 1

    for i in range(start_index, start_index + count):
        if kind == 'uniform':
            samples = gen_uniform_samples(num_samples, kmax, seed + i)
        elif kind == 'radial':
            samples = gen_radial_samples(num_samples, kmax, seed + i)
        else:
            raise ValueError("Unknown kind. Use: uniform|radial")

        if fixed_name:
            name = preset if preset else "data"
        else:
            idx_str = f"{i:0{digits}d}"
            name = f"mrig_{idx_str}"

        uks_path, data_path = write_one(
            out_dir, name, num_samples, kmax, acq_size, recon_size, grid_size,
            oversample, kernel_width, use_lut, samples, overwrite
        )
        write_meta(os.path.splitext(uks_path)[0], {
            "num_samples": num_samples,
            "grid_size": grid_size,
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
        # Read .uks config
        with open(src, 'r', encoding='utf-8') as f:
            content = f.read()
        # Also copy .uks.data if exists
        data_src = src + ".data"
        if not os.path.exists(data_src):
            raise FileNotFoundError(f"Missing data file: {data_src}")
        with open(data_src, 'rb') as f:
            data_content = f.read()
        # Write to output
        ensure_dir(out_dir)
        if fixed_name:
            base = os.path.splitext(os.path.basename(src))[0]
            name = base
        else:
            idx_str = f"{j:0{digits}d}"
            name = f"mrig_{idx_str}"
        uks_dst = safe_join(out_dir, f"{name}.uks", overwrite)
        data_dst = safe_join(out_dir, f"{name}.uks.data", overwrite)
        with open(uks_dst, 'w', encoding='utf-8') as f:
            f.write(content)
        with open(data_dst, 'wb') as f:
            f.write(data_content)
        write_meta(os.path.splitext(uks_dst)[0], {
            "source": os.path.abspath(src),
        })


# -------------------------
# CLI
# -------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MRI-gridding input generator (.uks)")
    sub = p.add_subparsers(dest='cmd', required=True)

    sp_syn = sub.add_parser('synthetic', help='Generate synthetic MRI samples')
    sp_syn.add_argument('--out-dir', required=True, help='Output directory (created recursively)')
    sp_syn.add_argument('--count', type=int, default=1, help='Number of datasets to generate')
    sp_syn.add_argument('--preset', choices=list(PRESETS.keys()), default=None, help='Preset size')
    sp_syn.add_argument('--num-samples', type=int, default=None, help='Override number of samples')
    sp_syn.add_argument('--grid-size-x', type=int, default=None, help='Override grid X size')
    sp_syn.add_argument('--grid-size-y', type=int, default=None, help='Override grid Y size')
    sp_syn.add_argument('--grid-size-z', type=int, default=None, help='Override grid Z size')
    sp_syn.add_argument('--kind', choices=['uniform', 'radial'], default='uniform',
                        help='Sample distribution')
    sp_syn.add_argument('--seed', type=int, default=0, help='Base RNG seed')
    sp_syn.add_argument('--start-index', type=int, default=0, help='Starting index')
    sp_syn.add_argument('--digits', type=int, default=4, help='Zero-padding width')
    sp_syn.add_argument('--overwrite', action='store_true', help='Allow overwriting existing files')
    sp_syn.add_argument('--fixed-name', action='store_true',
                        help='Use fixed naming like small.uks')

    sp_src = sub.add_parser('from-sources', help='Copy/validate existing .uks files')
    sp_src.add_argument('--src-glob', required=True, help='Glob for existing .uks files')
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
            num_samples=args.num_samples,
            grid_size_x=args.grid_size_x, grid_size_y=args.grid_size_y, grid_size_z=args.grid_size_z,
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

