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
# Format: LBM .of (obstacle field)
#   Text file with SIZE_Z*(SIZE_Y+1) lines
#   Each line has SIZE_X characters: '.' = fluid, 'o' = obstacle
#   One blank line between Y slices
# -------------------------

def encode_obstacle_field(size_x: int, size_y: int, size_z: int, obstacles: list) -> str:
    """
    obstacles: list of (x, y, z) tuples marking obstacle cells
    Returns multi-line text suitable for .of format
    """
    obs_set = set(obstacles)
    lines = []
    for z in range(size_z):
        for y in range(size_y):
            row = []
            for x in range(size_x):
                if (x, y, z) in obs_set:
                    row.append('o')
                else:
                    row.append('.')
            lines.append(''.join(row))
        lines.append('')  # blank line between Z slices
    return '\n'.join(lines)


def validate_dims(size_x: int, size_y: int, size_z: int) -> None:
    if size_x <= 0 or size_y <= 0 or size_z <= 0:
        raise ValueError("All dimensions must be positive")
    if size_x > 1000 or size_y > 1000 or size_z > 1000:
        raise ValueError("Dimensions too large (max 1000 per axis)")


# -------------------------
# Presets
# -------------------------

PRESETS: Dict[str, Tuple[int, int, int]] = {
    # name: (SIZE_X, SIZE_Y, SIZE_Z)
    "mini":   (60, 60, 80),
    "short":  (120, 120, 150),
    "medium": (180, 180, 200),
    "long":   (240, 240, 250),
    "extra-long": (300, 300, 300),
}


# -------------------------
# Synthetic generators
# -------------------------

def gen_empty(size_x: int, size_y: int, size_z: int, seed: int) -> list:
    """Empty domain (all fluid)"""
    return []


def gen_random_obstacles(size_x: int, size_y: int, size_z: int, seed: int, density: float = 0.05) -> list:
    """Random obstacles with given density"""
    rng = random.Random(seed)
    obs = []
    for z in range(size_z):
        for y in range(size_y):
            for x in range(size_x):
                if rng.random() < density:
                    obs.append((x, y, z))
    return obs


def gen_channel_walls(size_x: int, size_y: int, size_z: int, seed: int) -> list:
    """Top and bottom walls for channel flow"""
    obs = []
    for z in range(size_z):
        for x in range(size_x):
            obs.append((x, 0, z))  # bottom
            obs.append((x, size_y - 1, z))  # top
    return obs


def gen_box_cavity(size_x: int, size_y: int, size_z: int, seed: int) -> list:
    """Box boundaries with open top (lid-driven cavity style)"""
    obs = []
    for z in range(size_z):
        for y in range(size_y):
            for x in range(size_x):
                # walls on sides and bottom
                if x == 0 or x == size_x - 1 or y == 0 or z == 0 or z == size_z - 1:
                    obs.append((x, y, z))
    return obs


def write_one(out_dir: str, size_x: int, size_y: int, size_z: int, index: int, digits: int,
              data: str, overwrite: bool, fixed_name: bool) -> str:
    ensure_dir(out_dir)
    idx_str = f"{index:0{digits}d}"
    if fixed_name:
        filename = f"{size_x}_{size_y}_{size_z}_ldc.of"
    else:
        filename = f"lbm_{size_x}x{size_y}x{size_z}_{idx_str}.of"
    path = safe_join(out_dir, filename, overwrite)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(data)
    return path


def generate_synthetic(out_dir: str,
                       count: int,
                       preset: Optional[str],
                       seed: int,
                       size_x: Optional[int], size_y: Optional[int], size_z: Optional[int],
                       kind: str,
                       density: float,
                       start_index: int,
                       digits: int,
                       overwrite: bool,
                       fixed_name: bool) -> None:
    if preset is not None:
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset: {preset}")
        p_x, p_y, p_z = PRESETS[preset]
    else:
        p_x = p_y = p_z = None  # type: ignore

    size_x = size_x if size_x is not None else p_x
    size_y = size_y if size_y is not None else p_y
    size_z = size_z if size_z is not None else p_z

    if size_x is None or size_y is None or size_z is None:
        raise ValueError("Must provide preset or explicit --size-x/--size-y/--size-z")

    validate_dims(size_x, size_y, size_z)

    for i in range(start_index, start_index + count):
        if kind == 'empty':
            obs = gen_empty(size_x, size_y, size_z, seed + i)
        elif kind == 'random':
            obs = gen_random_obstacles(size_x, size_y, size_z, seed + i, density=density)
        elif kind == 'channel':
            obs = gen_channel_walls(size_x, size_y, size_z, seed + i)
        elif kind == 'cavity':
            obs = gen_box_cavity(size_x, size_y, size_z, seed + i)
        else:
            raise ValueError("Unknown kind. Use: empty|random|channel|cavity")

        data = encode_obstacle_field(size_x, size_y, size_z, obs)
        out_path = write_one(out_dir, size_x, size_y, size_z, i, digits, data, overwrite, fixed_name)
        write_meta(os.path.splitext(out_path)[0], {
            "size_x": size_x,
            "size_y": size_y,
            "size_z": size_z,
            "kind": kind,
            "seed": seed + i,
            "obstacle_count": len(obs),
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
        with open(src, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        lines = content.splitlines()
        if not lines:
            raise ValueError(f"Empty file: {src}")
        # Infer dimensions from first non-empty line
        first_line = next((l for l in lines if l.strip()), None)
        if not first_line:
            raise ValueError(f"No non-empty lines in {src}")
        size_x = len(first_line.strip())
        # Count non-blank lines to estimate size_z*(size_y+1)
        non_blank = sum(1 for l in lines if l.strip())
        # Heuristic: assume blank line after each y-slice
        # total lines = size_z * (size_y + 1)
        # For standard 120_120_150: 150*(120+1)=18150 lines
        # This is an approximation; real validation would parse structure
        if non_blank < 10:
            raise ValueError(f"Too few lines in {src}")
        # Simple heuristic: assume square x,y and estimate z
        size_y_est = int((non_blank / 150) ** 0.5) if non_blank > 1000 else 120
        size_z_est = non_blank // (size_y_est + 1) if size_y_est > 0 else 150
        size_y = size_y_est
        size_z = size_z_est
        # Just copy verbatim
        ensure_dir(out_dir)
        idx_str = f"{j:0{digits}d}"
        if fixed_name:
            filename = f"{size_x}_{size_y}_{size_z}_ldc.of"
        else:
            filename = f"lbm_{size_x}x{size_y}x{size_z}_{idx_str}.of"
        dst = safe_join(out_dir, filename, overwrite)
        with open(dst, 'w', encoding='utf-8') as fo:
            fo.write(content)
        write_meta(os.path.splitext(dst)[0], {
            "size_x": size_x,
            "size_y_est": size_y,
            "size_z_est": size_z,
            "source": os.path.abspath(src),
        })


# -------------------------
# CLI
# -------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LBM obstacle field generator (.of)")
    sub = p.add_subparsers(dest='cmd', required=True)

    sp_syn = sub.add_parser('synthetic', help='Generate synthetic obstacle fields')
    sp_syn.add_argument('--out-dir', required=True, help='Output directory (created recursively)')
    sp_syn.add_argument('--count', type=int, default=1, help='Number of files to generate')
    sp_syn.add_argument('--preset', choices=list(PRESETS.keys()), default=None, help='Preset size')
    sp_syn.add_argument('--size-x', type=int, default=None, help='Override SIZE_X')
    sp_syn.add_argument('--size-y', type=int, default=None, help='Override SIZE_Y')
    sp_syn.add_argument('--size-z', type=int, default=None, help='Override SIZE_Z')
    sp_syn.add_argument('--kind', choices=['empty', 'random', 'channel', 'cavity'], default='empty',
                        help='Obstacle pattern')
    sp_syn.add_argument('--density', type=float, default=0.05, help='Obstacle density for random kind')
    sp_syn.add_argument('--seed', type=int, default=0, help='Base RNG seed')
    sp_syn.add_argument('--start-index', type=int, default=0, help='Starting index')
    sp_syn.add_argument('--digits', type=int, default=4, help='Zero-padding width')
    sp_syn.add_argument('--overwrite', action='store_true', help='Allow overwriting existing files')
    sp_syn.add_argument('--fixed-name', action='store_true',
                        help='Use fixed LBM naming like 120_120_150_ldc.of')

    sp_src = sub.add_parser('from-sources', help='Copy/validate existing .of files')
    sp_src.add_argument('--src-glob', required=True, help='Glob for existing .of files')
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
            size_x=args.size_x, size_y=args.size_y, size_z=args.size_z,
            kind=args.kind,
            density=args.density,
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

