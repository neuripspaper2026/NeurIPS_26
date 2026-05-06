#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import json
import argparse
import struct
from typing import Tuple, Optional, Dict
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
# Format: histo img.bin
#   Header: 4 x uint32 little-endian: img_w, img_h, histo_w, histo_h
#   Body  : img_w*img_h x uint32 little-endian pixel values in [0, histo_w*histo_h-1]
# -------------------------

def encode_histo_input(img: np.ndarray, histo_w: int, histo_h: int) -> bytes:
    if img.dtype != np.uint32:
        img = img.astype(np.uint32, copy=False)
    img_h, img_w = img.shape
    header = struct.pack('<IIII', img_w, img_h, histo_w, histo_h)
    return header + img.tobytes(order='C')


def validate_dims(img_w: int, img_h: int, histo_w: int, histo_h: int) -> None:
    if img_w <= 0 or img_h <= 0:
        raise ValueError("Image width/height must be positive")
    if histo_w <= 0 or histo_h <= 0:
        raise ValueError("Histogram width/height must be positive")
    total_bins = histo_w * histo_h
    if total_bins > (1 << 31):
        raise ValueError("Total bins too large")


# -------------------------
# Presets
# -------------------------

PRESETS: Dict[str, Tuple[int, int, int, int]] = {
    # name: (img_w, img_h, histo_w, histo_h)
    "mini": (256, 256, 256, 1),
    "default": (1024, 1024, 1024, 1),
    "medium": (2048, 2048, 1024, 1),
    "large": (4096, 4096, 1024, 1),
    "extra-large": (8192, 8192, 1024, 1),
}


# -------------------------
# Synthetic generators
# -------------------------

def gen_uniform(img_w: int, img_h: int, bins: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, bins, size=(img_h, img_w), dtype=np.uint32)


def gen_hotspots(img_w: int, img_h: int, bins: int, seed: int, spots: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty((img_h, img_w), dtype=np.uint32)
    centers = rng.integers(0, bins, size=spots, dtype=np.uint32)
    weights = rng.random(spots)
    weights = weights / weights.sum()
    # Mixture of hotspots and uniform background
    back_p = 0.2
    mask = rng.random((img_h, img_w)) < back_p
    out[mask] = rng.integers(0, bins, size=mask.sum(), dtype=np.uint32)
    # For non-background, pick nearest hotspot index by weight
    idxs = rng.choice(spots, size=(img_h, img_w), p=weights)
    out[~mask] = centers[idxs[~mask]]
    return out


def write_one(out_dir: str, index: int, digits: int, data: bytes, overwrite: bool, fixed_name: bool) -> str:
    ensure_dir(out_dir)
    idx_str = f"{index:0{digits}d}"
    filename = "img.bin" if fixed_name else f"img_{idx_str}.bin"
    path = safe_join(out_dir, filename, overwrite)
    with open(path, 'wb') as f:
        f.write(data)
    return path


def generate_synthetic(out_dir: str,
                       count: int,
                       preset: str,
                       seed: int,
                       img_w: Optional[int], img_h: Optional[int],
                       histo_w: Optional[int], histo_h: Optional[int],
                       kind: str,
                       start_index: int,
                       digits: int,
                       overwrite: bool,
                       fixed_name: bool) -> None:
    if preset is not None:
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset: {preset}")
        p_img_w, p_img_h, p_hw, p_hh = PRESETS[preset]
    else:
        p_img_w = p_img_h = p_hw = p_hh = None  # type: ignore

    img_w = img_w if img_w is not None else p_img_w
    img_h = img_h if img_h is not None else p_img_h
    histo_w = histo_w if histo_w is not None else p_hw
    histo_h = histo_h if histo_h is not None else p_hh

    if img_w is None or img_h is None or histo_w is None or histo_h is None:
        raise ValueError("Must provide preset or explicit --img-width/--img-height and --histo-width/--histo-height")

    validate_dims(img_w, img_h, histo_w, histo_h)
    bins = histo_w * histo_h

    for i in range(start_index, start_index + count):
        if kind == 'uniform':
            img = gen_uniform(img_w, img_h, bins, seed + i)
        elif kind == 'hotspots':
            img = gen_hotspots(img_w, img_h, bins, seed + i)
        else:
            raise ValueError("Unknown kind. Use: uniform|hotspots")

        data = encode_histo_input(img, histo_w, histo_h)
        out_path = write_one(out_dir, i, digits, data, overwrite, fixed_name)
        # sidecar meta
        write_meta(os.path.splitext(out_path)[0], {
            "img_width": img_w,
            "img_height": img_h,
            "histo_width": histo_w,
            "histo_height": histo_h,
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
            header = f.read(16)
            if len(header) != 16:
                raise ValueError(f"Invalid header length: {src}")
            img_w, img_h, hw, hh = struct.unpack('<IIII', header)
            validate_dims(img_w, img_h, hw, hh)
            body = f.read()
            expected = img_w * img_h * 4
            if len(body) != expected:
                raise ValueError(f"Invalid body length in {src}: got {len(body)}, expect {expected}")
            data = header + body
        ensure_dir(out_dir)
        idx_str = f"{j:0{digits}d}"
        filename = "img.bin" if fixed_name else f"img_{idx_str}.bin"
        dst = safe_join(out_dir, filename, overwrite)
        with open(dst, 'wb') as fo:
            fo.write(data)
        write_meta(os.path.splitext(dst)[0], {
            "img_width": img_w,
            "img_height": img_h,
            "histo_width": hw,
            "histo_height": hh,
            "source": os.path.abspath(src),
        })


# -------------------------
# CLI
# -------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Histogram input generator (img.bin)")
    sub = p.add_subparsers(dest='cmd', required=True)

    sp_syn = sub.add_parser('synthetic', help='Generate synthetic histogram inputs')
    sp_syn.add_argument('--out-dir', required=True, help='Output directory (created recursively)')
    sp_syn.add_argument('--count', type=int, default=1, help='Number of files to generate')
    sp_syn.add_argument('--preset', choices=list(PRESETS.keys()), default='default', help='Preset size')
    sp_syn.add_argument('--img-width', type=int, default=None, help='Override image width')
    sp_syn.add_argument('--img-height', type=int, default=None, help='Override image height')
    sp_syn.add_argument('--histo-width', type=int, default=None, help='Override histogram width')
    sp_syn.add_argument('--histo-height', type=int, default=None, help='Override histogram height')
    sp_syn.add_argument('--kind', choices=['uniform', 'hotspots'], default='uniform', help='Synthetic distribution')
    sp_syn.add_argument('--seed', type=int, default=0, help='Base RNG seed')
    sp_syn.add_argument('--start-index', type=int, default=0, help='Starting index')
    sp_syn.add_argument('--digits', type=int, default=4, help='Zero-padding width')
    sp_syn.add_argument('--overwrite', action='store_true', help='Allow overwriting existing files')
    sp_syn.add_argument('--fixed-name', action='store_true', help='Write file as img.bin without index')

    sp_src = sub.add_parser('from-sources', help='Copy/validate existing img.bin files')
    sp_src.add_argument('--src-glob', required=True, help='Glob for existing img.bin files')
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
            img_w=args.img_width, img_h=args.img_height,
            histo_w=args.histo_width, histo_h=args.histo_height,
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


