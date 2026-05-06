#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import glob
import argparse
from typing import Tuple, List, Optional
import numpy as np
from PIL import Image
import struct


# -------------------------
# Utilities
# -------------------------

def round_up_to_multiple_of_16(x: int) -> int:
    return (x + 15) // 16 * 16

def round_down_to_multiple_of_16(x: int) -> int:
    return (x // 16) * 16

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_bin_parboil16(img_u8: np.ndarray, out_path: str) -> None:
    """
    Write file in Parboil16 format expected by image.c:
    - Header: uint16 little-endian width, uint16 little-endian height
    - Pixels: width*height uint16 little-endian values (from input uint8, upcasted)
    """
    if img_u8.dtype != np.uint8:
        img_u8 = img_u8.astype(np.uint8, copy=False)
    h, w = img_u8.shape
    arr16_le = img_u8.astype(np.uint16, copy=False).astype('<u2', copy=False)
    with open(out_path, "wb") as f:
        f.write(struct.pack('<H', w))
        f.write(struct.pack('<H', h))
        f.write(arr16_le.tobytes(order="C"))

def write_shape_sidecar(out_path: str, width: int, height: int) -> None:
    # Record shape next to the bin file for clarity
    sidecar = out_path + ".shape.txt"
    with open(sidecar, "w", encoding="utf-8") as f:
        f.write(f"{width} {height}\n")


# -------------------------
# Image loading and size adaptation
# -------------------------

def load_image_to_gray_uint8(
    path: str,
    target_wh: Optional[Tuple[int, int]] = None,
    fit_mode: str = "crop"  # crop | pad | resize
) -> np.ndarray:
    """
    Load an image and convert it to grayscale uint8.
    - If target_wh is None: adapt to the nearest multiples of 16 according to fit_mode.
    - If target_wh is provided (w, h): output exactly this size.
    fit_mode:
      - crop: crop to the nearest multiples of 16 or to target_wh
      - pad:  pad with black to the nearest multiples of 16 or to target_wh
      - resize: resample to the nearest multiples of 16 or to target_wh (may change aspect ratio)
    """
    im = Image.open(path).convert("L")
    src_w, src_h = im.size

    if target_wh is None:
        # Auto align to multiples of 16
        if fit_mode == "crop":
            new_w = max(16, round_down_to_multiple_of_16(src_w))
            new_h = max(16, round_down_to_multiple_of_16(src_h))
            im = im.crop((0, 0, new_w, new_h))
        elif fit_mode == "pad":
            new_w = max(16, round_up_to_multiple_of_16(src_w))
            new_h = max(16, round_up_to_multiple_of_16(src_h))
            canvas = Image.new("L", (new_w, new_h), color=0)
            canvas.paste(im, (0, 0))
            im = canvas
        elif fit_mode == "resize":
            new_w = max(16, round_up_to_multiple_of_16(src_w))
            new_h = max(16, round_up_to_multiple_of_16(src_h))
            im = im.resize((new_w, new_h), Image.BICUBIC)
        else:
            raise ValueError("fit_mode must be one of crop/pad/resize")
    else:
        tgt_w, tgt_h = target_wh
        # Target size must be multiples of 16
        if tgt_w % 16 != 0 or tgt_h % 16 != 0:
            raise ValueError("target_wh must be multiples of 16")
        if fit_mode == "crop":
            # Prefer keeping the top-left region
            im = im.crop((0, 0, min(src_w, tgt_w), min(src_h, tgt_h)))
            canvas = Image.new("L", (tgt_w, tgt_h), color=0)
            canvas.paste(im, (0, 0))
            im = canvas
        elif fit_mode == "pad":
            canvas = Image.new("L", (tgt_w, tgt_h), color=0)
            canvas.paste(im, (0, 0))
            im = canvas
        elif fit_mode == "resize":
            im = im.resize((tgt_w, tgt_h), Image.BICUBIC)
        else:
            raise ValueError("fit_mode must be one of crop/pad/resize")

    return np.array(im, dtype=np.uint8)


# -------------------------
# Synthetic generators
# -------------------------

def gen_checkerboard(width: int, height: int, tile: int = 16) -> np.ndarray:
    y = np.arange(height)[:, None]
    x = np.arange(width)[None, :]
    board = ((x // tile + y // tile) % 2) * 255
    return board.astype(np.uint8)

def gen_gradient(width: int, height: int, horizontal: bool = True) -> np.ndarray:
    if horizontal:
        grad = np.linspace(0, 255, width, dtype=np.float32)[None, :].repeat(height, axis=0)
    else:
        grad = np.linspace(0, 255, height, dtype=np.float32)[:, None].repeat(width, axis=1)
    return np.clip(grad, 0, 255).astype(np.uint8)

def gen_noise(width: int, height: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width), dtype=np.uint8)

def overlay_rectangles(
    base: np.ndarray,
    rects: List[Tuple[int, int, int, int, int]]  # (x, y, w, h, val)
) -> np.ndarray:
    img = base.copy()
    for x, y, w, h, val in rects:
        x2 = min(x + w, img.shape[1])
        y2 = min(y + h, img.shape[0])
        img[y:y2, x:x2] = np.uint8(val)
    return img

def shift_image_zero_pad(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    h, w = img.shape
    out = np.zeros_like(img, dtype=np.uint8)
    x_from = max(0, -dx)
    y_from = max(0, -dy)
    x_to = min(w, w - dx)
    y_to = min(h, h - dy)
    if x_to > x_from and y_to > y_from:
        out[y_from + dy:y_to + dy, x_from + dx:x_to + dx] = img[y_from:y_to, x_from:x_to]
    return out

def make_synthetic_pair(
    width: int,
    height: int,
    kind: str = "shift",          # shift | noise | checkerboard | gradient | blocks_mix
    seed: int = 0,
    dx: int = 4,
    dy: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (ref, cur), both uint8 grayscale arrays of shape (H, W).
    """
    if width % 16 != 0 or height % 16 != 0:
        raise ValueError("Dimensions must be multiples of 16")

    if kind == "shift":
        ref = gen_checkerboard(width, height, tile=16)
        cur = shift_image_zero_pad(ref, dx=dx, dy=dy)
    elif kind == "noise":
        ref = gen_noise(width, height, seed=seed)
        cur = gen_noise(width, height, seed=seed + 1)
    elif kind == "checkerboard":
        ref = gen_checkerboard(width, height, tile=16)
        cur = gen_checkerboard(width, height, tile=24)  # 轻微结构变化
    elif kind == "gradient":
        ref = gen_gradient(width, height, horizontal=True)
        cur = gen_gradient(width, height, horizontal=False)
    elif kind == "blocks_mix":
        base = gen_gradient(width, height, horizontal=True)
        ref = overlay_rectangles(base, [
            (width // 8, height // 8, width // 6, height // 6, 30),
            (width // 2, height // 3, width // 5, height // 5, 220),
        ])
        cur = shift_image_zero_pad(ref, dx=dx, dy=dy)
    else:
        raise ValueError("Unknown kind. Use one of: shift|noise|checkerboard|gradient|blocks_mix")

    return ref.astype(np.uint8), cur.astype(np.uint8)


# -------------------------
# Build pair from two images
# -------------------------

def pair_from_two_images(
    ref_img_path: str,
    cur_img_path: str,
    target_wh: Optional[Tuple[int, int]],
    fit_mode: str = "crop"
) -> Tuple[np.ndarray, np.ndarray]:
    ref = load_image_to_gray_uint8(ref_img_path, target_wh=target_wh, fit_mode=fit_mode)
    cur = load_image_to_gray_uint8(cur_img_path, target_wh=target_wh, fit_mode=fit_mode)
    if ref.shape != cur.shape:
        raise ValueError("Reference and current images have different shapes")
    h, w = ref.shape
    if (w % 16) != 0 or (h % 16) != 0:
        raise ValueError("Output dimensions must be multiples of 16")
    return ref, cur


# -------------------------
# Writing and naming outputs
# -------------------------

def write_pair(
    ref: np.ndarray,
    cur: np.ndarray,
    out_dir: str,
    index: int,
    name_style: str = "sad",  # sad: ref.bin/cur.bin, alt: reference.bin/frame.bin
    size_tag: Optional[str] = None,
    digits: int = 4
) -> Tuple[str, str]:
    ensure_dir(out_dir)
    idx_str = f"{index:0{digits}d}"
    if name_style == "sad":
        if size_tag:
            ref_name = f"ref_{size_tag}_{idx_str}.bin"
            cur_name = f"cur_{size_tag}_{idx_str}.bin"
        else:
            ref_name = f"ref_{idx_str}.bin"
            cur_name = f"cur_{idx_str}.bin"
    else:
        if size_tag:
            ref_name = f"reference_{size_tag}_{idx_str}.bin"
            cur_name = f"frame_{size_tag}_{idx_str}.bin"
        else:
            ref_name = f"reference_{idx_str}.bin"
            cur_name = f"frame_{idx_str}.bin"

    ref_path = os.path.join(out_dir, ref_name)
    cur_path = os.path.join(out_dir, cur_name)
    save_bin_parboil16(ref, ref_path)
    save_bin_parboil16(cur, cur_path)
    # 记录尺寸
    h, w = ref.shape
    write_shape_sidecar(ref_path, w, h)
    write_shape_sidecar(cur_path, w, h)
    return ref_path, cur_path


# -------------------------
# Presets (default/large)
# -------------------------

PRESETS = {
    # Adjust as needed; keep multiples of 16
    "default": (640, 640),
    "large": (1920, 1920)
}


# -------------------------
# Size parsing utilities
# -------------------------

def parse_sizes(spec: str) -> List[Tuple[int, int]]:
    """
    Parse sizes string like "640x480,1280x736" into a list of (w, h); all must be multiples of 16.
    """
    sizes: List[Tuple[int, int]] = []
    if not spec:
        return sizes
    parts = [p.strip() for p in spec.split(',') if p.strip()]
    for p in parts:
        if 'x' not in p and 'X' not in p:
            raise ValueError(f"Invalid size item: {p}, must be WxH")
        token = p.lower()
        w_str, h_str = token.split('x', 1)
        w = int(w_str)
        h = int(h_str)
        if w % 16 != 0 or h % 16 != 0:
            raise ValueError(f"Size must be multiples of 16: {w}x{h}")
        sizes.append((w, h))
    return sizes


# -------------------------
# Batch generation entry points
# -------------------------

def generate_from_images(
    ref_glob: str,
    cur_glob: str,
    out_dir: str,
    target_preset: Optional[str] = None,
    target_wh: Optional[Tuple[int, int]] = None,
    fit_mode: str = "crop",
    name_style: str = "sad",
    start_index: int = 0,
    digits: int = 4
) -> None:
    """
    Generate paired outputs from two glob lists. Lengths must match; pairs are ordered by sorted lists.
    """
    ref_list = sorted(glob.glob(ref_glob))
    cur_list = sorted(glob.glob(cur_glob))
    if len(ref_list) != len(cur_list):
        raise ValueError("ref_glob and cur_glob have different counts")

    if target_preset:
        if target_preset not in PRESETS:
            raise ValueError(f"Unknown preset: {target_preset}")
        target_wh = PRESETS[target_preset]
    if target_wh is None:
        # If not specified, each image will be aligned to multiples of 16 according to fit_mode
        pass

    ensure_dir(out_dir)
    for i, (rp, cp) in enumerate(zip(ref_list, cur_list), start=start_index):
        ref, cur = pair_from_two_images(rp, cp, target_wh=target_wh, fit_mode=fit_mode)
        write_pair(ref, cur, out_dir, i, name_style=name_style, digits=digits)

def generate_synthetic_dataset(
    out_dir: str,
    count: int,
    preset: str = "default",
    kinds: Optional[List[str]] = None,
    base_seed: int = 0,
    dx: int = 4,
    dy: int = 2,
    name_style: str = "sad",
    width: Optional[int] = None,
    height: Optional[int] = None,
    size_in_name: bool = False,
    start_index: int = 0,
    digits: int = 4
) -> None:
    """
    Generate a synthetic dataset: a mixture of scenes.
    - If width/height are provided, they override the preset; both must be provided and be multiples of 16.
    """
    if width is not None or height is not None:
        if width is None or height is None:
            raise ValueError("width 与 height 需同时提供")
        if width % 16 != 0 or height % 16 != 0:
            raise ValueError("width/height 必须为 16 的倍数")
        target_sizes = [(width, height)]
    else:
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset: {preset}")
        target_sizes = [PRESETS[preset]]

    if kinds is None:
        kinds = ["shift", "noise", "checkerboard", "gradient", "blocks_mix"]

    ensure_dir(out_dir)
    for (w, h) in target_sizes:
        size_tag = f"{w}x{h}" if size_in_name else None
        for i in range(count):
            kind = kinds[(i - start_index) % len(kinds)] if len(kinds) > 0 else "shift"
            ref, cur = make_synthetic_pair(
                width=w, height=h,
                kind=kind, seed=base_seed + i, dx=dx, dy=dy
            )
            write_pair(ref, cur, out_dir, i, name_style=name_style, size_tag=size_tag, digits=digits)


def generate_synthetic_dataset_for_sizes(
    out_dir: str,
    count: int,
    sizes: List[Tuple[int, int]],
    kinds: Optional[List[str]] = None,
    base_seed: int = 0,
    dx: int = 4,
    dy: int = 2,
    name_style: str = "sad",
    by_size_subdir: bool = False,
    size_in_name: bool = False,
    start_index: int = 0,
    digits: int = 4
) -> None:
    """
    Generate datasets for multiple sizes.
    - by_size_subdir: create subdirectories per size at out_dir/WxH/
    - size_in_name: include size tag in file names, e.g., ref_640x480_0000.bin
    """
    if kinds is None:
        kinds = ["shift", "noise", "checkerboard", "gradient", "blocks_mix"]
    for (w, h) in sizes:
        if w % 16 != 0 or h % 16 != 0:
            raise ValueError(f"Size must be multiples of 16: {w}x{h}")
        this_out_dir = os.path.join(out_dir, f"{w}x{h}") if by_size_subdir else out_dir
        ensure_dir(this_out_dir)
        size_tag = f"{w}x{h}" if size_in_name else None
        for i in range(start_index, start_index + count):
            kind = kinds[(i - start_index) % len(kinds)] if len(kinds) > 0 else "shift"
            ref, cur = make_synthetic_pair(
                width=w, height=h,
                kind=kind, seed=base_seed + i, dx=dx, dy=dy
            )
            write_pair(ref, cur, this_out_dir, i, name_style=name_style, size_tag=size_tag, digits=digits)


# -------------------------
# CLI
# -------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate input binary pairs for SAD: (ref.bin, cur.bin) or (reference.bin, frame.bin)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # synthetic
    sp_syn = sub.add_parser("synthetic", help="Generate a synthetic dataset")
    sp_syn.add_argument("--out-dir", required=True, help="Output directory")
    sp_syn.add_argument("--count", type=int, default=10, help="Number of pairs to generate per size")
    sp_syn.add_argument("--preset", choices=list(PRESETS.keys()), default="default", help="Preset size")
    sp_syn.add_argument("--kinds", nargs="*", default=None,
                        help="Choose from: shift noise checkerboard gradient blocks_mix; default is all")
    sp_syn.add_argument("--seed", type=int, default=0, help="Base random seed")
    sp_syn.add_argument("--dx", type=int, default=4, help="Horizontal shift for applicable kinds")
    sp_syn.add_argument("--dy", type=int, default=2, help="Vertical shift for applicable kinds")
    sp_syn.add_argument("--name-style", choices=["sad", "alt"], default="sad",
                        help="sad: ref_XXXX.bin/cur_XXXX.bin; alt: reference_XXXX.bin/frame_XXXX.bin")
    sp_syn.add_argument("--width", type=int, default=None, help="Override preset width (multiple of 16)")
    sp_syn.add_argument("--height", type=int, default=None, help="Override preset height (multiple of 16)")
    sp_syn.add_argument("--sizes", type=str, default=None,
                        help="Multiple sizes list, e.g., '640x480,1280x736' (multiples of 16)")
    sp_syn.add_argument("--by-size-subdir", action="store_true",
                        help="Create a subdirectory per size: out_dir/WxH/")
    sp_syn.add_argument("--size-in-name", action="store_true",
                        help="Include size tag in file names, e.g., ref_640x480_0000.bin")
    sp_syn.add_argument("--start-index", type=int, default=0, help="Starting index for file numbering")
    sp_syn.add_argument("--digits", type=int, default=4, help="Zero-padding width for indices")

    # from-images
    sp_img = sub.add_parser("from-images", help="Generate pairs from two image lists")
    sp_img.add_argument("--ref-glob", required=True, help="Glob for reference images, e.g., 'ref_dir/*.png'")
    sp_img.add_argument("--cur-glob", required=True, help="Glob for current images, e.g., 'cur_dir/*.png'")
    sp_img.add_argument("--out-dir", required=True, help="Output directory")
    sp_img.add_argument("--preset", choices=list(PRESETS.keys()), default=None, help="Target preset size")
    sp_img.add_argument("--width", type=int, default=None, help="Target width (multiple of 16)")
    sp_img.add_argument("--height", type=int, default=None, help="Target height (multiple of 16)")
    sp_img.add_argument("--fit-mode", choices=["crop", "pad", "resize"], default="crop",
                        help="How to match the target size: crop/pad/resize")
    sp_img.add_argument("--name-style", choices=["sad", "alt"], default="sad",
                        help="Naming style: sad or alt")
    sp_img.add_argument("--start-index", type=int, default=0, help="Starting index for file numbering")
    sp_img.add_argument("--digits", type=int, default=4, help="Zero-padding width for indices")

    return p

def main():
    p = build_argparser()
    args = p.parse_args()

    if args.cmd == "synthetic":
        # Prefer --sizes; otherwise use --width/--height; otherwise fallback to preset
        if args.sizes:
            sizes = parse_sizes(args.sizes)
            generate_synthetic_dataset_for_sizes(
                out_dir=args.out_dir,
                count=args.count,
                sizes=sizes,
                kinds=args.kinds,
                base_seed=args.seed,
                dx=args.dx,
                dy=args.dy,
                name_style=args.name_style,
                by_size_subdir=args.by_size_subdir,
                size_in_name=args.size_in_name,
                start_index=args.start_index,
                digits=args.digits
            )
        else:
            generate_synthetic_dataset(
                out_dir=args.out_dir,
                count=args.count,
                preset=args.preset,
                kinds=args.kinds,
                base_seed=args.seed,
                dx=args.dx,
                dy=args.dy,
                name_style=args.name_style,
                width=args.width,
                height=args.height,
                size_in_name=args.size_in_name,
                start_index=args.start_index,
                digits=args.digits
            )
    elif args.cmd == "from-images":
        tgt_wh = None
        if args.preset is not None:
            tgt_wh = PRESETS[args.preset]
        elif args.width is not None and args.height is not None:
            if args.width % 16 != 0 or args.height % 16 != 0:
                raise SystemExit("width/height must be multiples of 16")
            tgt_wh = (args.width, args.height)

        generate_from_images(
            ref_glob=args.ref_glob,
            cur_glob=args.cur_glob,
            out_dir=args.out_dir,
            target_preset=args.preset,
            target_wh=tgt_wh,
            fit_mode=args.fit_mode,
            name_style=args.name_style,
            start_index=args.start_index,
            digits=args.digits
        )
    else:
        p.print_help()

if __name__ == "__main__":
    main()