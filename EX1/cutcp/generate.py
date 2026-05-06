#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import argparse
from typing import List, Tuple, Optional
import random


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_path_join(out_dir: str, filename: str, overwrite: bool) -> str:
    path = os.path.join(out_dir, filename)
    if (not overwrite) and os.path.exists(path):
        raise FileExistsError(f"Output file already exists: {path}. Use --overwrite to replace it.")
    return path


def write_text(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def format_cryst1(a: float = 0.0, b: float = 0.0, c: float = 0.0,
                  alpha: float = 90.0, beta: float = 90.0, gamma: float = 90.0) -> str:
    return f"CRYST1{a:8.3f}{b:8.3f}{c:8.3f}{alpha:7.2f}{beta:7.2f}{gamma:7.2f} P 1           1\n"


def format_atom(serial: int,
                name: str,
                resname: str,
                resseq: int,
                x: float, y: float, z: float,
                charge: float, radius: float) -> str:
    return (
        f"ATOM{serial:7d}  {name:<3s} {resname:<4s}{resseq:6d}"
        f"{x:11.3f}{y:8.3f}{z:8.3f} {charge:.3f} {radius:.3f}\n"
    )


def write_pqr(out_path: str,
              cryst1: Tuple[float, float, float, float, float, float],
              atoms: List[Tuple[int, str, str, int, float, float, float, float, float]]) -> None:
    lines = [format_cryst1(*cryst1)]
    for serial, name, res, resseq, x, y, z, q, r in atoms:
        lines.append(format_atom(serial, name, res, resseq, x, y, z, q, r))
    write_text(out_path, "".join(lines))


DEFAULT_ATOM_SPECS = {
    "H": (0.417, 1.000),
    "O": (-0.834, 1.520),
    "C": (-0.120, 1.700),
    "N": (-0.300, 1.550),
}


def generate_random_atoms(count: int,
                          box_half: float,
                          seed: int,
                          residue: str = "RES",
                          start_resseq: int = 1) -> List[Tuple[int, str, str, int, float, float, float, float, float]]:
    rng = random.Random(seed)
    items: List[Tuple[int, str, str, int, float, float, float, float, float]] = []
    names = list(DEFAULT_ATOM_SPECS.keys())
    resseq = start_resseq
    for serial in range(1, count + 1):
        name = rng.choice(names)
        q, r = DEFAULT_ATOM_SPECS[name]
        x = rng.uniform(-box_half, box_half)
        y = rng.uniform(-box_half, box_half)
        z = rng.uniform(-box_half, box_half)
        items.append((serial, name, "RES", resseq, x, y, z, q, r))
        if serial % 3 == 0:
            resseq += 1
    return items


PRESETS = {
    "mini": (3000, 10.0),
    #"small": (1000, 20.0),
    "medium": (40000, 40.0),
    #"large": (50000, 60.0),
    "extra-large": (200000, 100.0),
}


def generate_synthetic(out_dir: str,
                       count: int,
                       preset: str,
                       seed: int,
                       start_index: int,
                       digits: int,
                       overwrite: bool,
                       num_atoms: Optional[int] = None,
                       box_half: Optional[float] = None) -> None:
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}")
    preset_atoms, preset_box = PRESETS[preset]
    n_atoms = num_atoms if (num_atoms is not None) else preset_atoms
    b_half = box_half if (box_half is not None) else preset_box

    ensure_dir(out_dir)
    for i in range(start_index, start_index + count):
        atoms = generate_random_atoms(n_atoms, b_half, seed + i)
        cryst = (0.0, 0.0, 0.0, 90.0, 90.0, 90.0)
        idx = f"{i:0{digits}d}"
        out_path = safe_path_join(out_dir, f"input_{idx}.pqr", overwrite)
        write_pqr(out_path, cryst, atoms)


def minimal_validate_pqr(path: str) -> None:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first = f.readline()
        if not first.startswith("CRYST1"):
            raise ValueError(f"PQR missing CRYST1 line: {path}")
        ok = False
        for line in f:
            if line.startswith("ATOM"):
                parts = line.split()
                if len(parts) >= 10:
                    float(parts[-1]); float(parts[-2]); float(parts[-3]); float(parts[-4]);
                    ok = True
                    break
        if not ok:
            raise ValueError(f"PQR has no valid ATOM lines: {path}")


def from_sources(src_glob: str,
                 out_dir: str,
                 start_index: int,
                 digits: int,
                 overwrite: bool) -> None:
    paths = sorted(glob.glob(src_glob))
    if not paths:
        raise FileNotFoundError(f"No files matched: {src_glob}")
    ensure_dir(out_dir)
    for j, src in enumerate(paths, start=start_index):
        minimal_validate_pqr(src)
        idx = f"{j:0{digits}d}"
        dst = safe_path_join(out_dir, f"input_{idx}.pqr", overwrite)
        with open(src, "rb") as fi, open(dst, "wb") as fo:
            fo.write(fi.read())


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CUTCP dataset generator (PQR format)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp_syn = sub.add_parser("synthetic", help="Generate synthetic PQR files")
    sp_syn.add_argument("--out-dir", required=True, help="Output directory (recursively created)")
    sp_syn.add_argument("--count", type=int, default=1, help="Number of files to generate")
    sp_syn.add_argument("--preset", choices=list(PRESETS.keys()), default="small", help="Preset scale")
    sp_syn.add_argument("--seed", type=int, default=0, help="Base RNG seed")
    sp_syn.add_argument("--start-index", type=int, default=0, help="Starting index for filenames")
    sp_syn.add_argument("--digits", type=int, default=4, help="Zero-padding width for indices")
    sp_syn.add_argument("--overwrite", action="store_true", help="Allow overwriting existing files")
    sp_syn.add_argument("--num-atoms", type=int, default=None, help="Override atoms per file")
    sp_syn.add_argument("--box-half", type=float, default=None, help="Override half box size for coordinates")

    sp_src = sub.add_parser("from-sources", help="Copy/normalize existing PQR files")
    sp_src.add_argument("--src-glob", required=True, help="Glob for input PQR files")
    sp_src.add_argument("--out-dir", required=True, help="Output directory (recursively created)")
    sp_src.add_argument("--start-index", type=int, default=0)
    sp_src.add_argument("--digits", type=int, default=4)
    sp_src.add_argument("--overwrite", action="store_true")

    return p


def main() -> None:
    p = build_argparser()
    args = p.parse_args()

    if args.cmd == "synthetic":
        generate_synthetic(
            out_dir=args.out_dir,
            count=args.count,
            preset=args.preset,
            seed=args.seed,
            start_index=args.start_index,
            digits=args.digits,
            overwrite=args.overwrite,
            num_atoms=args.num_atoms,
            box_half=args.box_half,
        )
    elif args.cmd == "from-sources":
        from_sources(
            src_glob=args.src_glob,
            out_dir=args.out_dir,
            start_index=args.start_index,
            digits=args.digits,
            overwrite=args.overwrite,
        )
    else:
        p.print_help()


if __name__ == "__main__":
    main()


