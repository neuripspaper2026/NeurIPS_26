#!/usr/bin/env python3
"""
Stencil Input Generator

Generates 3D grid data for the stencil benchmark.
The stencil benchmark performs iterative 7-point stencil computations on 3D grids.

Input Format:
  - Binary file with nx×ny×nz float32 values (little-endian)
  - Layout: z-major, then y, then x (for(z) for(y) for(x))
  - No header, just raw float32 values

Output files are named: {nx}x{ny}x{nz}.bin
"""

import argparse
import sys
import struct
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import random


# ============================================================================
# Presets: (nx, ny, nz) - Grid dimensions
# ============================================================================
PRESETS: Dict[str, Tuple[int, int, int]] = {
    "mini":    (64,  64,  16),      # Tiny for quick testing
    "small":   (128, 128, 32),      # Matches parboil small
    "medium":  (256, 256, 48),      # Medium scale
    "default": (512, 512, 64),      # Matches parboil default
    "extra-large":   (768, 768, 96),      # Extra-large scale
}


# ============================================================================
# Data Generation
# ============================================================================

def generate_uniform(nx: int, ny: int, nz: int, 
                     value: float, seed: Optional[int] = None) -> List[float]:
    """Generate uniform grid with constant value."""
    size = nx * ny * nz
    return [value] * size


def generate_random(nx: int, ny: int, nz: int, 
                    min_val: float = 0.0, max_val: float = 1.0,
                    seed: Optional[int] = None) -> List[float]:
    """Generate random grid values."""
    if seed is not None:
        random.seed(seed)
    
    size = nx * ny * nz
    return [random.uniform(min_val, max_val) for _ in range(size)]


def generate_gaussian(nx: int, ny: int, nz: int,
                      center_x: Optional[float] = None,
                      center_y: Optional[float] = None,
                      center_z: Optional[float] = None,
                      sigma: float = 10.0,
                      seed: Optional[int] = None) -> List[float]:
    """Generate Gaussian distribution centered in the grid."""
    if seed is not None:
        random.seed(seed)
    
    # Default to grid center
    if center_x is None:
        center_x = nx / 2.0
    if center_y is None:
        center_y = ny / 2.0
    if center_z is None:
        center_z = nz / 2.0
    
    grid = []
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                dx = ix - center_x
                dy = iy - center_y
                dz = iz - center_z
                dist_sq = dx*dx + dy*dy + dz*dz
                value = float(1.0 * (2.718281828 ** (-dist_sq / (2.0 * sigma * sigma))))
                grid.append(value)
    
    return grid


def generate_checkerboard(nx: int, ny: int, nz: int,
                          val0: float = 0.0, val1: float = 1.0,
                          block_size: int = 4,
                          seed: Optional[int] = None) -> List[float]:
    """Generate checkerboard pattern."""
    grid = []
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                bx = (ix // block_size) % 2
                by = (iy // block_size) % 2
                bz = (iz // block_size) % 2
                parity = (bx + by + bz) % 2
                grid.append(val1 if parity else val0)
    
    return grid


# ============================================================================
# File I/O
# ============================================================================

def validate_grid(nx: int, ny: int, nz: int, grid: List[float]) -> None:
    """Validate grid dimensions and data."""
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError(f"Invalid grid dimensions: nx={nx}, ny={ny}, nz={nz} (must be positive)")
    
    expected_size = nx * ny * nz
    if len(grid) != expected_size:
        raise ValueError(
            f"Grid size mismatch: expected {expected_size} elements "
            f"(nx={nx}, ny={ny}, nz={nz}), got {len(grid)}"
        )
    
    # Check for invalid float values
    for i, val in enumerate(grid):
        if not isinstance(val, (int, float)):
            raise ValueError(f"Invalid value type at index {i}: {type(val)}")
        if val != val:  # NaN check
            raise ValueError(f"NaN value detected at index {i}")


def write_stencil_file(path: str, nx: int, ny: int, nz: int, grid: List[float]) -> None:
    """
    Write stencil input file.
    
    Format: Raw binary float32 values in little-endian, z-major order.
    """
    validate_grid(nx, ny, nz, grid)
    
    with open(path, 'wb') as f:
        # Write grid data: z-major, then y, then x
        for value in grid:
            f.write(struct.pack('<f', value))  # little-endian float32


def safe_join(out_dir: str, filename: str, overwrite: bool) -> str:
    """Create output path, checking for overwrites."""
    path = Path(out_dir) / filename
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {path}\n"
            f"Use --overwrite to replace it."
        )
    return str(path)


def ensure_dir(path: str) -> None:
    """Recursively create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def write_one(out_dir: str, name_prefix: str, nx: int, ny: int, nz: int,
              grid: List[float], overwrite: bool, 
              fixed_name: bool = False) -> str:
    """
    Write one stencil input file.
    
    Returns the path to the written file.
    """
    ensure_dir(out_dir)
    
    # Filename format: {nx}x{ny}x{nz}.bin or prefix_index.bin
    if fixed_name:
        filename = f"{nx}x{ny}x{nz}.bin"
    else:
        filename = f"{name_prefix}{nx}x{ny}x{nz}.bin"
    
    file_path = safe_join(out_dir, filename, overwrite)
    write_stencil_file(file_path, nx, ny, nz, grid)
    
    return file_path


# ============================================================================
# CLI Subcommands
# ============================================================================

def cmd_synthetic(args):
    """Generate synthetic stencil data."""
    # Resolve dimensions
    if args.preset:
        if args.preset not in PRESETS:
            print(f"Error: Unknown preset '{args.preset}'", file=sys.stderr)
            print(f"Available presets: {', '.join(PRESETS.keys())}", file=sys.stderr)
            sys.exit(1)
        nx, ny, nz = PRESETS[args.preset]
    else:
        if args.nx is None or args.ny is None or args.nz is None:
            print("Error: Must specify either --preset or all of (--nx, --ny, --nz)", 
                  file=sys.stderr)
            sys.exit(1)
        nx, ny, nz = args.nx, args.ny, args.nz
    
    # Override dimensions if explicitly provided
    if args.nx is not None:
        nx = args.nx
    if args.ny is not None:
        ny = args.ny
    if args.nz is not None:
        nz = args.nz
    
    # Validate
    if nx <= 0 or ny <= 0 or nz <= 0:
        print(f"Error: Grid dimensions must be positive (got nx={nx}, ny={ny}, nz={nz})", 
              file=sys.stderr)
        sys.exit(1)
    
    # Generate data based on kind
    if args.kind == "uniform":
        grid = generate_uniform(nx, ny, nz, args.value, args.seed)
    elif args.kind == "random":
        grid = generate_random(nx, ny, nz, args.min_val, args.max_val, args.seed)
    elif args.kind == "gaussian":
        grid = generate_gaussian(nx, ny, nz, seed=args.seed)
    elif args.kind == "checkerboard":
        grid = generate_checkerboard(nx, ny, nz, seed=args.seed)
    else:
        print(f"Error: Unknown kind '{args.kind}'", file=sys.stderr)
        sys.exit(1)
    
    # Generate multiple files
    for i in range(args.count):
        index = args.start_index + i
        if args.fixed_name:
            name_prefix = ""
        else:
            name_prefix = f"input_{index:0{args.digits}d}_"
        
        # For multiple files with different seeds
        if args.count > 1 and args.kind == "random" and args.seed is not None:
            grid = generate_random(nx, ny, nz, args.min_val, args.max_val, 
                                   args.seed + i)
        
        file_path = write_one(args.out_dir, name_prefix, nx, ny, nz, grid, 
                             args.overwrite, args.fixed_name)
        
        size_mb = (nx * ny * nz * 4) / (1024 * 1024)
        print(f"Generated: {file_path} ({nx}×{ny}×{nz}, {size_mb:.2f} MB)")


def cmd_from_sources(args):
    """Convert external sources to stencil format."""
    import glob
    
    # Find source files
    if args.src_glob:
        src_files = sorted(glob.glob(args.src_glob))
        if not src_files:
            print(f"Error: No files match pattern: {args.src_glob}", file=sys.stderr)
            sys.exit(1)
    elif args.src_list:
        src_files = args.src_list
    else:
        print("Error: Must specify either --src-glob or --src-list", file=sys.stderr)
        sys.exit(1)
    
    # Get dimensions
    if args.preset:
        nx, ny, nz = PRESETS[args.preset]
    else:
        if args.nx is None or args.ny is None or args.nz is None:
            print("Error: Must specify either --preset or all of (--nx, --ny, --nz)", 
                  file=sys.stderr)
            sys.exit(1)
        nx, ny, nz = args.nx, args.ny, args.nz
    
    # Process each source file
    for i, src_file in enumerate(src_files):
        print(f"Converting: {src_file}")
        
        # Read raw binary data
        try:
            with open(src_file, 'rb') as f:
                data = f.read()
            
            expected_bytes = nx * ny * nz * 4
            if len(data) < expected_bytes:
                print(f"Warning: File {src_file} is smaller than expected "
                      f"({len(data)} < {expected_bytes} bytes)", file=sys.stderr)
            
            # Parse as float32 little-endian
            num_floats = len(data) // 4
            grid = list(struct.unpack(f'<{num_floats}f', data[:num_floats * 4]))
            
            # Pad or truncate to expected size
            expected_size = nx * ny * nz
            if len(grid) < expected_size:
                grid.extend([0.0] * (expected_size - len(grid)))
            elif len(grid) > expected_size:
                grid = grid[:expected_size]
            
        except Exception as e:
            print(f"Error reading {src_file}: {e}", file=sys.stderr)
            continue
        
        # Write output
        index = args.start_index + i
        name_prefix = f"input_{index:0{args.digits}d}_"
        
        file_path = write_one(args.out_dir, name_prefix, nx, ny, nz, grid, 
                             args.overwrite, args.fixed_name)
        print(f"  -> {file_path}")


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate stencil benchmark input data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 1 default-sized grid with random values
  python generate.py synthetic --out-dir data/default/input --preset default --kind random --count 1 --fixed-name
  
  # Generate 10 small grids with Gaussian distribution
  python generate.py synthetic --out-dir data/small/input --preset small --kind gaussian --count 10 --start-index 0
  
  # Generate custom-sized grid with uniform values
  python generate.py synthetic --out-dir data/custom/input --nx 256 --ny 256 --nz 64 --kind uniform --value 1.0 --count 1
  
  # Convert external binary files
  python generate.py from-sources --src-glob "raw/*.bin" --out-dir data/test/input --preset small
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # ---- Synthetic subcommand ----
    syn = subparsers.add_parser('synthetic', help='Generate synthetic stencil data')
    syn.add_argument('--out-dir', type=str, required=True,
                     help='Output directory (created if needed)')
    syn.add_argument('--count', type=int, default=1,
                     help='Number of files to generate (default: 1)')
    syn.add_argument('--preset', type=str, choices=list(PRESETS.keys()),
                     help=f'Grid size preset: {", ".join(PRESETS.keys())}')
    syn.add_argument('--nx', type=int, help='Grid size in X dimension')
    syn.add_argument('--ny', type=int, help='Grid size in Y dimension')
    syn.add_argument('--nz', type=int, help='Grid size in Z dimension')
    syn.add_argument('--kind', type=str, 
                     choices=['uniform', 'random', 'gaussian', 'checkerboard'],
                     default='random',
                     help='Data pattern (default: random)')
    syn.add_argument('--value', type=float, default=1.0,
                     help='Value for uniform pattern (default: 1.0)')
    syn.add_argument('--min-val', type=float, default=0.0,
                     help='Min value for random pattern (default: 0.0)')
    syn.add_argument('--max-val', type=float, default=1.0,
                     help='Max value for random pattern (default: 1.0)')
    syn.add_argument('--seed', type=int, default=None,
                     help='Random seed for reproducibility')
    syn.add_argument('--start-index', type=int, default=0,
                     help='Starting index for file naming (default: 0)')
    syn.add_argument('--digits', type=int, default=4,
                     help='Number of digits for zero-padding indices (default: 4)')
    syn.add_argument('--fixed-name', action='store_true',
                     help='Use fixed naming: {nx}x{ny}x{nz}.bin (ignores index)')
    syn.add_argument('--overwrite', action='store_true',
                     help='Overwrite existing files')
    syn.set_defaults(func=cmd_synthetic)
    
    # ---- From-sources subcommand ----
    src = subparsers.add_parser('from-sources', 
                                help='Convert external data to stencil format')
    src.add_argument('--out-dir', type=str, required=True,
                     help='Output directory')
    src.add_argument('--src-glob', type=str,
                     help='Glob pattern for source files (e.g., "raw/*.bin")')
    src.add_argument('--src-list', type=str, nargs='+',
                     help='List of source files')
    src.add_argument('--preset', type=str, choices=list(PRESETS.keys()),
                     help='Grid size preset')
    src.add_argument('--nx', type=int, help='Grid size in X dimension')
    src.add_argument('--ny', type=int, help='Grid size in Y dimension')
    src.add_argument('--nz', type=int, help='Grid size in Z dimension')
    src.add_argument('--start-index', type=int, default=0,
                     help='Starting index for output naming')
    src.add_argument('--digits', type=int, default=4,
                     help='Number of digits for zero-padding')
    src.add_argument('--fixed-name', action='store_true',
                     help='Use fixed naming: {nx}x{ny}x{nz}.bin')
    src.add_argument('--overwrite', action='store_true',
                     help='Overwrite existing files')
    src.set_defaults(func=cmd_from_sources)
    
    # Parse and execute
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()

