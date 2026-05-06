#!/usr/bin/env python3
"""
TPACF (Two-Point Angular Correlation Function) Input Generator

Generates astronomical point data for the TPACF benchmark.
TPACF computes angular correlations between data points and random points.

Input Format:
  - Text files with two columns: RA (right ascension) and DEC (declination)
  - RA range: [0, 360) degrees
  - DEC range: [-90, 90] degrees
  - One data point file + N random point files

Output files:
  - Datapnts.1: Data points file
  - Randompnts.1, Randompnts.2, ..., Randompnts.N: Random points files
"""

import argparse
import sys
import random
import math
from pathlib import Path
from typing import Tuple, List, Optional, Dict


# ============================================================================
# Presets: (npoints, random_count)
# npoints: number of points per file
# random_count: number of random point files (always 100)
# ============================================================================
PRESETS: Dict[str, Tuple[int, int]] = {
    "mini":        (100,    100),   # Tiny for quick testing
    "small":       (487,    100),   # Matches parboil small
    "medium":      (4096,   100),   # Matches parboil medium
    "large":       (10391,  100),   # Matches parboil large
    "extra-large": (20000,  100),   # Extra large scale
}


# ============================================================================
# Data Generation
# ============================================================================

def generate_uniform_sky(npoints: int, seed: Optional[int] = None) -> List[Tuple[float, float]]:
    """
    Generate uniformly distributed points on a sphere.
    Uses the equal-area projection method.
    
    Returns list of (ra, dec) tuples in degrees.
    """
    if seed is not None:
        random.seed(seed)
    
    points = []
    for _ in range(npoints):
        # Uniform distribution on sphere using equal-area projection
        # u is uniform in [0, 1], maps to cos(dec)
        u = random.uniform(-1, 1)  # cos(dec) ranges from -1 to 1
        dec = math.degrees(math.asin(u))  # DEC in [-90, 90]
        
        # RA is uniform in [0, 360)
        ra = random.uniform(0, 360)
        
        points.append((ra, dec))
    
    return points


def generate_clustered_sky(npoints: int, nclusters: int = 5, 
                           cluster_spread: float = 10.0,
                           seed: Optional[int] = None) -> List[Tuple[float, float]]:
    """
    Generate clustered points on the sky.
    
    Args:
        npoints: Total number of points
        nclusters: Number of clusters
        cluster_spread: Spread of each cluster in degrees
        seed: Random seed
    
    Returns list of (ra, dec) tuples in degrees.
    """
    if seed is not None:
        random.seed(seed)
    
    # Generate cluster centers
    cluster_centers = []
    for _ in range(nclusters):
        u = random.uniform(-1, 1)
        dec = math.degrees(math.asin(u))
        ra = random.uniform(0, 360)
        cluster_centers.append((ra, dec))
    
    # Generate points around clusters
    points = []
    points_per_cluster = npoints // nclusters
    remainder = npoints % nclusters
    
    for i, (center_ra, center_dec) in enumerate(cluster_centers):
        n = points_per_cluster + (1 if i < remainder else 0)
        
        for _ in range(n):
            # Gaussian distribution around cluster center
            ra = center_ra + random.gauss(0, cluster_spread)
            dec = center_dec + random.gauss(0, cluster_spread)
            
            # Wrap RA to [0, 360)
            ra = ra % 360
            
            # Clamp DEC to [-90, 90]
            dec = max(-90, min(90, dec))
            
            points.append((ra, dec))
    
    return points


def generate_grid_sky(npoints: int, seed: Optional[int] = None) -> List[Tuple[float, float]]:
    """
    Generate points on a regular grid pattern.
    
    Returns list of (ra, dec) tuples in degrees.
    """
    if seed is not None:
        random.seed(seed)
    
    # Create a grid
    n_ra = int(math.sqrt(npoints * 2))  # More RA divisions
    n_dec = int(math.sqrt(npoints / 2))  # Fewer DEC divisions
    
    points = []
    for i in range(npoints):
        idx_ra = i % n_ra
        idx_dec = i // n_ra
        
        ra = (idx_ra / n_ra) * 360
        dec = (idx_dec / n_dec) * 180 - 90
        
        # Add small random jitter
        if seed is not None:
            ra += random.uniform(-1, 1)
            dec += random.uniform(-0.5, 0.5)
        
        # Ensure bounds
        ra = ra % 360
        dec = max(-90, min(90, dec))
        
        points.append((ra, dec))
    
    return points[:npoints]


# ============================================================================
# File I/O
# ============================================================================

def validate_points(points: List[Tuple[float, float]], npoints: int) -> None:
    """Validate point data."""
    if len(points) != npoints:
        raise ValueError(
            f"Point count mismatch: expected {npoints}, got {len(points)}"
        )
    
    for i, (ra, dec) in enumerate(points):
        if not (0 <= ra < 360):
            raise ValueError(
                f"Invalid RA at index {i}: {ra} (must be in [0, 360))"
            )
        if not (-90 <= dec <= 90):
            raise ValueError(
                f"Invalid DEC at index {i}: {dec} (must be in [-90, 90])"
            )


def write_point_file(path: str, points: List[Tuple[float, float]]) -> None:
    """
    Write points to file.
    
    Format: Two columns of floats (RA DEC), space-separated, one point per line.
    """
    with open(path, 'w') as f:
        for ra, dec in points:
            f.write(f"{ra:.7f} {dec:.7f}\n")


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


def write_dataset(out_dir: str, npoints: int, random_count: int,
                  data_points: List[Tuple[float, float]],
                  random_points_list: List[List[Tuple[float, float]]],
                  overwrite: bool) -> Tuple[str, List[str]]:
    """
    Write complete TPACF dataset.
    
    Returns tuple of (data_file_path, list_of_random_file_paths).
    """
    ensure_dir(out_dir)
    
    # Validate
    validate_points(data_points, npoints)
    if len(random_points_list) != random_count:
        raise ValueError(
            f"Random file count mismatch: expected {random_count}, "
            f"got {len(random_points_list)}"
        )
    
    for i, rnd_pts in enumerate(random_points_list):
        validate_points(rnd_pts, npoints)
    
    # Write data points file
    data_path = safe_join(out_dir, "Datapnts.1", overwrite)
    write_point_file(data_path, data_points)
    
    # Write random points files
    random_paths = []
    for i, rnd_pts in enumerate(random_points_list, start=1):
        rnd_path = safe_join(out_dir, f"Randompnts.{i}", overwrite)
        write_point_file(rnd_path, rnd_pts)
        random_paths.append(rnd_path)
    
    return data_path, random_paths


# ============================================================================
# CLI Subcommands
# ============================================================================

def cmd_synthetic(args):
    """Generate synthetic TPACF data."""
    # Resolve dimensions
    if args.preset:
        if args.preset not in PRESETS:
            print(f"Error: Unknown preset '{args.preset}'", file=sys.stderr)
            print(f"Available presets: {', '.join(PRESETS.keys())}", file=sys.stderr)
            sys.exit(1)
        npoints, random_count = PRESETS[args.preset]
    else:
        if args.npoints is None or args.random_count is None:
            print("Error: Must specify either --preset or both --npoints and --random-count", 
                  file=sys.stderr)
            sys.exit(1)
        npoints = args.npoints
        random_count = args.random_count
    
    # Override if explicitly provided
    if args.npoints is not None:
        npoints = args.npoints
    if args.random_count is not None:
        random_count = args.random_count
    
    # Validate
    if npoints <= 0:
        print(f"Error: npoints must be positive (got {npoints})", file=sys.stderr)
        sys.exit(1)
    if random_count <= 0:
        print(f"Error: random_count must be positive (got {random_count})", 
              file=sys.stderr)
        sys.exit(1)
    
    # Generate data points
    print(f"Generating {npoints} data points...")
    if args.data_kind == "uniform":
        data_points = generate_uniform_sky(npoints, args.seed)
    elif args.data_kind == "clustered":
        data_points = generate_clustered_sky(npoints, seed=args.seed)
    elif args.data_kind == "grid":
        data_points = generate_grid_sky(npoints, seed=args.seed)
    else:
        print(f"Error: Unknown data kind '{args.data_kind}'", file=sys.stderr)
        sys.exit(1)
    
    # Generate random points
    print(f"Generating {random_count} random point files...")
    random_points_list = []
    for i in range(random_count):
        # Always use uniform distribution for random points
        # Use different seed for each file
        seed = args.seed + i + 1000 if args.seed is not None else None
        rnd_pts = generate_uniform_sky(npoints, seed)
        random_points_list.append(rnd_pts)
    
    # Write dataset
    data_path, random_paths = write_dataset(
        args.out_dir, npoints, random_count,
        data_points, random_points_list, args.overwrite
    )
    
    print(f"Generated dataset:")
    print(f"  Data file: {data_path} ({npoints} points)")
    print(f"  Random files: {len(random_paths)} files ({npoints} points each)")
    print(f"  Total: {npoints * (1 + random_count)} points")


def cmd_from_sources(args):
    """Convert external sources to TPACF format."""
    import glob
    
    # Get dimensions
    if args.preset:
        npoints, random_count = PRESETS[args.preset]
    else:
        if args.npoints is None or args.random_count is None:
            print("Error: Must specify either --preset or both --npoints and --random-count", 
                  file=sys.stderr)
            sys.exit(1)
        npoints = args.npoints
        random_count = args.random_count
    
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
    
    if len(src_files) < random_count + 1:
        print(f"Warning: Need {random_count + 1} files (1 data + {random_count} random), "
              f"got {len(src_files)}", file=sys.stderr)
    
    # Read and convert files
    data_points = []
    random_points_list = []
    
    for idx, src_file in enumerate(src_files):
        print(f"Reading: {src_file}")
        
        try:
            with open(src_file, 'r') as f:
                points = []
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        ra = float(parts[0])
                        dec = float(parts[1])
                        points.append((ra, dec))
            
            # Pad or truncate to expected size
            if len(points) < npoints:
                print(f"  Warning: File has {len(points)} points, padding to {npoints}")
                # Pad with uniform random points
                while len(points) < npoints:
                    ra = random.uniform(0, 360)
                    u = random.uniform(-1, 1)
                    dec = math.degrees(math.asin(u))
                    points.append((ra, dec))
            elif len(points) > npoints:
                print(f"  Warning: File has {len(points)} points, truncating to {npoints}")
                points = points[:npoints]
            
            # First file is data, rest are random
            if idx == 0:
                data_points = points
            else:
                random_points_list.append(points)
            
            if len(random_points_list) >= random_count:
                break
        
        except Exception as e:
            print(f"Error reading {src_file}: {e}", file=sys.stderr)
            continue
    
    # Ensure we have enough files
    while len(random_points_list) < random_count:
        print(f"Generating random file {len(random_points_list) + 1}...")
        rnd_pts = generate_uniform_sky(npoints, args.seed)
        random_points_list.append(rnd_pts)
    
    # Write dataset
    data_path, random_paths = write_dataset(
        args.out_dir, npoints, random_count,
        data_points, random_points_list, args.overwrite
    )
    
    print(f"Converted dataset:")
    print(f"  Data file: {data_path}")
    print(f"  Random files: {len(random_paths)} files")


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate TPACF benchmark input data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate small dataset (487 points, 100 random files)
  python generate.py synthetic --out-dir input_data/small/input --preset small --data-kind uniform --seed 42
  
  # Generate medium dataset (4096 points, 100 random files)
  python generate.py synthetic --out-dir input_data/medium/input --preset medium --data-kind uniform --seed 123
  
  # Generate custom dataset
  python generate.py synthetic --out-dir input_data/custom/input --npoints 1000 --random-count 50 --data-kind clustered --seed 0
  
  # Convert external files
  python generate.py from-sources --src-glob "raw/*.txt" --out-dir input_data/test/input --preset small
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # ---- Synthetic subcommand ----
    syn = subparsers.add_parser('synthetic', help='Generate synthetic TPACF data')
    syn.add_argument('--out-dir', type=str, required=True,
                     help='Output directory (created if needed)')
    syn.add_argument('--preset', type=str, choices=list(PRESETS.keys()),
                     help=f'Size preset: {", ".join(PRESETS.keys())}')
    syn.add_argument('--npoints', type=int,
                     help='Number of points per file')
    syn.add_argument('--random-count', type=int,
                     help='Number of random point files')
    syn.add_argument('--data-kind', type=str,
                     choices=['uniform', 'clustered', 'grid'],
                     default='uniform',
                     help='Data distribution pattern (default: uniform)')
    syn.add_argument('--seed', type=int, default=None,
                     help='Random seed for reproducibility')
    syn.add_argument('--overwrite', action='store_true',
                     help='Overwrite existing files')
    syn.set_defaults(func=cmd_synthetic)
    
    # ---- From-sources subcommand ----
    src = subparsers.add_parser('from-sources',
                                help='Convert external data to TPACF format')
    src.add_argument('--out-dir', type=str, required=True,
                     help='Output directory')
    src.add_argument('--src-glob', type=str,
                     help='Glob pattern for source files')
    src.add_argument('--src-list', type=str, nargs='+',
                     help='List of source files')
    src.add_argument('--preset', type=str, choices=list(PRESETS.keys()),
                     help='Size preset')
    src.add_argument('--npoints', type=int,
                     help='Number of points per file')
    src.add_argument('--random-count', type=int,
                     help='Number of random point files')
    src.add_argument('--seed', type=int, default=None,
                     help='Random seed for filling missing data')
    src.add_argument('--overwrite', action='store_true',
                     help='Overwrite existing files')
    src.set_defaults(func=cmd_from_sources)
    
    # Parse and execute
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()

