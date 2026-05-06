#!/usr/bin/env python3
"""
Data generator for qsort-mibench benchmark.

This benchmark sorts 3D vertex points (x, y, z) by their distance from the origin.
Input format: text file with one line per vertex, three space/tab-separated integers.
"""

import argparse
import random
import sys
from pathlib import Path


# Predefined size presets
# Note: Requires MAXARRAY=31000000 and malloc() to handle all sizes
PRESETS = {
    "mini": {
        "count": 60000,
        "coord_range": (-2**31, 2**31 - 1),
        "description": "Mini size (60K vertices)"
    },
    "small": {
        "count": 300000,
        "coord_range": (-2**31, 2**31 - 1),
        "description": "Small size (300K vertices)"
    },
    "medium": {
        "count": 2000000,
        "coord_range": (-2**31, 2**31 - 1),
        "description": "Medium size (2M vertices)"
    },
    "large": {
        "count": 10000000,
        "coord_range": (-2**31, 2**31 - 1),
        "description": "Large size (10M vertices)"
    },
    "extra-large": {
        "count": 30000000,
        "coord_range": (-2**31, 2**31 - 1),
        "description": "Extra-large size (30M vertices)"
    }
}


def validate_config(count, coord_range):
    """Validate generation configuration."""
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    
    min_coord, max_coord = coord_range
    if min_coord >= max_coord:
        raise ValueError(f"Invalid coordinate range: [{min_coord}, {max_coord})")
    
    # Check if range is within 32-bit signed integer
    if min_coord < -2**31 or max_coord > 2**31 - 1:
        raise ValueError(f"Coordinate range must be within 32-bit signed integer range")


def generate_vertex_data(count, coord_range, seed=None):
    """
    Generate random 3D vertex coordinates.
    
    Args:
        count: Number of vertices to generate
        coord_range: Tuple (min, max) for coordinate values
        seed: Random seed for reproducibility
    
    Returns:
        List of (x, y, z) tuples
    """
    if seed is not None:
        random.seed(seed)
    
    min_coord, max_coord = coord_range
    vertices = []
    
    for _ in range(count):
        x = random.randint(min_coord, max_coord)
        y = random.randint(min_coord, max_coord)
        z = random.randint(min_coord, max_coord)
        vertices.append((x, y, z))
    
    return vertices


def encode_to_text(vertices):
    """
    Encode vertices to text format (qsort input format).
    
    Format: Each line contains three tab-separated integers (x, y, z)
    
    Args:
        vertices: List of (x, y, z) tuples
    
    Returns:
        String containing formatted text
    """
    lines = []
    for x, y, z in vertices:
        lines.append(f"{x}\t{y}\t{z}")
    return "\n".join(lines) + "\n"


def write_output(vertices, output_path, overwrite=False):
    """
    Write vertices to output file.
    
    Args:
        vertices: List of (x, y, z) tuples
        output_path: Path to output file
        overwrite: Whether to overwrite existing files
    """
    output_path = Path(output_path)
    
    # Check if file exists
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}\n"
            f"Use --overwrite to force overwrite"
        )
    
    # Create parent directories
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Encode and write
    text_content = encode_to_text(vertices)
    
    with open(output_path, 'w') as f:
        f.write(text_content)
    
    print(f"Generated {len(vertices)} vertices -> {output_path} ({output_path.stat().st_size} bytes)")


def write_metadata(vertices, meta_path):
    """Write metadata file with generation information."""
    meta_path = Path(meta_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(meta_path, 'w') as f:
        f.write(f"vertex_count: {len(vertices)}\n")
        f.write(f"format: text (tab-separated)\n")
        f.write(f"fields: x y z (int32)\n")


def cmd_synthetic(args):
    """Handle 'synthetic' subcommand."""
    # Load preset if specified
    if args.preset:
        if args.preset not in PRESETS:
            print(f"Error: Unknown preset '{args.preset}'", file=sys.stderr)
            print(f"Available presets: {', '.join(PRESETS.keys())}", file=sys.stderr)
            return 1
        
        preset = PRESETS[args.preset]
        count = args.count if args.count is not None else preset["count"]
        coord_range = preset["coord_range"]
        print(f"Using preset '{args.preset}': {preset['description']}")
    else:
        if args.count is None:
            print("Error: --count is required when not using a preset", file=sys.stderr)
            return 1
        count = args.count
        coord_range = (-2**31, 2**31 - 1)
    
    # Validate configuration
    try:
        validate_config(count, coord_range)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    # Generate file name
    if args.name:
        filename = args.name
    else:
        prefix = args.prefix or "input"
        suffix = args.suffix or ".dat"
        if args.preset:
            filename = f"{prefix}_{args.preset}{suffix}"
        else:
            filename = f"{prefix}_custom{suffix}"
    
    output_path = Path(args.out_dir) / filename
    
    # Generate data
    print(f"Generating {count} vertices (seed={args.seed})...")
    vertices = generate_vertex_data(count, coord_range, seed=args.seed)
    
    # Write output
    try:
        write_output(vertices, output_path, overwrite=args.overwrite)
    except FileExistsError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    # Write metadata if requested
    if args.write_meta:
        meta_path = output_path.with_suffix('.meta')
        write_metadata(vertices, meta_path)
        print(f"Metadata written to {meta_path}")
    
    return 0


def cmd_from_sources(args):
    """Handle 'from-sources' subcommand (placeholder)."""
    print("Error: 'from-sources' not implemented for qsort-mibench", file=sys.stderr)
    print("This benchmark generates synthetic 3D coordinates only", file=sys.stderr)
    return 1


def main():
    parser = argparse.ArgumentParser(
        description="Data generator for qsort-mibench benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate mini size data (50K vertices)
  python generate.py synthetic --preset mini --out-dir input_data/mini/input

  # Generate all sizes
  python generate.py synthetic --preset small --out-dir input_data/small/input
  python generate.py synthetic --preset medium --out-dir input_data/medium/input
  python generate.py synthetic --preset large --out-dir input_data/large/input
  python generate.py synthetic --preset extra-large --out-dir input_data/extra-large/input

  # Custom size with specific seed
  python generate.py synthetic --count 100000 --seed 42 --out-dir data/custom

Available presets:
  mini        : 60K vertices
  small       : 300K vertices
  medium      : 2M vertices
  large       : 10M vertices
  extra-large : 30M vertices

Note: Requires MAXARRAY=31000000 and malloc() in qsort_large.c
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Sub-command to run')
    
    # Synthetic subcommand
    synthetic_parser = subparsers.add_parser('synthetic', help='Generate synthetic 3D vertex data')
    synthetic_parser.add_argument('--out-dir', type=str, required=True,
                                   help='Output directory (will be created if needed)')
    synthetic_parser.add_argument('--count', type=int,
                                   help='Number of vertices to generate')
    synthetic_parser.add_argument('--preset', type=str, choices=list(PRESETS.keys()),
                                   help='Use predefined size preset')
    synthetic_parser.add_argument('--seed', type=int, default=12345,
                                   help='Random seed for reproducibility (default: 12345)')
    synthetic_parser.add_argument('--name', type=str,
                                   help='Output filename (default: auto-generated from preset)')
    synthetic_parser.add_argument('--prefix', type=str, default='input',
                                   help='Filename prefix (default: input)')
    synthetic_parser.add_argument('--suffix', type=str, default='.dat',
                                   help='Filename suffix (default: .dat)')
    synthetic_parser.add_argument('--overwrite', action='store_true',
                                   help='Overwrite existing files')
    synthetic_parser.add_argument('--write-meta', action='store_true',
                                   help='Write metadata file alongside output')
    
    # From-sources subcommand (placeholder)
    from_sources_parser = subparsers.add_parser('from-sources',
                                                 help='Convert from external sources (not implemented)')
    from_sources_parser.add_argument('--out-dir', type=str, required=True,
                                      help='Output directory')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command == 'synthetic':
        return cmd_synthetic(args)
    elif args.command == 'from-sources':
        return cmd_from_sources(args)
    else:
        print(f"Error: Unknown command '{args.command}'", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())

