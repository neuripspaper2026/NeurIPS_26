#!/usr/bin/env python3
"""
CFD (Euler3D) Input Data Generator

Generates synthetic computational fluid dynamics mesh data for the Euler3D benchmark.
The input format consists of:
- Line 1: Number of elements (nel)
- Lines 2-nel+1: For each element:
  - area (float)
  - 4 neighboring elements (int, -1 for boundary)
  - 12 normal components (float, 3D normals for each of 4 neighbors)

Format per line:
area  nb1 nx1 ny1 nz1  nb2 nx2 ny2 nz2  nb3 nx3 ny3 nz3  nb4 nx4 ny4 nz4
"""

import argparse
import sys
import os
import random
from pathlib import Path


# Constants
NNB = 4  # Number of neighbors per element
NDIM = 3  # Spatial dimensions

# Presets: (nel, description)
PRESETS = {
    "mini": (12000, "Mini dataset (~12K elements, ~0.2MB)"),
    "small": (48000, "Small dataset (~48K elements, ~0.8MB)"),
    "medium": (97046, "Medium dataset (~97K elements, ~1.6MB)"),
    "large": (193474, "Large dataset (~193K elements, ~3.2MB)"),
    "extra-large": (232536, "Extra-large dataset (~232K elements, ~3.8MB)"),
}


def generate_mesh_element(element_id, nel, seed_offset=0):
    """
    Generate a single mesh element with area, neighbors, and normals.
    
    Args:
        element_id: Element index (0-based)
        nel: Total number of elements
        seed_offset: Seed offset for randomness
    
    Returns:
        Tuple of (area, neighbors, normals)
    """
    # Seed for reproducibility (per element)
    rng = random.Random(seed_offset + element_id)
    
    # Generate area (typically 0.1 to 0.5 for realistic meshes)
    area = rng.uniform(0.1, 0.5)
    
    # Generate 4 neighbors with structured connectivity
    neighbors = []
    for j in range(NNB):
        # Boundary elements (-1) or structured neighbors
        if element_id < 10 or rng.random() < 0.05:  # First few elements and 5% random boundary
            neighbors.append(-1)
        else:
            # Use nearby elements as neighbors for better connectivity
            # This creates a more realistic mesh structure
            offset = rng.choice([-3, -2, -1, 1, 2, 3])
            nb = element_id + offset
            if nb < 0 or nb >= nel:
                neighbors.append(-1)  # Boundary
            else:
                neighbors.append(nb + 1)  # Convert to 1-based Fortran numbering
    
    # Generate 12 normal components (3D normals for 4 neighbors)
    # Use unit normals pointing in reasonable directions
    normals = []
    for j in range(NNB):
        # Generate a random direction
        theta = rng.uniform(0, 2.0 * 3.14159265359)  # azimuthal angle
        phi = rng.uniform(0, 3.14159265359)  # polar angle
        
        # Convert to Cartesian coordinates (unit sphere)
        import math
        nx = math.sin(phi) * math.cos(theta)
        ny = math.sin(phi) * math.sin(theta)
        nz = math.cos(phi)
        
        # Scale to match the range seen in real data (typically < 2.0)
        scale = rng.uniform(0.5, 1.5)
        normals.extend([nx * scale, ny * scale, nz * scale])
    
    return area, neighbors, normals


def generate_mesh(nel, seed=42):
    """
    Generate a complete mesh with nel elements.
    
    Args:
        nel: Number of elements
        seed: Random seed for reproducibility
    
    Returns:
        List of (area, neighbors, normals) tuples
    """
    elements = []
    for i in range(nel):
        elem = generate_mesh_element(i, nel, seed)
        elements.append(elem)
    return elements


def write_mesh_file(elements, output_file):
    """
    Write mesh data to file in CFD format.
    
    Args:
        elements: List of (area, neighbors, normals) tuples
        output_file: Output file path
    """
    nel = len(elements)
    
    with open(output_file, 'w') as f:
        # Line 1: Number of elements
        f.write(f"{nel:10d}\n")
        
        # Lines 2-nel+1: Element data
        for area, neighbors, normals in elements:
            # Format: area  nb1 nx1 ny1 nz1  nb2 nx2 ny2 nz2  nb3 nx3 ny3 nz3  nb4 nx4 ny4 nz4
            line_parts = [f"{area:18.7E}"]
            
            for j in range(NNB):
                nb = neighbors[j]
                nx = normals[j * NDIM + 0]
                ny = normals[j * NDIM + 1]
                nz = normals[j * NDIM + 2]
                line_parts.append(f"{nb:10d} {nx:18.7E} {ny:18.7E} {nz:18.7E}")
            
            f.write("  ".join(line_parts) + "\n")


def validate_mesh(elements, nel):
    """
    Perform lightweight validation on the mesh.
    
    Args:
        elements: List of (area, neighbors, normals) tuples
        nel: Expected number of elements
    
    Raises:
        ValueError: If validation fails
    """
    if len(elements) != nel:
        raise ValueError(f"Expected {nel} elements, got {len(elements)}")
    
    for i, (area, neighbors, normals) in enumerate(elements):
        if area <= 0:
            raise ValueError(f"Element {i}: area must be positive, got {area}")
        
        if len(neighbors) != NNB:
            raise ValueError(f"Element {i}: expected {NNB} neighbors, got {len(neighbors)}")
        
        if len(normals) != NNB * NDIM:
            raise ValueError(f"Element {i}: expected {NNB * NDIM} normal components, got {len(normals)}")
        
        # Check neighbor indices are valid or boundary (-1)
        for j, nb in enumerate(neighbors):
            if nb != -1 and (nb < 1 or nb > nel):
                raise ValueError(f"Element {i}, neighbor {j}: invalid index {nb} (must be -1 or 1-{nel})")


def cmd_synthetic(args):
    """Generate synthetic mesh data."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine nel
    if args.preset:
        if args.preset not in PRESETS:
            print(f"Error: Unknown preset '{args.preset}'. Available: {', '.join(PRESETS.keys())}", file=sys.stderr)
            sys.exit(1)
        nel, _ = PRESETS[args.preset]
        print(f"Using preset '{args.preset}': {nel} elements")
    elif args.nel:
        nel = args.nel
    else:
        print("Error: Must specify either --preset or --nel", file=sys.stderr)
        sys.exit(1)
    
    # Determine output filename
    if args.output_name:
        filename = args.output_name
    else:
        # Generate default filename based on nel
        if nel < 1000:
            size_str = f"{nel}"
        elif nel < 1000000:
            size_str = f"{nel // 1000:03d}K"
        else:
            size_str = f"{nel / 1000000:.1f}M"
        filename = f"fvcorr.domn.{size_str}"
    
    output_file = out_dir / filename
    
    # Check if file exists
    if output_file.exists() and not args.overwrite:
        print(f"Error: Output file '{output_file}' already exists. Use --overwrite to replace.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Generating mesh with {nel} elements...")
    elements = generate_mesh(nel, seed=args.seed)
    
    print(f"Validating mesh...")
    validate_mesh(elements, nel)
    
    print(f"Writing to '{output_file}'...")
    write_mesh_file(elements, output_file)
    
    file_size = output_file.stat().st_size
    print(f"✓ Successfully generated {nel} elements ({file_size / 1024 / 1024:.2f} MB)")
    
    # Optionally write metadata
    if args.write_meta:
        meta_file = output_file.with_suffix('.meta')
        with open(meta_file, 'w') as f:
            f.write(f"nel: {nel}\n")
            f.write(f"seed: {args.seed}\n")
            f.write(f"preset: {args.preset or 'custom'}\n")
        print(f"✓ Metadata written to '{meta_file}'")


def main():
    parser = argparse.ArgumentParser(
        description="CFD (Euler3D) Input Data Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets:
""" + "\n".join(f"  {name:12s} - {desc}" for name, desc in [(k, v[1]) for k, v in PRESETS.items()]) + """

Examples:
  # Generate mini dataset
  python generate.py synthetic --out-dir mini/input --preset mini

  # Generate small dataset
  python generate.py synthetic --out-dir small/input --preset small

  # Generate custom size
  python generate.py synthetic --out-dir custom/input --nel 50000 --seed 123
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Synthetic command
    syn_parser = subparsers.add_parser('synthetic', help='Generate synthetic mesh data')
    syn_parser.add_argument('--out-dir', type=str, required=True, help='Output directory (will be created if needed)')
    syn_parser.add_argument('--preset', type=str, choices=list(PRESETS.keys()), help='Preset configuration')
    syn_parser.add_argument('--nel', type=int, help='Number of elements (overrides preset)')
    syn_parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    syn_parser.add_argument('--output-name', type=str, help='Output filename (default: auto-generated)')
    syn_parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
    syn_parser.add_argument('--write-meta', action='store_true', help='Write metadata .meta file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'synthetic':
        cmd_synthetic(args)
    else:
        print(f"Error: Unknown command '{args.command}'", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

