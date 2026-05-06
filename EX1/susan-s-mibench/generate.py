#!/usr/bin/env python3
"""
Data Generator for susan-s-mibench

Generates PGM (Portable Gray Map) image files for the SUSAN corner/edge detector benchmark.
Supports synthetic image generation with various sizes and patterns.

PGM Format (P5 - binary):
- Header: "P5\n"
- Comment: "# CREATOR: ...\n" (optional)
- Dimensions: "width height\n"
- Max gray value: "255\n"
- Binary pixel data: width×height bytes (0-255)
"""

import argparse
import os
import sys
import random
import math
from pathlib import Path

# Preset configurations (matching original benchmarks + additional sizes)
PRESETS = {
    'mini': {'width': 76, 'height': 95, 'description': 'Mini size (matches input_small.pgm)'},
    'small': {'width': 2048, 'height': 1536, 'description': 'Small size'},
    'medium': {'width': 4096, 'height': 3072, 'description': 'Medium size'},
    'large': {'width': 8192, 'height': 6144, 'description': 'Large size'},
    'extra-large': {'width': 16384, 'height': 12288, 'description': 'Extra-large size'},
}


def validate_params(width, height):
    """Validate image parameters."""
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid dimensions: {width}x{height} (must be positive)")
    if width > 32768 or height > 32768:
        raise ValueError(f"Dimensions too large: {width}x{height} (max 32768x32768)")
    return True


def generate_synthetic_image(width, height, pattern='random', seed=12345):
    """
    Generate synthetic grayscale image data.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        pattern: Pattern type ('random', 'gradient', 'checkerboard', 'edges', 'corners')
        seed: Random seed for reproducibility
        
    Returns:
        bytearray of pixel data (width×height bytes)
    """
    random.seed(seed)
    pixels = bytearray(width * height)
    
    if pattern == 'random':
        # Pure random noise
        for i in range(width * height):
            pixels[i] = random.randint(0, 255)
            
    elif pattern == 'gradient':
        # Horizontal gradient
        for y in range(height):
            for x in range(width):
                value = int((x / width) * 255)
                pixels[y * width + x] = value
                
    elif pattern == 'checkerboard':
        # Checkerboard pattern (8x8 squares)
        square_size = 8
        for y in range(height):
            for x in range(width):
                if ((x // square_size) + (y // square_size)) % 2 == 0:
                    pixels[y * width + x] = 255
                else:
                    pixels[y * width + x] = 0
                    
    elif pattern == 'edges':
        # Horizontal and vertical edges
        for y in range(height):
            for x in range(width):
                base = random.randint(80, 120)
                # Add edges every 32 pixels
                if x % 32 < 2 or y % 32 < 2:
                    base = random.randint(200, 255)
                pixels[y * width + x] = base
                
    elif pattern == 'corners':
        # Corner-like features for SUSAN detector
        for y in range(height):
            for x in range(width):
                base = random.randint(100, 150)
                # Add corner features
                cx, cy = width // 2, height // 2
                dx, dy = abs(x - cx), abs(y - cy)
                if (dx * dx + dy * dy) < (min(width, height) // 4) ** 2:
                    base = random.randint(50, 100)
                # Add some edges
                if x % 48 < 3 or y % 48 < 3:
                    base = random.randint(180, 220)
                pixels[y * width + x] = base
                
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    
    return pixels


def encode_pgm(width, height, pixels, creator_comment=None):
    """
    Encode image data to PGM P5 format (binary).
    
    Args:
        width: Image width
        height: Image height
        pixels: bytearray of pixel data (width×height bytes)
        creator_comment: Optional creator comment
        
    Returns:
        bytes: Complete PGM file data
    """
    if len(pixels) != width * height:
        raise ValueError(f"Pixel count mismatch: expected {width*height}, got {len(pixels)}")
    
    # Build header
    header = b"P5\n"
    if creator_comment:
        header += f"# {creator_comment}\n".encode('ascii')
    else:
        header += b"# CREATOR: generate.py (susan-s-mibench data generator)\n"
    header += f"{width} {height}\n".encode('ascii')
    header += b"255\n"
    
    # Combine header and pixel data
    return header + bytes(pixels)


def write_pgm(file_path, pgm_data, overwrite=False):
    """
    Write PGM data to file.
    
    Args:
        file_path: Output file path
        pgm_data: Complete PGM file data (bytes)
        overwrite: Whether to overwrite existing files
    """
    file_path = Path(file_path)
    
    if file_path.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {file_path} (use --overwrite)")
    
    # Create parent directories
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write file
    with open(file_path, 'wb') as f:
        f.write(pgm_data)
    
    file_size = len(pgm_data)
    print(f"✓ Generated {file_path} ({file_size:,} bytes)")


def cmd_synthetic(args):
    """Generate synthetic PGM image(s)."""
    # Handle preset
    if args.preset:
        if args.preset not in PRESETS:
            print(f"Error: Unknown preset '{args.preset}'", file=sys.stderr)
            print(f"Available presets: {', '.join(PRESETS.keys())}", file=sys.stderr)
            return 1
        preset = PRESETS[args.preset]
        width = args.width if args.width else preset['width']
        height = args.height if args.height else preset['height']
    else:
        if not args.width or not args.height:
            print("Error: Must specify --width and --height or use --preset", file=sys.stderr)
            return 1
        width = args.width
        height = args.height
    
    # Validate parameters
    try:
        validate_params(width, height)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    # Determine output path
    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        if args.name:
            filename = args.name if args.name.endswith('.pgm') else f"{args.name}.pgm"
        elif args.preset:
            filename = f"input_{args.preset}.pgm"
        else:
            filename = f"input_{width}x{height}.pgm"
        
        output_path = out_dir / filename
    else:
        print("Error: --out-dir is required", file=sys.stderr)
        return 1
    
    # Generate image
    print(f"Generating {width}×{height} PGM image (pattern: {args.pattern}, seed: {args.seed})...")
    try:
        pixels = generate_synthetic_image(width, height, args.pattern, args.seed)
        pgm_data = encode_pgm(width, height, pixels)
        write_pgm(output_path, pgm_data, args.overwrite)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0


def cmd_info(args):
    """Display preset information."""
    print("Available presets for susan-s-mibench:\n")
    for name, config in PRESETS.items():
        width, height = config['width'], config['height']
        size_mb = (width * height + 100) / 1024 / 1024  # Approximate with header
        print(f"  {name:12s}  {width:4d}×{height:3d}  ~{size_mb:5.2f} MB  {config['description']}")
    print("\nPatterns: random, gradient, checkerboard, edges, corners")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Data generator for susan-s-mibench (PGM image generator)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate mini size with random pattern
  python generate.py synthetic --preset mini --out-dir input_data/mini/input
  
  # Generate large size with corners pattern
  python generate.py synthetic --preset large --pattern corners --out-dir input_data/large/input
  
  # Generate custom size
  python generate.py synthetic --width 640 --height 480 --out-dir custom/
  
  # List available presets
  python generate.py info
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # Synthetic subcommand
    synthetic_parser = subparsers.add_parser('synthetic', help='Generate synthetic PGM images')
    synthetic_parser.add_argument('--preset', choices=list(PRESETS.keys()),
                                   help='Preset configuration (mini/small/medium/large/extra-large)')
    synthetic_parser.add_argument('--width', type=int, help='Image width (overrides preset)')
    synthetic_parser.add_argument('--height', type=int, help='Image height (overrides preset)')
    synthetic_parser.add_argument('--pattern', default='random',
                                   choices=['random', 'gradient', 'checkerboard', 'edges', 'corners'],
                                   help='Image pattern type (default: random)')
    synthetic_parser.add_argument('--seed', type=int, default=12345,
                                   help='Random seed for reproducibility (default: 12345)')
    synthetic_parser.add_argument('--out-dir', required=True, help='Output directory')
    synthetic_parser.add_argument('--name', help='Output filename (default: auto-generated)')
    synthetic_parser.add_argument('--overwrite', action='store_true',
                                   help='Overwrite existing files')
    
    # Info subcommand
    info_parser = subparsers.add_parser('info', help='Show preset information')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command == 'synthetic':
        return cmd_synthetic(args)
    elif args.command == 'info':
        return cmd_info(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())

