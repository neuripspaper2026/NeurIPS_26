#!/usr/bin/env python3
"""
B+Tree Input Data Generator

Generates input data files for the B+Tree benchmark.

Input format:
- mil.txt: First line is count N, followed by N integers (keys to insert)
- command.txt: Test commands (e.g., "j <start> <end>" for range query, "k <key>" for search)
"""

import argparse
import sys
import os
import random
from pathlib import Path

# Preset configurations: (num_keys, description)
PRESETS = {
    "mini": (1_000_000, "Mini dataset (1M keys)"),
    "small": (4_000_000, "Small dataset (4M keys)"),
    "medium": (8_000_000, "Medium dataset (8M keys)"),
    "large": (20_000_000, "Large dataset (20M keys)"),
    "extra-large": (24_000_000, "Extra-large dataset (24M keys)"),
}

# Command presets based on dataset size
# Note: The 65,535 limit in the code is a CUDA artifact but doesn't apply to CPU-only execution
# For CPU-only (no GPU, no parallel), we can use larger query counts
COMMAND_PRESETS = {
    "mini": [
        ("j", 6000, 3000),      # Range query from 6000 with 3000 entries
        ("k", 10000, None),     # 10,000 random point queries
    ],
    "small": [
        ("j", 20000, 10000),
        ("k", 50000, None),     # 50,000 random point queries
    ],
    "medium": [
        ("j", 40000, 20000),
        ("k", 100000, None),    # 100,000 random point queries
    ],
    "large": [
        ("j", 100000, 50000),
        ("k", 200000, None),    # 200,000 random point queries
    ],
    "extra-large": [
        ("j", 120000, 60000),
        ("k", 300000, None),    # 300,000 random point queries
    ],
}


def generate_keys_sequential(num_keys, seed=42):
    """Generate sequential keys from 0 to num_keys-1."""
    return list(range(num_keys))


def generate_keys_random(num_keys, seed=42):
    """Generate random unique keys."""
    rng = random.Random(seed)
    keys = list(range(num_keys * 2))  # Larger pool
    rng.shuffle(keys)
    return keys[:num_keys]


def generate_keys_mixed(num_keys, seed=42):
    """Generate mixed pattern: 80% sequential, 20% random gaps."""
    rng = random.Random(seed)
    sequential = int(num_keys * 0.8)
    random_count = num_keys - sequential
    
    keys = list(range(sequential))
    # Add random keys with gaps
    max_key = num_keys * 3
    random_keys = set()
    while len(random_keys) < random_count:
        random_keys.add(rng.randint(sequential, max_key))
    
    keys.extend(sorted(random_keys))
    return keys


def write_mil_file(keys, output_file):
    """Write keys to mil.txt format."""
    num_keys = len(keys)
    with open(output_file, 'w') as f:
        f.write(f"{num_keys}\n")
        for key in keys:
            f.write(f"{key}\n")


def write_command_file(commands, output_file):
    """Write commands to command.txt format."""
    with open(output_file, 'w') as f:
        for cmd in commands:
            if cmd[0] == 'j':  # Range query
                f.write(f"j {cmd[1]} {cmd[2]}\n")
            elif cmd[0] == 'k':  # Point query
                f.write(f"k {cmd[1]}\n")
            else:
                f.write(f"{cmd[0]}\n")


def validate_keys(keys, num_keys):
    """Validate generated keys."""
    if len(keys) != num_keys:
        raise ValueError(f"Expected {num_keys} keys, got {len(keys)}")
    
    # Check for valid integers
    for i, key in enumerate(keys):
        if not isinstance(key, int):
            raise ValueError(f"Key at index {i} is not an integer: {key}")
        if key < 0:
            raise ValueError(f"Key at index {i} is negative: {key}")


def cmd_synthetic(args):
    """Generate synthetic B+Tree input data."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine number of keys
    if args.preset:
        if args.preset not in PRESETS:
            print(f"Error: Unknown preset '{args.preset}'. Available: {', '.join(PRESETS.keys())}", file=sys.stderr)
            sys.exit(1)
        num_keys, _ = PRESETS[args.preset]
        print(f"Using preset '{args.preset}': {num_keys:,} keys")
        commands = COMMAND_PRESETS.get(args.preset, [("k", num_keys // 2, None)])
    elif args.num_keys:
        num_keys = args.num_keys
        # Default commands for custom size
        commands = [("j", num_keys // 10, num_keys // 20), ("k", num_keys // 2, None)]
    else:
        print("Error: Must specify either --preset or --num-keys", file=sys.stderr)
        sys.exit(1)
    
    # Determine output filenames
    mil_file = out_dir / (args.mil_name or "mil.txt")
    command_file = out_dir / (args.command_name or "command.txt")
    
    # Check overwrite
    if mil_file.exists() and not args.overwrite:
        print(f"Error: Output file '{mil_file}' already exists. Use --overwrite to replace.", file=sys.stderr)
        sys.exit(1)
    if command_file.exists() and not args.overwrite:
        print(f"Error: Output file '{command_file}' already exists. Use --overwrite to replace.", file=sys.stderr)
        sys.exit(1)
    
    # Generate keys based on pattern
    print(f"Generating {num_keys:,} keys (pattern: {args.pattern})...")
    if args.pattern == "sequential":
        keys = generate_keys_sequential(num_keys, seed=args.seed)
    elif args.pattern == "random":
        keys = generate_keys_random(num_keys, seed=args.seed)
    elif args.pattern == "mixed":
        keys = generate_keys_mixed(num_keys, seed=args.seed)
    else:
        print(f"Error: Unknown pattern '{args.pattern}'", file=sys.stderr)
        sys.exit(1)
    
    # Validate
    print(f"Validating keys...")
    validate_keys(keys, num_keys)
    
    # Write mil.txt
    print(f"Writing keys to '{mil_file}'...")
    write_mil_file(keys, mil_file)
    
    # Write command.txt
    print(f"Writing commands to '{command_file}'...")
    write_command_file(commands, command_file)
    
    # Report
    mil_size = mil_file.stat().st_size
    cmd_size = command_file.stat().st_size
    print(f"✓ Successfully generated:")
    print(f"  - {mil_file} ({mil_size / 1024 / 1024:.2f} MB, {num_keys:,} keys)")
    print(f"  - {command_file} ({cmd_size} bytes, {len(commands)} commands)")
    
    # Write metadata if requested
    if args.write_meta:
        meta_file = out_dir / "dataset.meta"
        with open(meta_file, 'w') as f:
            f.write(f"num_keys: {num_keys}\n")
            f.write(f"pattern: {args.pattern}\n")
            f.write(f"seed: {args.seed}\n")
            f.write(f"preset: {args.preset or 'custom'}\n")
        print(f"✓ Metadata written to '{meta_file}'")


def main():
    parser = argparse.ArgumentParser(
        description="B+Tree Input Data Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets:
""" + "\n".join(f"  {name:12s} - {desc}" for name, desc in [(k, v[1]) for k, v in PRESETS.items()]) + """

Examples:
  # Generate mini dataset (sequential keys)
  python generate.py synthetic --out-dir input_data/mini/input --preset mini

  # Generate small dataset with random keys
  python generate.py synthetic --out-dir input_data/small/input --preset small --pattern random

  # Generate custom size
  python generate.py synthetic --out-dir input_data/custom/input --num-keys 5000000 --seed 123
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Synthetic data generation
    syn_parser = subparsers.add_parser('synthetic', help='Generate synthetic B+Tree input data')
    syn_parser.add_argument('--out-dir', type=str, required=True,
                           help='Output directory (will be created if needed)')
    syn_parser.add_argument('--preset', type=str, choices=list(PRESETS.keys()),
                           help='Preset configuration')
    syn_parser.add_argument('--num-keys', type=int,
                           help='Number of keys to generate (overrides preset)')
    syn_parser.add_argument('--pattern', type=str, default='sequential',
                           choices=['sequential', 'random', 'mixed'],
                           help='Key generation pattern (default: sequential)')
    syn_parser.add_argument('--seed', type=int, default=42,
                           help='Random seed (default: 42)')
    syn_parser.add_argument('--mil-name', type=str,
                           help='Output filename for keys (default: mil.txt)')
    syn_parser.add_argument('--command-name', type=str,
                           help='Output filename for commands (default: command.txt)')
    syn_parser.add_argument('--overwrite', action='store_true',
                           help='Overwrite existing files')
    syn_parser.add_argument('--write-meta', action='store_true',
                           help='Write metadata .meta file')
    
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

