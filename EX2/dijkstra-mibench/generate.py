#!/usr/bin/env python3
"""
Data generator for dijkstra-mibench

Generates adjacency matrix data files for Dijkstra shortest path algorithm.
The input format is a 100x100 matrix of integers representing edge weights.
"""

import os
import sys
import argparse
import random
from pathlib import Path
from typing import Tuple, Optional

# Size presets: (num_nodes, sparsity, max_weight)
# sparsity: probability that an edge exists (0.0 = no edges, 1.0 = fully connected)
PRESETS = {
    'mini': {
        'num_nodes': 100,
        'sparsity': 0.3,
        'max_weight': 100,
        'description': 'Mini (100 nodes, 30% sparse)'
    },
    'small': {
        'num_nodes': 400,
        'sparsity': 0.20,
        'max_weight': 100,
        'description': 'Small (400 nodes, 20% sparse)'
    },
    'medium': {
        'num_nodes': 1000,
        'sparsity': 0.12,
        'max_weight': 100,
        'description': 'Medium (1000 nodes, 12% sparse)'
    },
    'large': {
        'num_nodes': 2000,
        'sparsity': 0.08,
        'max_weight': 100,
        'description': 'Large (2000 nodes, 8% sparse)'
    },
    'extra-large': {
        'num_nodes': 4000,
        'sparsity': 0.04,
        'max_weight': 100,
        'description': 'Extra-large (4000 nodes, 4% sparse)'
    }
}

NONE = 9999  # Special value indicating no edge


def generate_adjacency_matrix(
    num_nodes: int,
    sparsity: float,
    max_weight: int,
    seed: Optional[int] = None
) -> list:
    """
    Generate a random adjacency matrix for a directed graph.
    
    Args:
        num_nodes: Number of nodes in the graph
        sparsity: Probability that an edge exists (0.0 to 1.0)
        max_weight: Maximum edge weight (weights are in [1, max_weight])
        seed: Random seed for reproducibility
    
    Returns:
        2D list representing the adjacency matrix
    """
    if seed is not None:
        random.seed(seed)
    
    matrix = []
    for i in range(num_nodes):
        row = []
        for j in range(num_nodes):
            if i == j:
                # No self-loops
                row.append(NONE)
            elif random.random() < sparsity:
                # Edge exists with random weight
                weight = random.randint(1, max_weight)
                row.append(weight)
            else:
                # No edge
                row.append(NONE)
        matrix.append(row)
    
    return matrix


def write_matrix_to_file(matrix: list, output_path: Path) -> None:
    """
    Write adjacency matrix to file in the format expected by dijkstra.
    Format: space-separated integers, 10 values per line.
    
    Args:
        matrix: 2D list of integers
        output_path: Path to output file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    num_nodes = len(matrix)
    
    with open(output_path, 'w') as f:
        values_per_line = 10
        all_values = []
        
        # Flatten matrix into a list
        for row in matrix:
            all_values.extend(row)
        
        # Write values in groups of 10 per line
        for i in range(0, len(all_values), values_per_line):
            chunk = all_values[i:i + values_per_line]
            line = ' '.join(str(v) for v in chunk)
            f.write(line + ' \n')


def validate_config(num_nodes: int, sparsity: float, max_weight: int) -> None:
    """Validate generation parameters."""
    if num_nodes < 2:
        raise ValueError(f"num_nodes must be at least 2, got {num_nodes}")
    if num_nodes > 10000:
        raise ValueError(f"num_nodes too large (max 10000), got {num_nodes}")
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError(f"sparsity must be in [0.0, 1.0], got {sparsity}")
    if max_weight < 1 or max_weight > 9998:
        raise ValueError(f"max_weight must be in [1, 9998], got {max_weight}")


def cmd_synthetic(args):
    """Generate synthetic adjacency matrix data."""
    
    # Get configuration
    if args.preset:
        if args.preset not in PRESETS:
            print(f"Error: Unknown preset '{args.preset}'", file=sys.stderr)
            print(f"Available presets: {', '.join(PRESETS.keys())}", file=sys.stderr)
            return 1
        
        config = PRESETS[args.preset]
        num_nodes = config['num_nodes']
        sparsity = config['sparsity']
        max_weight = config['max_weight']
        
        # Allow override
        if args.num_nodes:
            num_nodes = args.num_nodes
        if args.sparsity is not None:
            sparsity = args.sparsity
        if args.max_weight:
            max_weight = args.max_weight
    else:
        # No preset, require explicit parameters
        if not args.num_nodes:
            print("Error: --num-nodes required when not using --preset", file=sys.stderr)
            return 1
        num_nodes = args.num_nodes
        sparsity = args.sparsity if args.sparsity is not None else 0.2
        max_weight = args.max_weight if args.max_weight else 100
    
    # Validate
    try:
        validate_config(num_nodes, sparsity, max_weight)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    # Generate matrix
    print(f"Generating {num_nodes}x{num_nodes} adjacency matrix...")
    print(f"  Sparsity: {sparsity:.2%}")
    print(f"  Max weight: {max_weight}")
    print(f"  Seed: {args.seed if args.seed is not None else 'random'}")
    
    matrix = generate_adjacency_matrix(num_nodes, sparsity, max_weight, args.seed)
    
    # Calculate statistics
    total_edges = sum(1 for row in matrix for val in row if val != NONE)
    max_possible = num_nodes * (num_nodes - 1)  # Excluding diagonal
    actual_sparsity = total_edges / max_possible if max_possible > 0 else 0
    
    print(f"  Generated edges: {total_edges} ({actual_sparsity:.2%} of max {max_possible})")
    
    # Write to file
    out_dir = Path(args.out_dir)
    output_path = out_dir / "input.dat"
    
    write_matrix_to_file(matrix, output_path)
    
    file_size = output_path.stat().st_size
    print(f"  Output: {output_path}")
    print(f"  Size: {file_size:,} bytes")
    
    # Also create output directory for benchmark results
    (out_dir.parent / "output").mkdir(parents=True, exist_ok=True)
    
    print("✓ Done")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Generate adjacency matrix data for dijkstra-mibench',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate mini preset (100 nodes)
  python generate.py synthetic --preset mini --out-dir input_data/mini/input

  # Generate all presets
  for size in mini small medium large extra-large; do
    python generate.py synthetic --preset $size --out-dir input_data/$size/input --seed 42
  done

  # Custom configuration
  python generate.py synthetic --num-nodes 300 --sparsity 0.2 --max-weight 50 \\
    --out-dir input_data/custom/input --seed 12345

Available presets:
""" + '\n'.join(f"  {k}: {v['description']}" for k, v in PRESETS.items())
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Synthetic data generation
    syn_parser = subparsers.add_parser('synthetic', help='Generate synthetic adjacency matrix')
    syn_parser.add_argument('--preset', choices=PRESETS.keys(),
                           help='Use preset configuration')
    syn_parser.add_argument('--num-nodes', type=int,
                           help='Number of nodes in graph (overrides preset)')
    syn_parser.add_argument('--sparsity', type=float,
                           help='Edge probability 0.0-1.0 (overrides preset)')
    syn_parser.add_argument('--max-weight', type=int,
                           help='Maximum edge weight (overrides preset)')
    syn_parser.add_argument('--seed', type=int,
                           help='Random seed for reproducibility')
    syn_parser.add_argument('--out-dir', required=True,
                           help='Output directory for generated data')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command == 'synthetic':
        return cmd_synthetic(args)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

