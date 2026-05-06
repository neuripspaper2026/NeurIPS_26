#!/usr/bin/env python3
"""
Canneal Benchmark Input Generator

Generates netlist files and parameter configurations for different benchmark sizes.

Format: First line contains: NUM_ELEMENTS GRID_WIDTH GRID_HEIGHT
Following lines: ELEMENT_NAME TYPE CONNECTIONS... END
"""

import os
import random
import argparse
from pathlib import Path


def generate_netlist(num_elements, grid_width, grid_height, output_file, seed=42):
    """
    Generate a netlist file for canneal benchmark.
    
    Args:
        num_elements: Number of circuit elements
        grid_width: Width of placement grid
        grid_height: Height of placement grid
        output_file: Output file path
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Generate element names (a, b, c, ..., aa, ab, ...)
    def get_element_name(index):
        name = ""
        while index >= 0:
            name = chr(ord('a') + (index % 26)) + name
            index = index // 26 - 1
        return name
    
    element_names = [get_element_name(i) for i in range(num_elements)]
    
    # Write netlist file
    with open(output_file, 'w') as f:
        # Header: num_elements grid_width grid_height
        f.write(f"{num_elements}\t{grid_width}\t{grid_height}\n")
        
        # Generate each element
        for i, name in enumerate(element_names):
            # Type: 1 or 2 (randomly chosen)
            elem_type = random.randint(1, 2)
            
            # Number of connections (typically 3-6)
            num_connections = random.randint(3, 6)
            
            # Generate random connections to other elements
            connections = []
            for _ in range(num_connections):
                # Choose a random element (can be self)
                conn_idx = random.randint(0, num_elements - 1)
                connections.append(element_names[conn_idx])
            
            # Write element line
            line = f"{name}\t{elem_type}\t" + "\t".join(connections) + "\tEND\n"
            f.write(line)
    
    print(f"Generated netlist: {output_file}")
    print(f"  - Elements: {num_elements}")
    print(f"  - Grid: {grid_width}x{grid_height}")


def create_param_file(output_dir, nthreads, nswaps, temp, netlist_path, nsteps):
    """Create a parameter file for easy reference."""
    param_file = os.path.join(output_dir, "params.txt")
    with open(param_file, 'w') as f:
        f.write(f"# Canneal Parameters\n")
        f.write(f"NTHREADS={nthreads}\n")
        f.write(f"NSWAPS={nswaps}\n")
        f.write(f"TEMP={temp}\n")
        f.write(f"NETLIST={netlist_path}\n")
        f.write(f"NSTEPS={nsteps}\n")
        f.write(f"\n# Command:\n")
        f.write(f"# ./canneal {nthreads} {nswaps} {temp} {netlist_path} {nsteps}\n")
    print(f"Created parameter file: {param_file}")


# Predefined size configurations
# Scaled so that current medium (~30s) becomes extra-large
SIZE_CONFIGS = {
    'mini': {
        'num_elements': 100,
        'grid_width': 15,
        'grid_height': 15,
        'nthreads': 1,
        'nswaps': 1000,
        'temp': 2000,
        'nsteps': 50,
        'seed': 42
    },
    'small': {
        'num_elements': 500,
        'grid_width': 30,
        'grid_height': 30,
        'nthreads': 1,
        'nswaps': 50000,
        'temp': 2500,
        'nsteps': 150,
        'seed': 43
    },
    'medium': {
        'num_elements': 2000,
        'grid_width': 60,
        'grid_height': 60,
        'nthreads': 1,
        'nswaps': 200000,
        'temp': 3000,
        'nsteps': 400,
        'seed': 44
    },
    'large': {
        'num_elements': 5000,
        'grid_width': 100,
        'grid_height': 100,
        'nthreads': 1,
        'nswaps': 500000,
        'temp': 3500,
        'nsteps': 700,
        'seed': 45
    },
    'extra-large': {
        'num_elements': 10000,
        'grid_width': 150,
        'grid_height': 150,
        'nthreads': 1,
        'nswaps': 1000000,
        'temp': 4000,
        'nsteps': 1000,
        'seed': 46
    }
}


def generate_all_sizes(base_dir='input_data'):
    """Generate netlist files for all predefined sizes."""
    base_path = Path(base_dir)
    
    print("=" * 60)
    print("Canneal Benchmark Input Generator")
    print("=" * 60)
    print()
    
    for size_name, config in SIZE_CONFIGS.items():
        print(f"Generating {size_name} configuration...")
        
        # Create directories
        size_dir = base_path / size_name
        input_dir = size_dir / 'input'
        output_dir = size_dir / 'output'
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate netlist
        netlist_file = input_dir / f'input_{size_name}.nets'
        generate_netlist(
            num_elements=config['num_elements'],
            grid_width=config['grid_width'],
            grid_height=config['grid_height'],
            output_file=str(netlist_file),
            seed=config['seed']
        )
        
        # Create parameter file
        netlist_rel_path = f"input_data/{size_name}/input/input_{size_name}.nets"
        create_param_file(
            output_dir=str(input_dir),
            nthreads=config['nthreads'],
            nswaps=config['nswaps'],
            temp=config['temp'],
            netlist_path=netlist_rel_path,
            nsteps=config['nsteps']
        )
        
        print()
    
    print("=" * 60)
    print("All configurations generated successfully!")
    print("=" * 60)
    print()
    print("Size Summary:")
    print(f"{'Size':<12} {'Elements':<10} {'Grid':<12} {'Swaps':<10} {'Steps':<8}")
    print("-" * 60)
    for size_name, config in SIZE_CONFIGS.items():
        grid_str = f"{config['grid_width']}x{config['grid_height']}"
        print(f"{size_name:<12} {config['num_elements']:<10} {grid_str:<12} "
              f"{config['nswaps']:<10} {config['nsteps']:<8}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate netlist files for Canneal benchmark'
    )
    parser.add_argument(
        '--size',
        choices=['mini', 'small', 'medium', 'large', 'extra-large', 'all'],
        default='all',
        help='Size configuration to generate (default: all)'
    )
    parser.add_argument(
        '--custom',
        action='store_true',
        help='Generate custom configuration (requires additional args)'
    )
    parser.add_argument(
        '--elements',
        type=int,
        help='Number of elements (for custom mode)'
    )
    parser.add_argument(
        '--grid-width',
        type=int,
        help='Grid width (for custom mode)'
    )
    parser.add_argument(
        '--grid-height',
        type=int,
        help='Grid height (for custom mode)'
    )
    parser.add_argument(
        '--output',
        help='Output file path (for custom mode)'
    )
    
    args = parser.parse_args()
    
    if args.custom:
        if not all([args.elements, args.grid_width, args.grid_height, args.output]):
            parser.error("Custom mode requires --elements, --grid-width, "
                        "--grid-height, and --output")
        generate_netlist(args.elements, args.grid_width, args.grid_height, args.output)
    else:
        if args.size == 'all':
            generate_all_sizes()
        else:
            config = SIZE_CONFIGS[args.size]
            size_dir = Path('input_data') / args.size
            input_dir = size_dir / 'input'
            input_dir.mkdir(parents=True, exist_ok=True)
            netlist_file = input_dir / f'input_{args.size}.nets'
            generate_netlist(
                config['num_elements'],
                config['grid_width'],
                config['grid_height'],
                str(netlist_file),
                config['seed']
            )


if __name__ == '__main__':
    main()

