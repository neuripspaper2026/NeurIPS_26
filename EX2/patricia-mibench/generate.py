#!/usr/bin/env python3
"""
Data generator for patricia-mibench

Generates UDP packet trace data for Patricia trie longest-prefix matching.
The input format is: timestamp src_ip dest_ip src_port dest_port (space-separated)
"""

import os
import sys
import argparse
import random
from pathlib import Path
from typing import Optional

# Size presets: (num_packets, num_unique_ips, num_ports)
PRESETS = {
    'mini': {
        'num_packets': 62721,
        'num_unique_ips': 1200,
        'num_ports': 1024,
        'description': 'Mini (62K packets)'
    },
    'small': {
        'num_packets': 900000,
        'num_unique_ips': 8000,
        'num_ports': 8192,
        'description': 'Small (900K packets)'
    },
    'medium': {
        'num_packets': 3000000,
        'num_unique_ips': 15000,
        'num_ports': 16384,
        'description': 'Medium (3M packets)'
    },
    'large': {
        'num_packets': 12000000,
        'num_unique_ips': 40000,
        'num_ports': 32768,
        'description': 'Large (12M packets)'
    },
    'extra-large': {
        'num_packets': 30000000,
        'num_unique_ips': 80000,
        'num_ports': 65535,
        'description': 'Extra-large (30M packets)'
    }
}


def generate_udp_trace(
    num_packets: int,
    num_unique_ips: int,
    num_ports: int,
    seed: Optional[int] = None
) -> list:
    """
    Generate synthetic UDP packet trace data.
    
    Args:
        num_packets: Number of packets to generate
        num_unique_ips: Number of unique IP addresses in the pool
        num_ports: Maximum port number (ports will be in range [1, num_ports])
        seed: Random seed for reproducibility
    
    Returns:
        List of packet tuples: (timestamp, src_ip, dest_ip, src_port, dest_port)
    """
    if seed is not None:
        random.seed(seed)
    
    # Generate IP pool (represented as integers 1 to num_unique_ips)
    ip_pool = list(range(1, num_unique_ips + 1))
    
    packets = []
    timestamp = 0.0
    
    for i in range(num_packets):
        # Increment timestamp (simulate packet arrival)
        # Random inter-arrival time between 0.0001 and 0.01 seconds
        timestamp += random.uniform(0.0001, 0.01)
        
        # Select source and destination IPs
        src_ip = random.choice(ip_pool)
        dest_ip = random.choice(ip_pool)
        
        # Select ports (biased towards common ports for realism)
        if random.random() < 0.3:
            # 30% chance of using common ports
            src_port = random.choice([53, 80, 443, 22, 21, 25])
        else:
            src_port = random.randint(1, num_ports)
        
        if random.random() < 0.3:
            dest_port = random.choice([53, 80, 443, 22, 21, 25])
        else:
            dest_port = random.randint(1, num_ports)
        
        packets.append((timestamp, src_ip, dest_ip, src_port, dest_port))
    
    return packets


def write_trace_to_file(packets: list, output_path: Path) -> None:
    """
    Write UDP trace data to file.
    Format: timestamp src_ip dest_ip src_port dest_port (space-separated)
    
    Args:
        packets: List of packet tuples
        output_path: Path to output file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for timestamp, src_ip, dest_ip, src_port, dest_port in packets:
            f.write(f"{timestamp:.6f} {src_ip} {dest_ip} {src_port} {dest_port}\n")


def validate_config(num_packets: int, num_unique_ips: int, num_ports: int) -> None:
    """Validate generation parameters."""
    if num_packets < 1:
        raise ValueError(f"num_packets must be at least 1, got {num_packets}")
    if num_packets > 50000000:
        raise ValueError(f"num_packets too large (max 50M), got {num_packets}")
    if num_unique_ips < 2:
        raise ValueError(f"num_unique_ips must be at least 2, got {num_unique_ips}")
    if num_unique_ips > 1000000:
        raise ValueError(f"num_unique_ips too large (max 1M), got {num_unique_ips}")
    if num_ports < 1 or num_ports > 65535:
        raise ValueError(f"num_ports must be in [1, 65535], got {num_ports}")


def cmd_synthetic(args):
    """Generate synthetic UDP trace data."""
    
    # Get configuration
    if args.preset:
        if args.preset not in PRESETS:
            print(f"Error: Unknown preset '{args.preset}'", file=sys.stderr)
            print(f"Available presets: {', '.join(PRESETS.keys())}", file=sys.stderr)
            return 1
        
        config = PRESETS[args.preset]
        num_packets = config['num_packets']
        num_unique_ips = config['num_unique_ips']
        num_ports = config['num_ports']
        
        # Allow override
        if args.num_packets:
            num_packets = args.num_packets
        if args.num_ips:
            num_unique_ips = args.num_ips
        if args.num_ports:
            num_ports = args.num_ports
    else:
        # No preset, require explicit parameters
        if not args.num_packets:
            print("Error: --num-packets required when not using --preset", file=sys.stderr)
            return 1
        num_packets = args.num_packets
        num_unique_ips = args.num_ips if args.num_ips else 1000
        num_ports = args.num_ports if args.num_ports else 1024
    
    # Validate
    try:
        validate_config(num_packets, num_unique_ips, num_ports)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    # Generate trace
    print(f"Generating UDP trace with {num_packets:,} packets...")
    print(f"  Unique IPs: {num_unique_ips:,}")
    print(f"  Port range: 1-{num_ports}")
    print(f"  Seed: {args.seed if args.seed is not None else 'random'}")
    
    packets = generate_udp_trace(num_packets, num_unique_ips, num_ports, args.seed)
    
    # Calculate statistics
    total_ips = len(set([p[1] for p in packets] + [p[2] for p in packets]))
    total_ports = len(set([p[3] for p in packets] + [p[4] for p in packets]))
    duration = packets[-1][0] if packets else 0.0
    
    print(f"  Generated packets: {len(packets):,}")
    print(f"  Actual unique IPs: {total_ips}")
    print(f"  Actual unique ports: {total_ports}")
    print(f"  Time span: {duration:.2f}s")
    
    # Write to file
    out_dir = Path(args.out_dir)
    output_path = out_dir / "input.udp"
    
    write_trace_to_file(packets, output_path)
    
    file_size = output_path.stat().st_size
    print(f"  Output: {output_path}")
    print(f"  Size: {file_size:,} bytes ({file_size / 1024 / 1024:.1f} MB)")
    
    # Also create output directory for benchmark results
    (out_dir.parent / "output").mkdir(parents=True, exist_ok=True)
    
    print("✓ Done")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Generate UDP packet trace data for patricia-mibench',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate mini preset (62K packets)
  python generate.py synthetic --preset mini --out-dir input_data/mini/input

  # Generate all presets
  for size in mini small medium large extra-large; do
    python generate.py synthetic --preset $size --out-dir input_data/$size/input --seed 42
  done

  # Custom configuration
  python generate.py synthetic --num-packets 100000 --num-ips 1500 --num-ports 2048 \\
    --out-dir input_data/custom/input --seed 12345

Available presets:
""" + '\n'.join(f"  {k}: {v['description']}" for k, v in PRESETS.items())
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Synthetic data generation
    syn_parser = subparsers.add_parser('synthetic', help='Generate synthetic UDP trace')
    syn_parser.add_argument('--preset', choices=PRESETS.keys(),
                           help='Use preset configuration')
    syn_parser.add_argument('--num-packets', type=int,
                           help='Number of packets to generate (overrides preset)')
    syn_parser.add_argument('--num-ips', type=int,
                           help='Number of unique IP addresses (overrides preset)')
    syn_parser.add_argument('--num-ports', type=int,
                           help='Maximum port number (overrides preset)')
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

