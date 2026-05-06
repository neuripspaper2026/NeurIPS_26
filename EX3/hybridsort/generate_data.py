#!/usr/bin/env python3
import random
import os

# Dataset sizes (number of floats)
datasets = {
    'mini': 262144,        # 2^18
    'small': 524288,       # 2^19
    'medium': 1048576,     # 2^20
    'large': 2097152,      # 2^21
    'extra-large': 4194304 # 2^22
}

random.seed(42)  # Fixed seed for reproducibility

for name, size in datasets.items():
    filepath = f'input_data/{name}/input.txt'
    print(f'Generating {name}: {size} floats...')
    
    with open(filepath, 'w') as f:
        for i in range(size):
            f.write(f'{random.random():.6f}\n')
    
    print(f'  Saved to {filepath}')

print('Done!')

