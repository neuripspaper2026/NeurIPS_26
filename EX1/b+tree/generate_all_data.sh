#!/bin/bash
# Generate all B+Tree input data sizes

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================="
echo "B+Tree Data Generator - Generate All Sizes"
echo "========================================="
echo ""

# Check if generate.py exists
if [ ! -f "generate.py" ]; then
    echo "Error: generate.py not found in current directory"
    exit 1
fi

# Array of sizes to generate
SIZES=("mini" "small" "medium" "large" "extra-large")

# Generate each size
for size in "${SIZES[@]}"; do
    echo "----------------------------------------"
    echo "Generating $size dataset..."
    echo "----------------------------------------"
    
    python3 generate.py synthetic \
        --out-dir "input_data/$size/input" \
        --preset "$size" \
        --overwrite \
        --write-meta
    
    echo ""
done

echo "========================================="
echo "✓ All datasets generated successfully!"
echo "========================================="
echo ""
echo "Dataset summary:"
for size in "${SIZES[@]}"; do
    if [ -f "input_data/$size/input/mil.txt" ]; then
        size_mb=$(du -sh "input_data/$size/input/mil.txt" | cut -f1)
        lines=$(wc -l < "input_data/$size/input/mil.txt")
        keys=$((lines - 1))
        echo "  $size: $keys keys, $size_mb"
    fi
done
echo ""
echo "To run benchmarks, see: input_data/input_data.txt"

