#!/bin/bash
#
# Test Canneal Benchmark with All Sizes
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

BINARY="./EX1_optimized_codes/canneal_gcc"
OUTPUT_DIR="./EX5_correctness"

# Check if binary exists
if [ ! -f "$BINARY" ]; then
    echo -e "${RED}Error: Binary not found at $BINARY${NC}"
    echo "Please run 'make' first"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "================================================================"
echo "Canneal Benchmark - Testing All Sizes"
echo "================================================================"
echo ""

# Test configurations
declare -A configs
configs[mini]="1 1000 2000 input_data/mini/input/input_mini.nets 50"
configs[small]="1 50000 2500 input_data/small/input/input_small.nets 150"
configs[medium]="1 200000 3000 input_data/medium/input/input_medium.nets 400"
configs[large]="1 500000 3500 input_data/large/input/input_large.nets 700"
configs[extra-large]="1 1000000 4000 input_data/extra-large/input/input_extra-large.nets 1000"

# Run each configuration
for size in mini small medium large extra-large; do
    echo "================================================================"
    echo -e "${BLUE}Testing: $size${NC}"
    echo "================================================================"
    
    params=${configs[$size]}
    output_file="$OUTPUT_DIR/output_${size}.txt"
    
    echo "Command: $BINARY $params"
    echo "Output: $output_file"
    echo ""
    
    # Run with timing
    if /usr/bin/time -f "Time: %E (wall clock)" $BINARY $params > "$output_file" 2>&1; then
        echo -e "${GREEN}✓ Success${NC}"
        
        # Extract key metrics
        initial_cost=$(grep "Initial routing cost:" "$output_file" | awk '{print $NF}')
        final_cost=$(grep "Final cost:" "$output_file" | awk '{print $NF}')
        improvement=$(grep "Improvement:" "$output_file" | awk '{print $2}')
        
        echo "  Initial cost: $initial_cost"
        echo "  Final cost:   $final_cost"
        echo "  Improvement:  $improvement"
    else
        echo -e "${RED}✗ Failed${NC}"
    fi
    
    echo ""
done

echo "================================================================"
echo -e "${GREEN}All tests completed!${NC}"
echo "================================================================"
echo ""
echo "Output files saved in: $OUTPUT_DIR/"
echo ""
echo "Summary:"
echo "--------"
for size in mini small medium large extra-large; do
    output_file="$OUTPUT_DIR/output_${size}.txt"
    if [ -f "$output_file" ]; then
        final_cost=$(grep "Final cost:" "$output_file" | awk '{print $NF}')
        printf "%-15s Final cost: %s\n" "$size" "$final_cost"
    fi
done

