#!/bin/bash
# Test dijkstra-mibench across all dataset sizes

BINARY="./EX1_optimized_codes/dijkstra_gcc"
SIZES=("mini" "small" "medium" "large" "extra-large")

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "════════════════════════════════════════════════════════════════"
echo "  Dijkstra-mibench - All Sizes Test"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Check if binary exists
if [ ! -f "$BINARY" ]; then
    echo "Error: Binary $BINARY not found. Please run 'make' first."
    exit 1
fi

echo "Binary: $BINARY"
echo ""

# Test each size
for size in "${SIZES[@]}"; do
    input_file="input_data/$size/input/input.dat"
    
    if [ ! -f "$input_file" ]; then
        echo -e "${YELLOW}⚠ Skipping $size: input file not found${NC}"
        continue
    fi
    
    echo -e "${BLUE}═══ Testing $size ═══${NC}"
    echo "Input: $input_file"
    
    # Get file size
    file_size=$(du -h "$input_file" | cut -f1)
    echo "File size: $file_size"
    
    # Run benchmark and capture timing
    echo "Running..."
    start_time=$(date +%s.%N)
    output=$($BINARY "$input_file" 2>&1)
    end_time=$(date +%s.%N)
    
    # Calculate runtime
    runtime=$(echo "$end_time - $start_time" | bc)
    
    # Extract first line to check matrix size
    first_line=$(echo "$output" | head -1)
    
    # Count number of paths computed
    path_count=$(echo "$output" | grep -c "Shortest path is")
    
    echo "$first_line"
    echo "Paths computed: $path_count"
    echo -e "${GREEN}Runtime: ${runtime}s${NC}"
    
    # Show first 3 paths as sample
    echo "Sample paths:"
    echo "$output" | grep "Shortest path is" | head -3 | sed 's/^/  /'
    
    echo ""
done

echo "════════════════════════════════════════════════════════════════"
echo -e "${GREEN}✓ All tests completed${NC}"
echo "════════════════════════════════════════════════════════════════"

