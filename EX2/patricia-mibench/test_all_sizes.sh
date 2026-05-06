#!/bin/bash
# Test patricia-mibench across all dataset sizes

BINARY="./EX1_optimized_codes/patricia_gcc"
SIZES=("mini" "small" "medium" "large" "extra-large")

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "════════════════════════════════════════════════════════════════"
echo "  Patricia-mibench - All Sizes Test"
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
    input_file="input_data/$size/input/input.udp"
    
    if [ ! -f "$input_file" ]; then
        echo -e "${YELLOW}⚠ Skipping $size: input file not found${NC}"
        continue
    fi
    
    echo -e "${BLUE}═══ Testing $size ═══${NC}"
    echo "Input: $input_file"
    
    # Get file size and packet count
    file_size=$(du -h "$input_file" | cut -f1)
    packet_count=$(wc -l < "$input_file")
    echo "File size: $file_size"
    echo "Packets: $packet_count"
    
    # Run benchmark and capture timing
    echo "Running..."
    start_time=$(date +%s.%N)
    output=$($BINARY "$input_file" 2>&1)
    end_time=$(date +%s.%N)
    exit_code=$?
    
    # Calculate runtime
    runtime=$(echo "$end_time - $start_time" | bc)
    
    # Count Found/Not Found
    found_count=$(echo "$output" | grep -c "Found")
    notfound_count=$(echo "$output" | grep -c "Not Found")
    
    echo -e "${GREEN}Runtime: ${runtime}s${NC}"
    echo "Results: $found_count Found, $notfound_count Not Found"
    
    # Calculate throughput
    if [ $(echo "$runtime > 0" | bc) -eq 1 ]; then
        throughput=$(echo "scale=2; $packet_count / $runtime" | bc)
        echo "Throughput: ${throughput} packets/sec"
    fi
    
    # Show first 3 results as sample
    echo "Sample output:"
    echo "$output" | head -3 | sed 's/^/  /'
    
    if [ $exit_code -ne 0 ]; then
        echo -e "${YELLOW}⚠ Warning: Binary exited with code $exit_code${NC}"
    fi
    
    echo ""
done

echo "════════════════════════════════════════════════════════════════"
echo -e "${GREEN}✓ All tests completed${NC}"
echo "════════════════════════════════════════════════════════════════"

