#!/bin/bash
# Test script for susan-s-mibench
# Tests all input sizes in smoothing mode

BINARY="./EX1_optimized_codes/susan_gcc"
OUT_DIR="test_outputs"

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "════════════════════════════════════════════════════════════════"
echo "SUSAN Benchmark - Testing All Sizes"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Check binary exists
if [ ! -f "$BINARY" ]; then
    echo "Error: Binary not found: $BINARY"
    echo "Please run 'make' first"
    exit 1
fi

# Create output directory
mkdir -p "$OUT_DIR"

# Test each size
for size in mini small medium large extra-large; do
    input_file="input_data/$size/input/input_$size.pgm"
    output_file="$OUT_DIR/output_$size.pgm"
    
    if [ ! -f "$input_file" ]; then
        echo -e "${YELLOW}[SKIP]${NC} $size - input file not found: $input_file"
        continue
    fi
    
    echo -e "${BLUE}[TEST]${NC} $size"
    echo "  Input:  $input_file"
    echo "  Output: $output_file"
    
    # Get input file info
    input_size=$(stat -c%s "$input_file" 2>/dev/null || stat -f%z "$input_file" 2>/dev/null)
    input_dims=$(file "$input_file" | grep -oP 'size = \K[0-9]+ x [0-9]+' || echo "unknown")
    echo "  Dimensions: $input_dims"
    echo "  Input size: $((input_size / 1024)) KB"
    
    # Run benchmark with time measurement
    echo "  Running smoothing mode..."
    /usr/bin/time -f "  Time: %E (real), %U (user), %S (sys)" \
        $BINARY "$input_file" "$output_file" -s 2>&1 | grep -E "(Time:|error|Error)"
    
    # Check if output was created
    if [ -f "$output_file" ]; then
        output_size=$(stat -c%s "$output_file" 2>/dev/null || stat -f%z "$output_file" 2>/dev/null)
        echo "  Output size: $((output_size / 1024)) KB"
        echo -e "  Status: ${GREEN}✓ Success${NC}"
    else
        echo -e "  Status: ${YELLOW}✗ No output file${NC}"
    fi
    
    echo ""
done

echo "════════════════════════════════════════════════════════════════"
echo "Summary"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Output files:"
ls -lh "$OUT_DIR"/*.pgm 2>/dev/null | awk '{print "  " $9 " - " $5}'

echo ""
echo "To view outputs (requires ImageMagick):"
echo "  display $OUT_DIR/output_mini.pgm"
echo "  # or convert to PNG:"
echo "  convert $OUT_DIR/output_mini.pgm output_mini.png"

echo ""
echo "To test other modes:"
echo "  # Edge detection"
echo "  $BINARY input_data/large/input/input_large.pgm output_edges.pgm -e"
echo ""
echo "  # Corner detection"
echo "  $BINARY input_data/large/input/input_large.pgm output_corners.pgm -c"

