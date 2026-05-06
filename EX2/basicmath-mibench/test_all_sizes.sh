#!/bin/bash

# Script to test all sizes
# Usage: ./test_all_sizes.sh [output_dir]

OUTPUT_DIR=${1:-"test_results"}
BINARY="./EX1_optimized_codes/basicmath_gcc"

# Check if binary exists
if [ ! -f "$BINARY" ]; then
    echo "Error: $BINARY not found!"
    echo "Please run 'make' first."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# List of sizes to test
SIZES="mini small medium large"

echo "=================================================="
echo "BasicMath Benchmark - Testing All Sizes"
echo "=================================================="
echo "Binary: $BINARY"
echo "Output directory: $OUTPUT_DIR"
echo "Sizes to test: $SIZES"
echo ""

# 测试每个 size
for size in $SIZES; do
    echo "--------------------------------------------------"
    echo "Testing size: $size"
    echo "--------------------------------------------------"
    
    output_file="$OUTPUT_DIR/output_${size}.txt"
    time_file="$OUTPUT_DIR/time_${size}.txt"
    
    # Run and record time
    { time $BINARY $size > "$output_file" 2>&1 ; } 2> "$time_file"
    
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✓ Success"
        
        # Extract time information
        real_time=$(grep "^real" "$time_file" | awk '{print $2}')
        echo "  Time: $real_time"
        
        # Count output lines
        line_count=$(wc -l < "$output_file")
        echo "  Output lines: $line_count"
        
        # File size
        file_size=$(du -h "$output_file" | awk '{print $1}')
        echo "  Output size: $file_size"
    else
        echo "✗ Failed (exit code: $exit_code)"
    fi
    echo ""
done

# Generate summary report
echo "=================================================="
echo "Summary Report"
echo "=================================================="
echo ""
printf "%-15s %-12s %-12s %-12s\n" "Size" "Time" "Lines" "File Size"
printf "%-15s %-12s %-12s %-12s\n" "---------------" "------------" "------------" "------------"

for size in $SIZES; do
    output_file="$OUTPUT_DIR/output_${size}.txt"
    time_file="$OUTPUT_DIR/time_${size}.txt"
    
    if [ -f "$time_file" ]; then
        real_time=$(grep "^real" "$time_file" | awk '{print $2}')
    else
        real_time="N/A"
    fi
    
    if [ -f "$output_file" ]; then
        line_count=$(wc -l < "$output_file")
        file_size=$(du -h "$output_file" | awk '{print $1}')
    else
        line_count="N/A"
        file_size="N/A"
    fi
    
    printf "%-15s %-12s %-12s %-12s\n" "$size" "$real_time" "$line_count" "$file_size"
done

echo ""
echo "=================================================="
echo "All tests completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=================================================="

