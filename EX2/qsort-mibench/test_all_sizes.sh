#!/bin/bash
# Test script for qsort-mibench with different input sizes

BINARY="./EX1_optimized_codes/qsort_large_gcc"
INPUT_DIR="input_data"

if [ ! -f "$BINARY" ]; then
    echo "Error: Binary not found: $BINARY"
    echo "Run 'make' first to compile the benchmark"
    exit 1
fi

echo "════════════════════════════════════════════════════════════════"
echo "qsort-mibench Test - All Sizes"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Note: Testing with current MAXARRAY setting"
echo ""

# Test all available sizes
for size in mini small medium large extra-large; do
    input_file="$INPUT_DIR/$size/input/input_$size.dat"
    
    if [ ! -f "$input_file" ]; then
        echo "⚠️  Skipping $size: file not found"
        continue
    fi
    
    echo "════════════════════════════════════════════════════════════════"
    echo "Testing: $size"
    echo "Input: $input_file"
    echo "Size: $(wc -l < "$input_file") vertices, $(du -h "$input_file" | cut -f1)"
    echo "────────────────────────────────────────────────────────────────"
    
    echo -n "Running... "
    start_time=$(date +%s.%N)
    
    output=$("$BINARY" "$input_file" 2>&1)
    exit_code=$?
    
    end_time=$(date +%s.%N)
    elapsed=$(echo "$end_time - $start_time" | bc)
    
    if [ $exit_code -eq 0 ] || [ $exit_code -eq 141 ]; then
        output_lines=$(echo "$output" | wc -l)
        expected_vertices=$(wc -l < "$input_file")
        echo "✓ Success"
        echo "  Input: $expected_vertices vertices"
        echo "  Output: $output_lines lines"
        echo "  Time: ${elapsed}s"
        if [ "$output_lines" -eq "$expected_vertices" ]; then
            echo "  ✓ Output count matches input"
        else
            echo "  ⚠ Warning: Output count mismatch"
        fi
    else
        echo "✗ Failed (exit code: $exit_code)"
    fi
    echo ""
done

echo "════════════════════════════════════════════════════════════════"
echo "Performance Summary"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "If you see failures for larger sizes, you need to:"
echo "  1. Edit qsort_large.c (or your variant in EX1_optimized_codes/)"
echo "  2. Change: #define MAXARRAY 600000  (line 6)"
echo "  3. Use malloc() instead of stack allocation (see current version)"
echo "  4. Run: make clean && make"
echo ""
echo "Current implementation uses dynamic memory allocation (malloc)"
echo "to support up to 600,000 vertices without stack overflow."
echo "════════════════════════════════════════════════════════════════"

