#!/bin/bash
# Generate input.data and check.data pairs for 5 different test sizes
# Strategy: Use baseline program to generate golden outputs

set -e

echo "=== Compiling baseline program ==="
make clean
make all

# Create input_data directory structure
mkdir -p input_data/{mini,small,medium,large,extra-large}/{input,output}

echo ""
echo "=== Generating test datasets ==="

# Function to create input.data with specific plaintext bytes
generate_input() {
    local file=$1
    shift
    local bytes=("$@")
    
    # Section 1: Key (32 bytes: 0-31)
    echo "%%" > "$file"
    for i in {0..31}; do
        echo "$i" >> "$file"
    done
    
    # Section 2: Plaintext (16 bytes)
    echo "%%" >> "$file"
    for byte in "${bytes[@]}"; do
        echo "$byte" >> "$file"
    done
}

# Mini size - pattern 1
echo "Generating mini dataset..."
generate_input "input_data/mini/input/input.data" \
    0 17 34 51 68 85 102 119 136 153 170 187 204 221 238 255

# Small size - pattern 2 (reverse order)
echo "Generating small dataset..."
generate_input "input_data/small/input/input.data" \
    1 35 69 103 137 171 205 239 254 220 186 152 118 84 50 16

# Medium size - pattern 3 (same as original default)
echo "Generating medium dataset..."
generate_input "input_data/medium/input/input.data" \
    0 17 34 51 68 85 102 119 136 153 170 187 204 221 238 255

# Large size - pattern 4 (descending)
echo "Generating large dataset..."
generate_input "input_data/large/input/input.data" \
    255 238 221 204 187 170 153 136 119 102 85 68 51 34 17 0

# Extra-large size - pattern 5 (alternating)
echo "Generating extra-large dataset..."
generate_input "input_data/extra-large/input/input.data" \
    165 90 240 15 195 60 150 105 18 33 135 120 222 237 180 75

echo ""
echo "=== Generating golden check.data files ==="

# For each size, run baseline to generate golden output
for size in mini small medium large extra-large; do
    echo "Processing $size..."
    input_file="input_data/$size/input/input.data"
    check_file="input_data/$size/output/check.data"
    temp_check="input_data/$size/output/temp_check.data"
    
    # Run baseline with dummy check file (use input as dummy check, ignore error)
    # The actual output goes to EX5_correctness/aes_output.data
    ./EX1_optimized_codes/aes_gcc "$input_file" "$input_file" 2>/dev/null || true
    
    # Copy the output as the golden check.data
    if [ -f "EX5_correctness/aes_output.data" ]; then
        cp "EX5_correctness/aes_output.data" "$check_file"
        echo "  ✓ Created $check_file"
    else
        echo "  ✗ Failed to generate $check_file"
    fi
    
    # Create command reference
    echo "./EX1_optimized_codes/aes_gcc $input_file $check_file" > "input_data/$size/input/command.txt"
done

echo ""
echo "=== Verification: Testing each dataset ==="
for size in mini small medium large extra-large; do
    input_file="input_data/$size/input/input.data"
    check_file="input_data/$size/output/check.data"
    
    if [ -f "$check_file" ]; then
        echo -n "Testing $size... "
        if ./EX1_optimized_codes/aes_gcc "$input_file" "$check_file" 2>&1 | grep -q "Success"; then
            echo "✓ PASS"
        else
            # Run again to see output
            ./EX1_optimized_codes/aes_gcc "$input_file" "$check_file" 2>&1 || true
        fi
    fi
done

echo ""
echo "=== Dataset generation complete ==="
echo "Directory structure:"
ls -R input_data/

echo ""
echo "Usage examples:"
echo "  ./EX1_optimized_codes/aes_gcc input_data/mini/input/input.data input_data/mini/output/check.data"
echo "  ./EX1_optimized_codes/aes_gcc input_data/small/input/input.data input_data/small/output/check.data"
echo "  ./EX1_optimized_codes/aes_gcc input_data/medium/input/input.data input_data/medium/output/check.data"

