#!/bin/bash
# Test script for the new Makefile

echo "=========================================="
echo "Testing b+tree Makefile"
echo "=========================================="
echo ""

cd "$(dirname "$0")"

echo "1. Listing detected variants..."
echo "------------------------------------------"
make list-variants
echo ""

echo "2. Checking if files are properly detected..."
echo "------------------------------------------"
ls -la EX3_optimized_codes/kernel_gpu_cuda*.cu 2>/dev/null || echo "No kernel files found in EX3_optimized_codes/"
echo ""

echo "3. Testing help command..."
echo "------------------------------------------"
make help
echo ""

echo "=========================================="
echo "To build all variants, run: make"
echo "To clean, run: make clean"
echo "=========================================="
