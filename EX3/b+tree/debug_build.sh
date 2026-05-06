#!/bin/bash
# Debug script to verify the Makefile path replacements

cd "$(dirname "$0")"

echo "=========================================="
echo "Debug: Verifying path replacements"
echo "=========================================="
echo ""

# Clean first
make clean 2>/dev/null

# Create build and output directories if they don't exist
mkdir -p .build EX3_optimized_codes

# Manually generate the baseline wrapper to inspect it
echo "Generating .build/kernel_gpu_cuda_wrapper_baseline.cu..."
sed -e 's|#include "./kernel_gpu_cuda\.cu"|#include "../EX3_optimized_codes/kernel_gpu_cuda.cu"|g' \
    -e 's|#include "./kernel_gpu_cuda_wrapper\.h"|#include "../kernel/kernel_gpu_cuda_wrapper.h"|g' \
    kernel/kernel_gpu_cuda_wrapper.cu > .build/kernel_gpu_cuda_wrapper_baseline.cu

echo ""
echo "=========================================="
echo "Content of generated wrapper:"
echo "=========================================="
grep "#include" .build/kernel_gpu_cuda_wrapper_baseline.cu

echo ""
echo "=========================================="
echo "Checking kernel files in EX3_optimized_codes:"
echo "=========================================="
ls -la EX3_optimized_codes/kernel_gpu_cuda*.cu 2>/dev/null || echo "No kernel files found!"

echo ""
echo "=========================================="
echo "Output directory: EX3_optimized_codes/"
echo "Executables will be placed in this directory"
echo ""
echo "Now try: make"
echo "=========================================="
