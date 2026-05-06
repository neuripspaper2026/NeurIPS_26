# Bilateral Filter Multi-Variant Build System

## Overview

Bilateral Filter has been modified to support:
1. **Time measurement** for kernel execution and total runtime
2. **Multi-variant compilation** for different kernel implementations

## Algorithm Background

### What is Bilateral Filter?

**Bilateral filter** is an edge-preserving nonlinear smoothing filter that:
- Preserves crisp edges while filtering noise
- Combines domain filter (spatial distance) and range filter (intensity difference)
- Uses weighted average based on both spatial and intensity similarity

**Key Parameters:**
- `gaussian_delta` (g_d): Controls spatial smoothing
- `euclidean_delta` (e_d): Controls intensity/range smoothing
- `filter_radius` (r): Size of filter window (2*r+1)
- `iterations`: Number of filtering passes

### Main Computational Kernel

```cuda
__global__ void d_bilateral_filter(uint *od, int w, int h, float e_d, int r,
                                   cudaTextureObject_t texObj)
```

**Location:** `EX3_optimized_codes/bilateral_kernel.cu` line 80

**What it does:**
1. For each pixel (x, y):
   - Read center pixel from texture
   - For all neighbors in radius r:
     - Calculate domain weight (gaussian distance)
     - Calculate range weight (intensity similarity)
     - Accumulate weighted pixel values
   - Write filtered result to output

**Execution:**
- Called in `bilateralFilterRGBA()` function
- Default: runs 150 cycles for benchmarking (`iCycles = 150`)
- Can run multiple iterations for cartoon effect

## Time Measurement

### Timing Strategy

```cpp
// Main timer
struct timespec main_start, main_end;
clock_gettime(CLOCK_MONOTONIC, &main_start);

// ... program execution ...

// Kernel timer (around benchmark loop)
struct timespec kernel_start, kernel_end;
clock_gettime(CLOCK_MONOTONIC, &kernel_start);

for (int i = 0; i < iCycles; i++) {
    bilateralFilterRGBA(...);  // Kernel launches inside
}
cudaDeviceSynchronize();

clock_gettime(CLOCK_MONOTONIC, &kernel_end);
total_kernel_time = elapsed_time;
```

**Key Points:**
- `KERNEL_TIME` = Total time for all iCycles (150 by default)
- `TOTAL_TIME` = Entire program (I/O + computation + initialization)
- Kernel timing includes all cycles, not per-cycle time

### Output Format

```
KERNEL_TIME: 0.123456789
TOTAL_TIME: 1.234567890
```

Written to:
- `stderr` by default
- File specified by `TIMING_LOG_FILE` environment variable

## Multi-Variant Makefile

### File Naming Convention

```
EX3_optimized_codes/
├── bilateral_kernel.cu              # Baseline kernel
├── bilateral_kernel_gpt4_v1.cu      # Variant 1
├── bilateral_kernel_claude_v2.cu    # Variant 2
├── bilateralFilter                  # Baseline executable
├── bilateralFilter_gpt4_v1          # Variant 1 executable
└── bilateralFilter_claude_v2        # Variant 2 executable
```

### Compilation Strategy

Unlike most benchmarks, bilateral has **multiple source files**:
- `bilateralFilter.cpp` (main program)
- `bmploader.cpp` (BMP image loading)
- `bilateral_kernel.cu` (CUDA kernel)

**Direct Compilation (No .o files):**
```makefile
# Baseline
$(OUTPUT_DIR)/bilateralFilter: bilateralFilter.cpp bmploader.cpp bilateral_kernel.cu
    nvcc -o $@ bilateralFilter.cpp bmploader.cpp bilateral_kernel.cu

# Variant
$(OUTPUT_DIR)/bilateralFilter_gpt4_v1: bilateralFilter.cpp bmploader.cpp bilateral_kernel_gpt4_v1.cu
    nvcc -o $@ bilateralFilter.cpp bmploader.cpp bilateral_kernel_gpt4_v1.cu
```

**Benefits:**
- No intermediate `.o` files
- Each variant is self-contained
- Simple build process
- Easy to manage

### How It Works

1. **Detection Phase:**
   ```makefile
   KERNEL_VARIANTS = $(wildcard $(OUTPUT_DIR)/bilateral_kernel_*.cu)
   VARIANT_IDS = gpt4_v1 claude_v2 ...
   BASELINE_EXISTS = yes/no
   ```

2. **Build Rules:**
   ```makefile
   # Each executable links all sources together
   $(OUTPUT_DIR)/bilateralFilter_%: bilateralFilter.cpp bmploader.cpp bilateral_kernel_%.cu
       nvcc -o $@ $^
   ```

3. **No Include Rewriting Needed:**
   - Unlike b+tree/bfs, no dynamic include path changes
   - Kernel file specified directly in compilation command
   - Simpler than template-based approaches

### Build Targets

```bash
make                # Build all variants
make clean          # Remove executables (safe - preserves .cu files)
make list-variants  # Show detected variants
make help           # Show help
```

### Build Output

```
==========================================
Building: EX3_optimized_codes/bilateralFilter (baseline)
==========================================
✓ Successfully built EX3_optimized_codes/bilateralFilter

==========================================
Building: EX3_optimized_codes/bilateralFilter_gpt4_v1 (variant gpt4_v1)
==========================================
✓ Successfully built EX3_optimized_codes/bilateralFilter_gpt4_v1

==========================================
Build Summary:
==========================================
✓ EX3_optimized_codes/bilateralFilter - SUCCESS
✓ EX3_optimized_codes/bilateralFilter_gpt4_v1 - SUCCESS
==========================================
```

## File Structure

```
EX3/bilateral/
├── bilateralFilter.cpp             # Main program (modified for timing)
├── bmploader.cpp                   # BMP loader (unchanged)
├── Makefile                        # Multi-variant build system
├── EX3_optimized_codes/
│   ├── bilateral_kernel.cu         # Baseline kernel
│   ├── bilateral_kernel_*.cu       # Variant kernels
│   ├── bilateralFilter             # Baseline executable
│   └── bilateralFilter_*           # Variant executables
└── input_data/
    ├── nature_monte.bmp            # Test image
    └── input_data.txt              # Dataset parameters
```

## Usage

### Command Line Format

```bash
./bilateralFilter <IMAGE> <euclidean_delta> <gaussian_delta> <filter_radius> <output_file>
```

### Parameters

- `IMAGE`: Input BMP image path
- `euclidean_delta` (e_d): Range filter parameter (e.g., 0.1)
- `gaussian_delta` (g_d): Domain filter parameter (e.g., 4)
- `filter_radius` (r): Filter window radius (e.g., 5)
- `output_file`: Output binary file path

### Example Execution

```bash
cd EX3/bilateral

# Baseline
TIMING_LOG_FILE=time.log \
./EX3_optimized_codes/bilateralFilter \
input_data/nature_monte.bmp 0.1 4 5 output.bin

cat time.log
# KERNEL_TIME: 0.234567890
# TOTAL_TIME: 1.456789012

# Variant
TIMING_LOG_FILE=time_variant.log \
./EX3_optimized_codes/bilateralFilter_gpt4_v1 \
input_data/nature_monte.bmp 0.1 4 5 output_variant.bin
```

## Important Notes

### Kernel Execution Cycles

The benchmark runs **150 cycles** by default (`iCycles = 150`):
```cpp
for (int i = 0; i < iCycles; i++) {
    bilateralFilterRGBA(...);
}
```

**KERNEL_TIME represents total time for all 150 cycles, not per-cycle time.**

### Texture Memory Usage

The kernel uses CUDA texture objects for efficient memory access:
```cuda
cudaTextureObject_t texObj = createTextureObject(...);
float4 pixel = tex2D<float4>(texObj, x, y);
```

Variants should maintain compatible texture access patterns.

### Multi-File Compilation

Unlike single-file CUDA programs:
- Main logic in `.cpp` file
- Kernel implementation in `.cu` file
- All compiled together in one `nvcc` command
- No separate compilation and linking phases

### Clean Target Safety

The clean rule is safe and only removes executables:

```makefile
cd $(OUTPUT_DIR) && rm -f bilateralFilter \
    $(shell ls bilateralFilter_* | grep -v '\.cu$$' | grep -v '\.h$$')
```

Preserves:
- ✅ `bilateral_kernel.cu` (source)
- ✅ `bilateral_kernel_gpt4_v1.cu` (source)

Removes:
- ✅ `bilateralFilter` (executable)
- ✅ `bilateralFilter_gpt4_v1` (executable)

## Verification

### Build Test

```bash
cd EX3/bilateral
make clean
make
ls -la EX3_optimized_codes/
```

Expected output:
```
-rwxr-xr-x bilateralFilter
-rwxr-xr-x bilateralFilter_gpt4_v1
-rw-r--r-- bilateral_kernel.cu
-rw-r--r-- bilateral_kernel_gpt4_v1.cu
```

### Timing Test

```bash
TIMING_LOG_FILE=time.log \
./EX3_optimized_codes/bilateralFilter \
input_data/nature_monte.bmp 0.1 4 5 output.bin

cat time.log
# Should show:
# KERNEL_TIME: <value>
# TOTAL_TIME: <value>
```

### Output Verification

```bash
ls -lh output.bin
# Should be width*height*4 bytes (RGBA format)

# For reference image (nature_monte.bmp is typically 1024x768):
# Expected size: 1024 * 768 * 4 = 3,145,728 bytes
```

## Algorithm Details

### Bilateral Filter Formula

For each pixel at (x, y):

```
filtered(x,y) = Σ w(i,j) * pixel(x+i, y+j) / Σ w(i,j)

where:
  w(i,j) = gaussian_domain(i,j) * gaussian_range(center, neighbor)
  
  gaussian_domain(i,j) = exp(-(i² + j²) / (2 * gaussian_delta²))
  gaussian_range(c, n) = exp(-||c - n||² / (2 * euclidean_delta²))
```

### Kernel Implementation

```cuda
__global__ void d_bilateral_filter(...) {
    // For each pixel
    float4 center = tex2D<float4>(texObj, x, y);
    float sum = 0, factor;
    float4 result = {0, 0, 0, 0};
    
    // For each neighbor in window
    for (int i = -r; i <= r; i++) {
        for (int j = -r; j <= r; j++) {
            float4 neighbor = tex2D<float4>(texObj, x+j, y+i);
            
            // Domain weight * Range weight
            factor = cGaussian[i+r] * cGaussian[j+r] *
                     euclideanLen(neighbor, center, e_d);
            
            result += factor * neighbor;
            sum += factor;
        }
    }
    
    output[y*w + x] = rgbaFloatToInt(result / sum);
}
```

## Comparison with Other Benchmarks

| Benchmark | Source Files | Kernel Count | Build Strategy |
|-----------|-------------|--------------|----------------|
| b+tree | 1 .c + kernels | 2 | Include rewriting |
| bfs | 1 .cu + kernels | 2 | Include rewriting |
| backprop | 1 .cu + kernel | 2 | Include rewriting |
| **bilateral** | **2 .cpp + 1 .cu** | **1** | **Direct compilation** |

**Bilateral is unique:**
- Mixed C++ and CUDA files
- BMP loader in separate file
- Direct multi-file compilation
- No include path manipulation needed

## Performance Considerations

### Expected Timing Ranges

For a 1024x768 image, 150 cycles:
- **Fast GPU:** 0.5-1.0 seconds
- **Medium GPU:** 1.0-2.0 seconds
- **Slow GPU:** 2.0-5.0 seconds

### Optimization Opportunities

1. **Texture Memory:** Already optimized with texture objects
2. **Constant Memory:** Gaussian array in `__constant__` memory
3. **Thread Block Size:** Currently 16x16, can experiment
4. **Filter Radius:** Smaller radius = faster, but less smoothing
5. **Loop Unrolling:** Unroll inner loops for known small radii

## Reference

Based on NVIDIA CUDA Samples bilateral filter implementation.

**Academic Reference:**
- **Paper:** "Bilateral Filtering for Gray and Color Images"
- **Author:** Carlo Tomasi, Roberto Manduchi
- **Conference:** ICCV 1998
- **Description:** Original bilateral filter algorithm

## Summary

**Key Takeaways:**

1. Bilateral filter is an edge-preserving smoothing filter
2. Main kernel: `d_bilateral_filter` (single kernel, not multiple like BFS)
3. Runs 150 benchmark cycles by default
4. Multi-file compilation (C++ + CUDA)
5. Direct compilation strategy (no .o files, no include rewriting)
6. Timing measures total of all cycles

**For Makefile/Build:**
- Detect `bilateral_kernel_*.cu` variants
- Compile all sources together: `nvcc -o exe main.cpp loader.cpp kernel.cu`
- Safe clean rule to preserve source files
- Simple and straightforward build process
