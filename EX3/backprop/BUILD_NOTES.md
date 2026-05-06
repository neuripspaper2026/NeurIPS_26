# Backprop Build Notes

## Time Measurement

Time measurement has been added to track both CUDA kernels:
1. `bpnn_layerforward_CUDA` - Forward propagation kernel
2. `bpnn_adjust_weights_cuda` - Weight adjustment kernel

Both kernel times are accumulated into `total_kernel_time` and output at the end.

### Modified Files
- `backprop_cuda.cu`: Added time measurement
  - Added `#include <time.h>`
  - Added global `total_kernel_time` variable
  - Added `main_start` timing at program start
  - Added kernel timing around both kernel calls
  - Added timing output at program end

## Multi-Variant Makefile

### File Structure

```
backprop/
├── backprop_cuda.cu              # Main CUDA file (includes kernel)
├── backprop.c                    # Helper functions
├── facetrain.c                   # Training interface
├── imagenet.c                    # Image loading
├── EX3_optimized_codes/          # Kernels & executables
│   ├── backprop_cuda_kernel.cu           # Baseline kernel
│   ├── backprop_cuda_kernel_<variant>.cu # Variant kernels
│   ├── backprop_cuda                     # Generated baseline executable
│   └── backprop_cuda_<variant>           # Generated variant executables
└── .build/                        # Temporary files (auto-generated)
```

### How It Works

Unlike b+tree which uses wrapper files, backprop directly includes the kernel file:
```c
#include "backprop_cuda_kernel.cu"
```

The Makefile handles this by:
1. Creating temporary copies of `backprop_cuda.cu` in `.build/`
2. Modifying the `#include` to point to the correct kernel variant:
   - Baseline: `#include "../EX3_optimized_codes/backprop_cuda_kernel.cu"`
   - Variant: `#include "../EX3_optimized_codes/backprop_cuda_kernel_<variant>.cu"`
3. Using `-I.` to add current directory to include path (for `backprop.h`)
4. Compiling the temporary file with all C source files

### Naming Convention

**Baseline kernel:**
```
EX3_optimized_codes/backprop_cuda_kernel.cu
→ generates: EX3_optimized_codes/backprop_cuda
```

**Variant kernels:**
```
EX3_optimized_codes/backprop_cuda_kernel_gpt4_v1.cu
→ generates: EX3_optimized_codes/backprop_cuda_gpt4_v1

EX3_optimized_codes/backprop_cuda_kernel_claude_v2.cu
→ generates: EX3_optimized_codes/backprop_cuda_claude_v2
```

### Build Commands

```bash
# Build all variants
make

# List detected variants
make list-variants

# Clean
make clean

# See help
make help
```

### Running Executables

```bash
# From root directory
EX3_optimized_codes/backprop_cuda 65536 [-o output.dat]

# Or from EX3_optimized_codes/
cd EX3_optimized_codes
./backprop_cuda 65536 [-o output.dat]
./backprop_cuda_gpt4_v1 65536 [-o output.dat]
```

### Key Differences from b+tree

| Aspect | b+tree | backprop |
|--------|--------|----------|
| Kernel inclusion | Wrapper files in `kernel/` | Direct `#include` in main file |
| Temporary files | Modified wrappers | Modified main CUDA file |
| Include path change | Wrapper includes kernel | Main includes kernel |
| Number of kernels | 2 separate kernel files | 1 kernel file with 2 kernels |

### Features

- ✅ No intermediate `.o` files
- ✅ Error tolerant (continues on failure)
- ✅ Automatic variant detection
- ✅ Executables in `EX3_optimized_codes/`
- ✅ Build summary at end
