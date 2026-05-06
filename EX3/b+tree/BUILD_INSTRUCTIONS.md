# B+Tree Multi-Variant Build Instructions

## Quick Start

```bash
# Clean previous builds
make clean

# Build all variants
make

# List detected variants
make list-variants
```

## File Structure

```
b+tree/
├── b+tree.c                          # Main source file
├── kernel/                            # Original kernel wrappers (headers only)
│   ├── kernel_gpu_cuda_wrapper.cu
│   ├── kernel_gpu_cuda_wrapper.h
│   ├── kernel_gpu_cuda_wrapper_2.cu
│   └── kernel_gpu_cuda_wrapper_2.h
├── EX3_optimized_codes/              # Kernel implementations & executables
│   ├── kernel_gpu_cuda.cu            # Baseline kernel 1
│   ├── kernel_gpu_cuda_2.cu          # Baseline kernel 2
│   ├── kernel_gpu_cuda_<model>_<trail>.cu    # Variant kernel 1
│   ├── kernel_gpu_cuda_2_<model>_<trail>.cu  # Variant kernel 2
│   ├── b+tree                         # Generated baseline executable
│   └── b+tree_<model>_<trail>        # Generated variant executables
├── .build/                            # Temporary wrappers (auto-generated)
└── util/                              # Utility functions
```

## Naming Convention

### Baseline Version
Place these files in `EX3_optimized_codes/`:
- `kernel_gpu_cuda.cu`
- `kernel_gpu_cuda_2.cu`

This will generate: `EX3_optimized_codes/b+tree`

### Variant Versions
For each variant, place BOTH files in `EX3_optimized_codes/`:
- `kernel_gpu_cuda_<model>_<trail>.cu`
- `kernel_gpu_cuda_2_<model>_<trail>.cu`

Examples (all executables placed in `EX3_optimized_codes/`):
```
kernel_gpu_cuda_gpt4_v1.cu + kernel_gpu_cuda_2_gpt4_v1.cu → EX3_optimized_codes/b+tree_gpt4_v1
kernel_gpu_cuda_claude_v2.cu + kernel_gpu_cuda_2_claude_v2.cu → EX3_optimized_codes/b+tree_claude_v2
kernel_gpu_cuda_llama_trial3.cu + kernel_gpu_cuda_2_llama_trial3.cu → EX3_optimized_codes/b+tree_llama_trial3
```

## Build Process

The Makefile automatically:

1. **Scans** `EX3_optimized_codes/` for all kernel variants
2. **Matches** pairs of `kernel_gpu_cuda_*` and `kernel_gpu_cuda_2_*` files
3. **Generates** temporary wrappers in `.build/` with corrected include paths
4. **Compiles** each variant directly to executable (no .o files)
5. **Continues** building even if some variants fail

## Error Handling

If a variant fails to compile:
- ✗ The error is shown
- ✓ Build continues with other variants
- ✓ Final summary shows which succeeded/failed

Example output:
```
==========================================
Building: EX3_optimized_codes/b+tree_gpt4_v1 (variant: gpt4_v1)
==========================================
✓ Successfully built EX3_optimized_codes/b+tree_gpt4_v1

==========================================
Building: EX3_optimized_codes/b+tree_claude_v3 (variant: claude_v3)
==========================================
error: some compilation error...
✗ Failed to build EX3_optimized_codes/b+tree_claude_v3 (continuing...)

==========================================
Build Summary:
==========================================
✓ EX3_optimized_codes/b+tree - SUCCESS
✓ EX3_optimized_codes/b+tree_gpt4_v1 - SUCCESS
✗ EX3_optimized_codes/b+tree_claude_v3 - FAILED
==========================================
```

## Troubleshooting

### Issue: "No such file or directory" errors

**Check 1:** Verify both kernel files exist for each variant
```bash
ls -la EX3_optimized_codes/kernel_gpu_cuda*.cu
```

**Check 2:** Ensure file naming matches exactly
- Both files must have the SAME suffix after `kernel_gpu_cuda_`
- File names are case-sensitive

**Check 3:** Run debug script
```bash
bash debug_build.sh
```

### Issue: All builds fail

**Check 1:** Verify CUDA is available
```bash
nvcc --version
```

**Check 2:** Check include paths
```bash
make list-variants
```

**Check 3:** Manually inspect generated wrapper
```bash
cat .build/kernel_gpu_cuda_wrapper_baseline.cu | grep "#include"
```

Should show:
```c
#include "../common.h"
#include "../util/cuda/cuda.h"
#include "../util/timer/timer.h"
#include "../EX3_optimized_codes/kernel_gpu_cuda.cu"
#include "../kernel/kernel_gpu_cuda_wrapper.h"
```

### Issue: Specific variant fails

Check the actual kernel file for syntax errors:
```bash
nvcc -c EX3_optimized_codes/kernel_gpu_cuda_<variant>.cu
```

## Makefile Targets

```bash
make                # Build all variants
make clean          # Remove executables and .build/
make distclean      # Deep clean (same as clean)
make list-variants  # Show detected variants
make help           # Show help message
```

## Advanced Usage

### Build only baseline
```bash
make EX3_optimized_codes/b+tree
```

### Build specific variant
```bash
make EX3_optimized_codes/b+tree_gpt4_v1
```

### Run an executable
```bash
# Run baseline
cd EX3_optimized_codes
./b+tree file ../input/data.txt command ../command.txt

# Run specific variant
cd EX3_optimized_codes
./b+tree_gpt4_v1 file ../input/data.txt command ../command.txt
```

### Force rebuild
```bash
make clean && make
```

### Suppress warnings during build
The Makefile already filters out warnings, but you can further customize:
```bash
make 2>&1 | grep -v "warning"
```

## Notes

- **No .o files**: Everything compiles directly to executable
- **Temporary files**: `.build/` directory contains auto-generated wrappers
- **Path corrections**: Wrappers automatically adjust include paths
- **Parallel builds**: Use `make -j4` for faster compilation (careful with error messages)
