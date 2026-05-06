# Bug Fix: Include Path Issue

## Problem

When building, the compiler couldn't find `backprop.h`:

```
fatal error: backprop.h: No such file or directory
 #include "backprop.h"
          ^~~~~~~~~~~~
```

## Root Cause

The temporary CUDA file was generated in `.build/` directory:
```
.build/backprop_cuda_baseline.cu
```

This file includes:
1. `backprop.h` (in parent directory)
2. `../EX3_optimized_codes/backprop_cuda_kernel.cu` (which also includes `backprop.h`)

When compiling from `.build/`, the relative path `"backprop.h"` couldn't resolve correctly.

## Solution

Added `-I.` to the compilation command to include the current directory (backprop/) in the include search path:

```makefile
-@$(NVCC) $(CPPFLAGS) -I. $(NVCCFLAGS) $(NVCC_LDLIBS) -lm -o $@ $^
```

This allows the compiler to find `backprop.h` in the backprop/ directory regardless of where the source file is located.

## What Changed

### Before:
```makefile
-@$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(NVCC_LDLIBS) -lm -o $@ $^
```

### After:
```makefile
-@$(NVCC) $(CPPFLAGS) -I. $(NVCCFLAGS) $(NVCC_LDLIBS) -lm -o $@ $^
```

### Applied to:
- Baseline build rule: `$(OUTPUT_DIR)/backprop_cuda`
- Variant build rule: `$(OUTPUT_DIR)/backprop_cuda_%`

## Testing

After this fix, the build should succeed:

```bash
make clean
make
# Should now successfully build all variants
```

## Technical Details

The `-I.` flag adds the current working directory (where make is run, i.e., `EX3/backprop/`) to the include search paths. This means:

- `#include "backprop.h"` will find `./backprop.h`
- `#include "../EX3_optimized_codes/backprop_cuda_kernel.cu"` will find the kernel file
- Inside the kernel file, `#include "backprop.h"` will also resolve correctly

This is cleaner than modifying all include paths in the temporary files.
