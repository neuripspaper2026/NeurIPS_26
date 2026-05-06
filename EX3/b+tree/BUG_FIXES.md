# Bug Fixes Summary

## 🚨 CRITICAL BUG: Clean Target Deletes Source Code

**Date:** 2026-01-26
**Severity:** CRITICAL
**Status:** FIXED

### Problem

The original clean rule used a dangerous wildcard:

```makefile
clean:
	$(RM) $(OUTPUT_DIR)/b+tree $(OUTPUT_DIR)/b+tree_*
```

This would delete:
- `b+tree` - executable ✓
- `b+tree_gpt4_v1` - executable ✓
- `kernel_gpu_cuda.cu` - source code ❌ (if matched by pattern)
- `kernel_gpu_cuda_gpt4_v1.cu` - source code ❌ (matched!)

### Solution

Changed to filter-based deletion:

```makefile
clean:
	@cd $(OUTPUT_DIR) 2>/dev/null && rm -f b+tree $$(ls b+tree_* 2>/dev/null | grep -v '\.cu$$' | grep -v '\.h$$' | grep -v '\.c$$' | grep -v '\.cpp$$') || true
```

Now only deletes executables (no file extension), preserving all source files.

### Verification

```bash
# Before fix - DANGEROUS:
ls EX3_optimized_codes/
# kernel_gpu_cuda.cu, b+tree, b+tree_gpt4_v1

make clean
ls EX3_optimized_codes/
# (empty - everything deleted!)

# After fix - SAFE:
ls EX3_optimized_codes/
# kernel_gpu_cuda.cu, b+tree, b+tree_gpt4_v1

make clean
ls EX3_optimized_codes/
# kernel_gpu_cuda.cu (source code preserved!)
```

### Impact

All Makefiles following the template have been updated:
- ✅ EX3/b+tree/Makefile
- ✅ EX3/backprop/Makefile
- ✅ .cursor/commands/add-time-and-makefile-for-ex3.md (template)

### Lesson Learned

**NEVER use simple wildcards like `$(RM) dir/prefix_*` when source files and executables share the same prefix!**

Always use filtered deletion with grep to exclude source file extensions.
