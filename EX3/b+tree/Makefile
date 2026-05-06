include ../common.mk

# Source files (excluding kernel wrappers)
MAIN_SRC = b+tree.c
UTIL_SRCS = util/timer/timer.c util/num/num.c util/cuda/cuda.cu

# Kernel wrapper directory
WRAPPER_DIR = kernel

# Find all variant kernel files in EX3_optimized_codes
KERNEL_1_VARIANTS := $(wildcard EX3_optimized_codes/kernel_gpu_cuda_*.cu)
KERNEL_2_VARIANTS := $(wildcard EX3_optimized_codes/kernel_gpu_cuda_2_*.cu)

# Extract variant suffixes (e.g., kernel_gpu_cuda_gpt4_v1.cu -> gpt4_v1)
VARIANT_SUFFIXES_1 := $(patsubst EX3_optimized_codes/kernel_gpu_cuda_%.cu,%,$(KERNEL_1_VARIANTS))
VARIANT_SUFFIXES_2 := $(patsubst EX3_optimized_codes/kernel_gpu_cuda_2_%.cu,%,$(KERNEL_2_VARIANTS))

# Find matching pairs (both kernel_1 and kernel_2 must exist)
VALID_VARIANTS := $(filter $(VARIANT_SUFFIXES_1),$(VARIANT_SUFFIXES_2))

# Check for baseline files (kernel_gpu_cuda.cu without suffix)
BASELINE_EXISTS := $(shell [ -f EX3_optimized_codes/kernel_gpu_cuda.cu ] && [ -f EX3_optimized_codes/kernel_gpu_cuda_2.cu ] && echo yes)

# Output directory for executables
OUTPUT_DIR = EX3_optimized_codes

# Generate executable names (with output directory prefix)
ifeq ($(BASELINE_EXISTS),yes)
    EXECUTABLES := $(OUTPUT_DIR)/b+tree
endif
EXECUTABLES += $(addprefix $(OUTPUT_DIR)/b+tree_,$(VALID_VARIANTS))

# Build directory for temporary wrappers
BUILD_DIR = .build
$(shell mkdir -p $(BUILD_DIR) $(OUTPUT_DIR))

# Default target
.PHONY: all
all: $(EXECUTABLES)
	@echo ""
	@echo "=========================================="
	@echo "Build Summary:"
	@echo "=========================================="
	@for exe in $(EXECUTABLES); do \
		if [ -f $$exe ]; then \
			echo "✓ $$exe - SUCCESS"; \
		else \
			echo "✗ $$exe - FAILED"; \
		fi; \
	done
	@echo "=========================================="

# Generate temporary wrapper for baseline
$(BUILD_DIR)/kernel_gpu_cuda_wrapper_baseline.cu: $(WRAPPER_DIR)/kernel_gpu_cuda_wrapper.cu
	@mkdir -p $(BUILD_DIR)
	@sed -e 's|#include "./kernel_gpu_cuda\.cu"|#include "../EX3_optimized_codes/kernel_gpu_cuda.cu"|g' \
	     -e 's|#include "./kernel_gpu_cuda_wrapper\.h"|#include "../kernel/kernel_gpu_cuda_wrapper.h"|g' $< > $@

$(BUILD_DIR)/kernel_gpu_cuda_wrapper_2_baseline.cu: $(WRAPPER_DIR)/kernel_gpu_cuda_wrapper_2.cu
	@mkdir -p $(BUILD_DIR)
	@sed -e 's|#include "./kernel_gpu_cuda_2\.cu"|#include "../EX3_optimized_codes/kernel_gpu_cuda_2.cu"|g' \
	     -e 's|#include "./kernel_gpu_cuda_wrapper_2\.h"|#include "../kernel/kernel_gpu_cuda_wrapper_2.h"|g' $< > $@

# Generate temporary wrappers for each variant
$(BUILD_DIR)/kernel_gpu_cuda_wrapper_%.cu: $(WRAPPER_DIR)/kernel_gpu_cuda_wrapper.cu
	@mkdir -p $(BUILD_DIR)
	@sed -e 's|#include "./kernel_gpu_cuda\.cu"|#include "../EX3_optimized_codes/kernel_gpu_cuda_$*.cu"|g' \
	     -e 's|#include "./kernel_gpu_cuda_wrapper\.h"|#include "../kernel/kernel_gpu_cuda_wrapper.h"|g' $< > $@

$(BUILD_DIR)/kernel_gpu_cuda_wrapper_2_%.cu: $(WRAPPER_DIR)/kernel_gpu_cuda_wrapper_2.cu
	@mkdir -p $(BUILD_DIR)
	@sed -e 's|#include "./kernel_gpu_cuda_2\.cu"|#include "../EX3_optimized_codes/kernel_gpu_cuda_2_$*.cu"|g' \
	     -e 's|#include "./kernel_gpu_cuda_wrapper_2\.h"|#include "../kernel/kernel_gpu_cuda_wrapper_2.h"|g' $< > $@

# Build baseline executable
$(OUTPUT_DIR)/b+tree: $(MAIN_SRC) $(UTIL_SRCS) $(BUILD_DIR)/kernel_gpu_cuda_wrapper_baseline.cu $(BUILD_DIR)/kernel_gpu_cuda_wrapper_2_baseline.cu
	@echo ""
	@echo "=========================================="
	@echo "Building: $@ (baseline)"
	@echo "=========================================="
	@mkdir -p $(OUTPUT_DIR)
	-@$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(NVCC_LDLIBS) -lm -lstdc++ -o $@ $^ 2>&1 | grep -v "warning:" || true
	@if [ -f $@ ]; then \
		echo "✓ Successfully built $@"; \
	else \
		echo "✗ Failed to build $@ (continuing...)"; \
	fi

# Build variant executables
$(OUTPUT_DIR)/b+tree_%: $(MAIN_SRC) $(UTIL_SRCS) $(BUILD_DIR)/kernel_gpu_cuda_wrapper_%.cu $(BUILD_DIR)/kernel_gpu_cuda_wrapper_2_%.cu
	@echo ""
	@echo "=========================================="
	@echo "Building: $@ (variant: $*)"
	@echo "=========================================="
	@mkdir -p $(OUTPUT_DIR)
	-@$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(NVCC_LDLIBS) -lm -lstdc++ -o $@ $^ 2>&1 | grep -v "warning:" || true
	@if [ -f $@ ]; then \
		echo "✓ Successfully built $@"; \
	else \
		echo "✗ Failed to build $@ (continuing...)"; \
	fi

# Clean target
.PHONY: clean
clean:
	@echo "Cleaning up..."
	@# Only remove executables (no extension), NOT .cu source files
	@cd $(OUTPUT_DIR) 2>/dev/null && rm -f b+tree $$(ls b+tree_* 2>/dev/null | grep -v '\.cu$$' | grep -v '\.h$$' | grep -v '\.c$$' | grep -v '\.cpp$$') || true
	$(RM) output.txt
	$(RM) -r $(BUILD_DIR)
	@echo "Clean complete."

# Deep clean (removes all build artifacts)
.PHONY: distclean
distclean: clean
	$(RM) -r .build

# List all detected variants
.PHONY: list-variants
list-variants:
	@echo "=========================================="
	@echo "Detected Variants"
	@echo "=========================================="
	@echo "Kernel 1 variants found:"
	@for v in $(VARIANT_SUFFIXES_1); do echo "  - $$v"; done
	@echo ""
	@echo "Kernel 2 variants found:"
	@for v in $(VARIANT_SUFFIXES_2); do echo "  - $$v"; done
	@echo ""
	@echo "Valid matching pairs:"
	@for v in $(VALID_VARIANTS); do echo "  - $$v"; done
	@echo ""
	@if [ "$(BASELINE_EXISTS)" = "yes" ]; then \
		echo "Baseline files: FOUND"; \
	else \
		echo "Baseline files: NOT FOUND"; \
	fi
	@echo ""
	@echo "Output directory: $(OUTPUT_DIR)/"
	@echo ""
	@echo "Executables to be generated:"
	@for exe in $(EXECUTABLES); do echo "  - $$exe"; done
	@echo "=========================================="

# Help target
.PHONY: help
help:
	@echo "=========================================="
	@echo "b+tree Multi-Variant Build System"
	@echo "=========================================="
	@echo ""
	@echo "Usage:"
	@echo "  make              Build all variants (continues on errors)"
	@echo "  make clean        Remove executables and build artifacts"
	@echo "  make distclean    Deep clean (removes .build directory)"
	@echo "  make list-variants List all detected variants"
	@echo "  make help         Show this help message"
	@echo ""
	@echo "File Naming Convention:"
	@echo "  Baseline:"
	@echo "    EX3_optimized_codes/kernel_gpu_cuda.cu"
	@echo "    EX3_optimized_codes/kernel_gpu_cuda_2.cu"
	@echo "    → generates: EX3_optimized_codes/b+tree"
	@echo ""
	@echo "  Variants:"
	@echo "    EX3_optimized_codes/kernel_gpu_cuda_<model>_<trail>.cu"
	@echo "    EX3_optimized_codes/kernel_gpu_cuda_2_<model>_<trail>.cu"
	@echo "    → generates: EX3_optimized_codes/b+tree_<model>_<trail>"
	@echo ""
	@echo "Output Directory:"
	@echo "  All executables are placed in: $(OUTPUT_DIR)/"
	@echo ""
	@echo "Features:"
	@echo "  - No intermediate .o files generated"
	@echo "  - Continues building on compilation errors"
	@echo "  - Both kernel files must exist for a variant to build"
	@echo "  - Executables placed in EX3_optimized_codes/"
	@echo "=========================================="

# Ensure the build directory exists
.PRECIOUS: $(BUILD_DIR)/kernel_gpu_cuda_wrapper_%.cu $(BUILD_DIR)/kernel_gpu_cuda_wrapper_2_%.cu
