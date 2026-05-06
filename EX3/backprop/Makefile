include ../common.mk

# Source files
MAIN_SRC = backprop_cuda.cu
C_SRCS = backprop.c facetrain.c imagenet.c

# Output directory for executables
OUTPUT_DIR = EX3_optimized_codes

# Build directory for temporary files
BUILD_DIR = .build
$(shell mkdir -p $(BUILD_DIR) $(OUTPUT_DIR))

# Find all variant kernel files
KERNEL_VARIANTS := $(wildcard $(OUTPUT_DIR)/backprop_cuda_kernel_*.cu)

# Extract variant suffixes (e.g., backprop_cuda_kernel_gpt4_v1.cu → gpt4_v1)
VARIANT_SUFFIXES := $(patsubst $(OUTPUT_DIR)/backprop_cuda_kernel_%.cu,%,$(KERNEL_VARIANTS))

# Check for baseline kernel
BASELINE_EXISTS := $(shell [ -f $(OUTPUT_DIR)/backprop_cuda_kernel.cu ] && echo yes)

# Generate executable names (with output directory prefix)
ifeq ($(BASELINE_EXISTS),yes)
    EXECUTABLES := $(OUTPUT_DIR)/backprop
endif
EXECUTABLES += $(addprefix $(OUTPUT_DIR)/backprop_,$(VARIANT_SUFFIXES))

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

# Generate temporary main CUDA file for baseline
# This modifies the #include to point to the baseline kernel in EX3_optimized_codes/
$(BUILD_DIR)/backprop_cuda_baseline.cu: $(MAIN_SRC)
	@mkdir -p $(BUILD_DIR)
	@sed 's|#include "backprop_cuda_kernel\.cu"|#include "../$(OUTPUT_DIR)/backprop_cuda_kernel.cu"|g' $< > $@

# Generate temporary main CUDA file for each variant
$(BUILD_DIR)/backprop_cuda_%.cu: $(MAIN_SRC)
	@mkdir -p $(BUILD_DIR)
	@sed 's|#include "backprop_cuda_kernel\.cu"|#include "../$(OUTPUT_DIR)/backprop_cuda_kernel_$*.cu"|g' $< > $@

# Build baseline executable
$(OUTPUT_DIR)/backprop: $(BUILD_DIR)/backprop_cuda_baseline.cu $(C_SRCS)
	@echo ""
	@echo "=========================================="
	@echo "Building: $@ (baseline)"
	@echo "=========================================="
	@mkdir -p $(OUTPUT_DIR)
	-@$(NVCC) $(CPPFLAGS) -I. $(NVCCFLAGS) $(NVCC_LDLIBS) -lm -o $@ $^ 2>&1 | grep -v "warning:" || true
	@if [ -f $@ ]; then \
		echo "✓ Successfully built $@"; \
	else \
		echo "✗ Failed to build $@ (continuing...)"; \
	fi

# Build variant executables
$(OUTPUT_DIR)/backprop_%: $(BUILD_DIR)/backprop_cuda_%.cu $(C_SRCS)
	@echo ""
	@echo "=========================================="
	@echo "Building: $@ (variant: $*)"
	@echo "=========================================="
	@mkdir -p $(OUTPUT_DIR)
	-@$(NVCC) $(CPPFLAGS) -I. $(NVCCFLAGS) $(NVCC_LDLIBS) -lm -o $@ $^ 2>&1 | grep -v "warning:" || true
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
	@cd $(OUTPUT_DIR) 2>/dev/null && rm -f backprop $$(ls backprop_* 2>/dev/null | grep -v '\.cu$$' | grep -v '\.h$$' | grep -v '\.c$$' | grep -v '\.cpp$$') || true
	$(RM) output.dat
	$(RM) -r $(BUILD_DIR)
	@echo "Clean complete."

# Deep clean
.PHONY: distclean
distclean: clean
	$(RM) -r .build

# List all detected variants
.PHONY: list-variants
list-variants:
	@echo "=========================================="
	@echo "Detected Variants"
	@echo "=========================================="
	@echo "Kernel variants found:"
	@for v in $(VARIANT_SUFFIXES); do echo "  - $$v"; done
	@echo ""
	@if [ "$(BASELINE_EXISTS)" = "yes" ]; then \
		echo "Baseline kernel: FOUND"; \
	else \
		echo "Baseline kernel: NOT FOUND"; \
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
	@echo "Backprop Multi-Variant Build System"
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
	@echo "    $(OUTPUT_DIR)/backprop_cuda_kernel.cu"
	@echo "    → generates: $(OUTPUT_DIR)/backprop"
	@echo ""
	@echo "  Variants:"
	@echo "    $(OUTPUT_DIR)/backprop_cuda_kernel_<model>_<trail>.cu"
	@echo "    → generates: $(OUTPUT_DIR)/backprop_<model>_<trail>"
	@echo ""
	@echo "Output Directory:"
	@echo "  All executables are placed in: $(OUTPUT_DIR)/"
	@echo ""
	@echo "Features:"
	@echo "  - No intermediate .o files generated"
	@echo "  - Continues building on compilation errors"
	@echo "  - Executables placed in EX3_optimized_codes/"
	@echo "=========================================="

# Preserve temporary files for debugging
.PRECIOUS: $(BUILD_DIR)/backprop_cuda_%.cu
