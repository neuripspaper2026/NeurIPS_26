include ../common.mk

# Output directory for executables
OUTPUT_DIR = EX3_optimized_codes
BUILD_DIR = .build

# Source files
MAIN_SRC = bfs.cu

# Find all kernel variant files
KERNEL_1_VARIANTS = $(wildcard $(OUTPUT_DIR)/kernel_*.cu)
KERNEL_2_VARIANTS = $(wildcard $(OUTPUT_DIR)/kernel2_*.cu)

# Extract model and trail identifiers from filenames
# kernel_<model>_<trail>.cu -> <model>_<trail>
KERNEL_1_IDS = $(patsubst $(OUTPUT_DIR)/kernel_%.cu,%,$(KERNEL_1_VARIANTS))
KERNEL_2_IDS = $(patsubst $(OUTPUT_DIR)/kernel2_%.cu,%,$(KERNEL_2_VARIANTS))

# Find valid variant pairs (both kernel.cu and kernel2.cu must exist)
VALID_VARIANTS = $(filter $(KERNEL_1_IDS),$(KERNEL_2_IDS))

# Check if baseline exists (both kernel.cu and kernel2.cu)
BASELINE_EXISTS = $(shell [ -f $(OUTPUT_DIR)/kernel.cu ] && [ -f $(OUTPUT_DIR)/kernel2.cu ] && echo yes)

# Generate executable names
ifeq ($(BASELINE_EXISTS),yes)
    EXECUTABLES = $(OUTPUT_DIR)/bfs $(addprefix $(OUTPUT_DIR)/bfs_,$(VALID_VARIANTS))
else
    EXECUTABLES = $(addprefix $(OUTPUT_DIR)/bfs_,$(VALID_VARIANTS))
endif

# Ensure output and build directories exist
$(shell mkdir -p $(OUTPUT_DIR) $(BUILD_DIR))

.PHONY: all
all: $(EXECUTABLES)
	@echo ""
	@echo "=========================================="
	@echo "Build Summary:"
	@echo "=========================================="
	@if [ -n "$(EXECUTABLES)" ]; then \
		for exe in $(EXECUTABLES); do \
			if [ -f $$exe ]; then \
				echo "✓ $$exe - SUCCESS"; \
			else \
				echo "✗ $$exe - FAILED"; \
			fi; \
		done; \
	else \
		echo "No variants found to build."; \
	fi
	@echo "=========================================="

# Build baseline version
ifeq ($(BASELINE_EXISTS),yes)
$(OUTPUT_DIR)/bfs: $(BUILD_DIR)/bfs_baseline.cu
	@echo ""
	@echo "=========================================="
	@echo "Building: $@ (baseline)"
	@echo "=========================================="
	-@$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(NVCC_LDLIBS) -o $@ $<
	@if [ -f $@ ]; then echo "✓ Successfully built $@"; fi
endif

# Build variant versions
$(OUTPUT_DIR)/bfs_%: $(BUILD_DIR)/bfs_%.cu
	@echo ""
	@echo "=========================================="
	@echo "Building: $@ (variant $*)"
	@echo "=========================================="
	-@$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(NVCC_LDLIBS) -o $@ $<
	@if [ -f $@ ]; then echo "✓ Successfully built $@"; fi

# Generate temporary baseline bfs.cu (redirects includes to baseline kernels)
$(BUILD_DIR)/bfs_baseline.cu: $(MAIN_SRC) $(OUTPUT_DIR)/kernel.cu $(OUTPUT_DIR)/kernel2.cu
	@mkdir -p $(BUILD_DIR)
	@sed 's|#include "EX3_optimized_codes/kernel\.cu"|#include "../$(OUTPUT_DIR)/kernel.cu"|g; s|#include "EX3_optimized_codes/kernel2\.cu"|#include "../$(OUTPUT_DIR)/kernel2.cu"|g' $(MAIN_SRC) > $@

# Generate temporary variant bfs.cu files (redirects includes to variant kernels)
$(BUILD_DIR)/bfs_%.cu: $(MAIN_SRC) $(OUTPUT_DIR)/kernel_%.cu $(OUTPUT_DIR)/kernel2_%.cu
	@mkdir -p $(BUILD_DIR)
	@sed 's|#include "EX3_optimized_codes/kernel\.cu"|#include "../$(OUTPUT_DIR)/kernel_$*.cu"|g; s|#include "EX3_optimized_codes/kernel2\.cu"|#include "../$(OUTPUT_DIR)/kernel2_$*.cu"|g' $(MAIN_SRC) > $@

# Clean target - CRITICAL: Only remove executables, NOT .cu source files
.PHONY: clean
clean:
	@echo "Cleaning up..."
	@# Only remove executables (no extension), NOT .cu/.h/.c source files
	@cd $(OUTPUT_DIR) 2>/dev/null && rm -f bfs $$(ls bfs_* 2>/dev/null | grep -v '\.cu$$' | grep -v '\.h$$' | grep -v '\.c$$' | grep -v '\.cpp$$') || true
	$(RM) output.txt
	$(RM) -r $(BUILD_DIR)
	@echo "Clean complete."

# List all detected variants
.PHONY: list-variants
list-variants:
	@echo "Detected kernel variants:"
	@echo "  Baseline: $(BASELINE_EXISTS)"
	@echo "  Kernel 1 IDs: $(KERNEL_1_IDS)"
	@echo "  Kernel 2 IDs: $(KERNEL_2_IDS)"
	@echo "  Valid pairs: $(VALID_VARIANTS)"
	@echo ""
	@echo "Executables to build:"
	@for exe in $(EXECUTABLES); do echo "  - $$exe"; done

.PHONY: help
help:
	@echo "BFS Multi-Variant Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all            - Build all variants (baseline + variants)"
	@echo "  clean          - Remove executables and temporary files"
	@echo "  list-variants  - Show detected variants"
	@echo "  help           - Show this help message"
	@echo ""
	@echo "Variant Detection:"
	@echo "  - Baseline: kernel.cu + kernel2.cu"
	@echo "  - Variants: kernel_<model>_<trail>.cu + kernel2_<model>_<trail>.cu"
	@echo ""
	@echo "Output:"
	@echo "  - Executables are placed in $(OUTPUT_DIR)/"
	@echo "  - Baseline: $(OUTPUT_DIR)/bfs"
	@echo "  - Variants: $(OUTPUT_DIR)/bfs_<model>_<trail>"
	@echo ""
	@echo "Note: Both kernel.cu and kernel2.cu variants must exist for a valid build."
