include ../common.mk

# Output directory for executables
OUTPUT_DIR = EX3_optimized_codes
BUILD_DIR = .build

# Source files
MAIN_CPP = bilateralFilter.cpp
BMP_CPP = bmploader.cpp
KERNEL_CU = $(OUTPUT_DIR)/bilateral_kernel.cu

# Find all kernel variant files
KERNEL_VARIANTS = $(wildcard $(OUTPUT_DIR)/bilateral_kernel_*.cu)

# Extract variant identifiers from filenames
# bilateral_kernel_<model>_<trail>.cu -> <model>_<trail>
VARIANT_IDS = $(patsubst $(OUTPUT_DIR)/bilateral_kernel_%.cu,%,$(KERNEL_VARIANTS))

# Check if baseline kernel exists
BASELINE_EXISTS = $(shell [ -f $(OUTPUT_DIR)/bilateral_kernel.cu ] && echo yes)

# Generate executable names
ifeq ($(BASELINE_EXISTS),yes)
    EXECUTABLES = $(OUTPUT_DIR)/bilateral $(addprefix $(OUTPUT_DIR)/bilateral_,$(VARIANT_IDS))
else
    EXECUTABLES = $(addprefix $(OUTPUT_DIR)/bilateral_,$(VARIANT_IDS))
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
$(OUTPUT_DIR)/bilateral: $(MAIN_CPP) $(BMP_CPP) $(OUTPUT_DIR)/bilateral_kernel.cu
	@echo ""
	@echo "=========================================="
	@echo "Building: $@ (baseline)"
	@echo "=========================================="
	-@$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(NVCC_LDLIBS) -o $@ $(MAIN_CPP) $(BMP_CPP) $(OUTPUT_DIR)/bilateral_kernel.cu
	@if [ -f $@ ]; then echo "✓ Successfully built $@"; fi
endif

# Build variant versions
$(OUTPUT_DIR)/bilateral_%: $(MAIN_CPP) $(BMP_CPP) $(OUTPUT_DIR)/bilateral_kernel_%.cu
	@echo ""
	@echo "=========================================="
	@echo "Building: $@ (variant $*)"
	@echo "=========================================="
	-@$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(NVCC_LDLIBS) -o $@ $(MAIN_CPP) $(BMP_CPP) $(OUTPUT_DIR)/bilateral_kernel_$*.cu
	@if [ -f $@ ]; then echo "✓ Successfully built $@"; fi

# Clean target - CRITICAL: Only remove executables, NOT .cu source files
.PHONY: clean
clean:
	@echo "Cleaning up..."
	@# Only remove executables (no extension), NOT .cu/.h/.c/.cpp source files
	@cd $(OUTPUT_DIR) 2>/dev/null && rm -f bilateral $$(ls bilateral_* 2>/dev/null | grep -v '\.cu$$' | grep -v '\.h$$' | grep -v '\.c$$' | grep -v '\.cpp$$') || true
	$(RM) output.ppm
	$(RM) -r $(BUILD_DIR)
	@echo "Clean complete."

# List all detected variants
.PHONY: list-variants
list-variants:
	@echo "Detected kernel variants:"
	@echo "  Baseline: $(BASELINE_EXISTS)"
	@echo "  Variant IDs: $(VARIANT_IDS)"
	@echo ""
	@echo "Executables to build:"
	@for exe in $(EXECUTABLES); do echo "  - $$exe"; done

.PHONY: help
help:
	@echo "Bilateral Filter Multi-Variant Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all            - Build all variants (baseline + variants)"
	@echo "  clean          - Remove executables and temporary files"
	@echo "  list-variants  - Show detected variants"
	@echo "  help           - Show this help message"
	@echo ""
	@echo "Variant Detection:"
	@echo "  - Baseline: bilateral_kernel.cu"
	@echo "  - Variants: bilateral_kernel_<model>_<trail>.cu"
	@echo ""
	@echo "Output:"
	@echo "  - Executables are placed in $(OUTPUT_DIR)/"
	@echo "  - Baseline: $(OUTPUT_DIR)/bilateral"
	@echo "  - Variants: $(OUTPUT_DIR)/bilateral_<model>_<trail>"
	@echo ""
	@echo "Note: All source files compiled directly into executable (no .o files)."
