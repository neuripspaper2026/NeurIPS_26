include ../common.mk

# Output directory for executables
OUTPUT_DIR = EX3_optimized_codes
BUILD_DIR = .build

EXE = hybridsort

# Main source file and fixed source files
MAIN_SRC = hybridsort.cu
BUCKETSORT_SRC = bucketsort.cu
MERGESORT_SRC = mergesort.cu

# Suppress warnings during compilation
NVCCFLAGS += -w

# Find bucketsort_kernel and mergesort_kernel variants in EX3_optimized_codes
BUCKETSORT_KERNEL_VARIANTS = $(wildcard $(OUTPUT_DIR)/bucketsort_kernel_*.cu)
BUCKETSORT_KERNEL_IDS = $(patsubst $(OUTPUT_DIR)/bucketsort_kernel_%.cu,%,$(BUCKETSORT_KERNEL_VARIANTS))

MERGESORT_KERNEL_VARIANTS = $(wildcard $(OUTPUT_DIR)/mergesort_kernel_*.cu)
MERGESORT_KERNEL_IDS = $(patsubst $(OUTPUT_DIR)/mergesort_kernel_%.cu,%,$(MERGESORT_KERNEL_VARIANTS))

# Check if baselines exist
BUCKETSORT_KERNEL_BASELINE_EXISTS = $(shell [ -f $(OUTPUT_DIR)/bucketsort_kernel.cu ] && echo yes)
MERGESORT_KERNEL_BASELINE_EXISTS = $(shell [ -f $(OUTPUT_DIR)/mergesort_kernel.cu ] && echo yes)

# Find common variant IDs (both bucketsort_kernel_ID and mergesort_kernel_ID must exist)
COMMON_VARIANT_IDS = $(filter $(BUCKETSORT_KERNEL_IDS),$(MERGESORT_KERNEL_IDS))

# Generate executable names
EXECUTABLES =
ifeq ($(BUCKETSORT_KERNEL_BASELINE_EXISTS),yes)
ifeq ($(MERGESORT_KERNEL_BASELINE_EXISTS),yes)
    EXECUTABLES += $(OUTPUT_DIR)/$(EXE)
endif
endif
EXECUTABLES += $(addprefix $(OUTPUT_DIR)/$(EXE)_,$(COMMON_VARIANT_IDS))

# Ensure output directory exists
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

# Build baseline version (baseline bucketsort_kernel + baseline mergesort_kernel)
ifeq ($(BUCKETSORT_KERNEL_BASELINE_EXISTS),yes)
ifeq ($(MERGESORT_KERNEL_BASELINE_EXISTS),yes)
$(OUTPUT_DIR)/$(EXE): $(MAIN_SRC) $(BUCKETSORT_SRC) $(MERGESORT_SRC) $(OUTPUT_DIR)/bucketsort_kernel.cu $(OUTPUT_DIR)/mergesort_kernel.cu
	@echo ""
	@echo "=========================================="
	@echo "Building: $@ (baseline)"
	@echo "=========================================="
	@cp $(OUTPUT_DIR)/bucketsort_kernel.cu bucketsort_kernel.cu
	@cp $(OUTPUT_DIR)/mergesort_kernel.cu mergesort_kernel.cu
	-@$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -c $(BUCKETSORT_SRC) -o bucketsort.o
	-@$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -c $(MERGESORT_SRC) -o mergesort.o
	-@$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(NVCC_LDLIBS) -o $@ $(MAIN_SRC) bucketsort.o mergesort.o
	@rm -f bucketsort_kernel.cu mergesort_kernel.cu bucketsort.o mergesort.o
	@if [ -f $@ ]; then echo "✓ Successfully built $@"; fi
endif
endif

# Build variant versions (paired bucketsort_kernel_ID + mergesort_kernel_ID)
$(OUTPUT_DIR)/$(EXE)_%: $(MAIN_SRC) $(BUCKETSORT_SRC) $(MERGESORT_SRC) $(OUTPUT_DIR)/bucketsort_kernel_%.cu $(OUTPUT_DIR)/mergesort_kernel_%.cu
	@echo ""
	@echo "=========================================="
	@echo "Building: $@ (variant $*)"
	@echo "=========================================="
	@cp $(OUTPUT_DIR)/bucketsort_kernel_$*.cu bucketsort_kernel.cu
	@cp $(OUTPUT_DIR)/mergesort_kernel_$*.cu mergesort_kernel.cu
	-@$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -c $(BUCKETSORT_SRC) -o bucketsort.o
	-@$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -c $(MERGESORT_SRC) -o mergesort.o
	-@$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(NVCC_LDLIBS) -o $@ $(MAIN_SRC) bucketsort.o mergesort.o
	@rm -f bucketsort_kernel.cu mergesort_kernel.cu bucketsort.o mergesort.o
	@if [ -f $@ ]; then echo "✓ Successfully built $@"; fi

.PHONY: clean
clean:
	@echo "Cleaning up..."
	@# Only remove executables (no extension), NOT .cu/.h/.c source files
	@cd $(OUTPUT_DIR) 2>/dev/null && rm -f $(EXE) $$(ls $(EXE)_* 2>/dev/null | grep -v '\.cu$$' | grep -v '\.h$$' | grep -v '\.c$$' | grep -v '\.cpp$$' | grep -v '\.cuh$$') || true
	$(RM) bucketsort.o mergesort.o bucketsort_kernel.cu mergesort_kernel.cu
	$(RM) -r $(BUILD_DIR)
	@echo "Clean complete."

.PHONY: list-variants
list-variants:
	@echo "Detected kernel files:"
	@echo "  bucketsort_kernel baseline: $(BUCKETSORT_KERNEL_BASELINE_EXISTS)"
	@echo "  bucketsort_kernel variant IDs: $(BUCKETSORT_KERNEL_IDS)"
	@echo "  mergesort_kernel baseline: $(MERGESORT_KERNEL_BASELINE_EXISTS)"
	@echo "  mergesort_kernel variant IDs: $(MERGESORT_KERNEL_IDS)"
	@echo ""
	@echo "Common paired variant IDs: $(COMMON_VARIANT_IDS)"
	@echo ""
	@echo "Executables to build:"
	@for exe in $(EXECUTABLES); do echo "  - $$exe"; done

.PHONY: help
help:
	@echo "Hybridsort Multi-Variant Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all            - Build all variants (baseline + variants)"
	@echo "  clean          - Remove executables and temporary files"
	@echo "  list-variants  - Show detected variants"
	@echo "  help           - Show this help message"
	@echo ""
	@echo "Variant Detection:"
	@echo "  - MUST have BOTH bucketsort_kernel_<ID>.cu AND mergesort_kernel_<ID>.cu"
	@echo "  - Baseline: bucketsort_kernel.cu + mergesort_kernel.cu"
	@echo "  - Variant format: bucketsort_kernel_<model>_<trail>.cu + mergesort_kernel_<model>_<trail>.cu"
	@echo ""
	@echo "Build Strategy:"
	@echo "  - Temporarily copy paired kernel files to main directory"
	@echo "  - Compile bucketsort.cu and mergesort.cu (which include the kernels)"
	@echo "  - Link with hybridsort.cu"
	@echo "  - Clean up temporary files"
	@echo ""
	@echo "Output:"
	@echo "  - Executables are placed in $(OUTPUT_DIR)/"
	@echo "  - hybridsort (baseline bucketsort_kernel + baseline mergesort_kernel)"
	@echo "  - hybridsort_<ID> (bucketsort_kernel_<ID> + mergesort_kernel_<ID>)"
