include ../common.mk

# Output directory for executables
OUTPUT_DIR = EX3_optimized_codes

EXE = srad_v1

# Main source file
MAIN_SRC = srad.cu

# Other source files (included via #include in srad.cu)
# We don't compile these separately, but list them as dependencies
OTHER_SRCS = define.c extract_kernel.cu prepare_kernel.cu reduce_kernel.cu compress_kernel.cu graphics.c resize.c timer.c device.c

# Kernel files in main directory (temporary placeholders for includes)
SRAD_KERNEL_LOCAL = srad_kernel.cu
SRAD2_KERNEL_LOCAL = srad2_kernel.cu

# Suppress warnings during compilation
NVCCFLAGS += -w

# Find kernel variants in EX3_optimized_codes
SRAD_KERNEL_VARIANTS = $(wildcard $(OUTPUT_DIR)/srad_kernel_*.cu)
SRAD_KERNEL_IDS = $(patsubst $(OUTPUT_DIR)/srad_kernel_%.cu,%,$(SRAD_KERNEL_VARIANTS))

SRAD2_KERNEL_VARIANTS = $(wildcard $(OUTPUT_DIR)/srad2_kernel_*.cu)
SRAD2_KERNEL_IDS = $(patsubst $(OUTPUT_DIR)/srad2_kernel_%.cu,%,$(SRAD2_KERNEL_VARIANTS))

# Check if baselines exist
SRAD_KERNEL_BASELINE_EXISTS = $(shell [ -f $(OUTPUT_DIR)/srad_kernel.cu ] && echo yes)
SRAD2_KERNEL_BASELINE_EXISTS = $(shell [ -f $(OUTPUT_DIR)/srad2_kernel.cu ] && echo yes)

# Find common variant IDs (both kernels must have the same variant ID)
COMMON_VARIANT_IDS = $(filter $(SRAD_KERNEL_IDS),$(SRAD2_KERNEL_IDS))

# Generate executable names
EXECUTABLES =
ifeq ($(SRAD_KERNEL_BASELINE_EXISTS),yes)
ifeq ($(SRAD2_KERNEL_BASELINE_EXISTS),yes)
    EXECUTABLES += $(OUTPUT_DIR)/$(EXE)
endif
endif
EXECUTABLES += $(addprefix $(OUTPUT_DIR)/$(EXE)_,$(COMMON_VARIANT_IDS))

# Ensure output directory exists
$(shell mkdir -p $(OUTPUT_DIR))

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

# Build baseline version (both baseline kernels)
ifeq ($(SRAD_KERNEL_BASELINE_EXISTS),yes)
ifeq ($(SRAD2_KERNEL_BASELINE_EXISTS),yes)
$(OUTPUT_DIR)/$(EXE): $(MAIN_SRC) $(OUTPUT_DIR)/srad_kernel.cu $(OUTPUT_DIR)/srad2_kernel.cu $(OTHER_SRCS)
	@echo ""
	@echo "=========================================="
	@echo "Building: $@ (baseline)"
	@echo "=========================================="
	@cp $(OUTPUT_DIR)/srad_kernel.cu $(SRAD_KERNEL_LOCAL)
	@cp $(OUTPUT_DIR)/srad2_kernel.cu $(SRAD2_KERNEL_LOCAL)
	-@$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(NVCC_LDLIBS) -o $@ $(MAIN_SRC)
	@rm -f $(SRAD_KERNEL_LOCAL) $(SRAD2_KERNEL_LOCAL)
	@rm -f *.o
	@if [ -f $@ ]; then echo "✓ Successfully built $@"; fi
endif
endif

# Build variant versions (both kernels must have the same variant ID)
$(OUTPUT_DIR)/$(EXE)_%: $(MAIN_SRC) $(OUTPUT_DIR)/srad_kernel_%.cu $(OUTPUT_DIR)/srad2_kernel_%.cu $(OTHER_SRCS)
	@echo ""
	@echo "=========================================="
	@echo "Building: $@ (variant $*)"
	@echo "=========================================="
	@cp $(OUTPUT_DIR)/srad_kernel_$*.cu $(SRAD_KERNEL_LOCAL)
	@cp $(OUTPUT_DIR)/srad2_kernel_$*.cu $(SRAD2_KERNEL_LOCAL)
	-@$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(NVCC_LDLIBS) -o $@ $(MAIN_SRC)
	@rm -f $(SRAD_KERNEL_LOCAL) $(SRAD2_KERNEL_LOCAL)
	@rm -f *.o
	@if [ -f $@ ]; then echo "✓ Successfully built $@"; fi

.PHONY: clean
clean:
	@echo "Cleaning up..."
	@# Only remove executables (no extension), NOT .cu/.h/.c source files
	@cd $(OUTPUT_DIR) 2>/dev/null && rm -f $(EXE) $$(ls $(EXE)_* 2>/dev/null | grep -v '\.cu$$' | grep -v '\.h$$' | grep -v '\.c$$' | grep -v '\.cpp$$') || true
	$(RM) $(SRAD_KERNEL_LOCAL) $(SRAD2_KERNEL_LOCAL)
	$(RM) *.o
	@echo "Clean complete."

.PHONY: list-variants
list-variants:
	@echo "Detected kernel files:"
	@echo "  srad_kernel baseline: $(SRAD_KERNEL_BASELINE_EXISTS)"
	@echo "  srad_kernel variant IDs: $(SRAD_KERNEL_IDS)"
	@echo "  srad2_kernel baseline: $(SRAD2_KERNEL_BASELINE_EXISTS)"
	@echo "  srad2_kernel variant IDs: $(SRAD2_KERNEL_IDS)"
	@echo ""
	@echo "Common paired variant IDs: $(COMMON_VARIANT_IDS)"
	@echo ""
	@echo "Executables to build:"
	@for exe in $(EXECUTABLES); do echo "  - $$exe"; done

.PHONY: help
help:
	@echo "SRAD_v1 Multi-Variant Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all            - Build all variants (baseline + variants)"
	@echo "  clean          - Remove executables and temporary files"
	@echo "  list-variants  - Show detected variants"
	@echo "  help           - Show this help message"
	@echo ""
	@echo "Variant Detection:"
	@echo "  - MUST have BOTH: srad_kernel_<ID>.cu AND srad2_kernel_<ID>.cu"
	@echo "  - Baseline: srad_kernel.cu + srad2_kernel.cu"
	@echo "  - Variant format: srad_kernel_<model>_<trail>.cu + srad2_kernel_<model>_<trail>.cu"
	@echo ""
	@echo "Build Strategy:"
	@echo "  - Temporarily copy both kernel files to main directory"
	@echo "  - Compile srad.cu (which includes the kernels via #include)"
	@echo "  - Clean up temporary files"
	@echo ""
	@echo "Output:"
	@echo "  - Executables are placed in $(OUTPUT_DIR)/"
	@echo "  - srad (baseline srad_kernel + srad2_kernel)"
	@echo "  - srad_<ID> (srad_kernel_<ID> + srad2_kernel_<ID>)"
