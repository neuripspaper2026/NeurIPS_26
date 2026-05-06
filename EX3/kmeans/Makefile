include ../common.mk

# Output directory for executables
OUTPUT_DIR = EX3_optimized_codes
BUILD_DIR = .build

EXE = kmeans

# Main source file and C source files
MAIN_SRC = kmeans_cuda.cu
C_SRCS = cluster.c kmeans.c kmeans_clustering.c rmse.c

# Suppress warnings during compilation
NVCCFLAGS += -w

# Find kmeans_cuda_kernel variants in EX3_optimized_codes
KERNEL_VARIANTS = $(wildcard $(OUTPUT_DIR)/kmeans_cuda_kernel_*.cu)
KERNEL_IDS = $(patsubst $(OUTPUT_DIR)/kmeans_cuda_kernel_%.cu,%,$(KERNEL_VARIANTS))

# Check if baseline exists
KERNEL_BASELINE_EXISTS = $(shell [ -f $(OUTPUT_DIR)/kmeans_cuda_kernel.cu ] && echo yes)

# Generate executable names
EXECUTABLES =
ifeq ($(KERNEL_BASELINE_EXISTS),yes)
    EXECUTABLES += $(OUTPUT_DIR)/$(EXE)
endif
EXECUTABLES += $(addprefix $(OUTPUT_DIR)/$(EXE)_,$(KERNEL_IDS))

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

# Build baseline version (direct compilation, no .o files)
ifeq ($(KERNEL_BASELINE_EXISTS),yes)
$(OUTPUT_DIR)/$(EXE): $(MAIN_SRC) $(OUTPUT_DIR)/kmeans_cuda_kernel.cu $(C_SRCS)
	@echo ""
	@echo "=========================================="
	@echo "Building: $@ (baseline)"
	@echo "=========================================="
	@cp $(OUTPUT_DIR)/kmeans_cuda_kernel.cu kmeans_cuda_kernel.cu
	-@$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(NVCC_LDLIBS) -o $@ $(MAIN_SRC) $(C_SRCS)
	@rm -f kmeans_cuda_kernel.cu
	@if [ -f $@ ]; then echo "✓ Successfully built $@"; fi
endif

# Build variant versions (direct compilation, no .o files)
$(OUTPUT_DIR)/$(EXE)_%: $(MAIN_SRC) $(OUTPUT_DIR)/kmeans_cuda_kernel_%.cu $(C_SRCS)
	@echo ""
	@echo "=========================================="
	@echo "Building: $@ (variant $*)"
	@echo "=========================================="
	@cp $(OUTPUT_DIR)/kmeans_cuda_kernel_$*.cu kmeans_cuda_kernel.cu
	-@$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(NVCC_LDLIBS) -o $@ $(MAIN_SRC) $(C_SRCS)
	@rm -f kmeans_cuda_kernel.cu
	@if [ -f $@ ]; then echo "✓ Successfully built $@"; fi

.PHONY: clean
clean:
	@echo "Cleaning up..."
	@# Only remove executables (no extension), NOT .cu/.h/.c source files
	@cd $(OUTPUT_DIR) 2>/dev/null && rm -f $(EXE) $$(ls $(EXE)_* 2>/dev/null | grep -v '\.cu$$' | grep -v '\.h$$' | grep -v '\.c$$' | grep -v '\.cpp$$') || true
	$(RM) kmeans_cuda_kernel.cu output.txt
	$(RM) -r $(BUILD_DIR)
	@echo "Clean complete."

.PHONY: list-variants
list-variants:
	@echo "Detected variants:"
	@echo "  kmeans_cuda_kernel baseline: $(KERNEL_BASELINE_EXISTS)"
	@echo "  kmeans_cuda_kernel variant IDs: $(KERNEL_IDS)"
	@echo ""
	@echo "Executables to build:"
	@for exe in $(EXECUTABLES); do echo "  - $$exe"; done

.PHONY: help
help:
	@echo "Kmeans Multi-Variant Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all            - Build all variants (baseline + variants)"
	@echo "  clean          - Remove executables and temporary files"
	@echo "  list-variants  - Show detected variants"
	@echo "  help           - Show this help message"
	@echo ""
	@echo "Variant Detection:"
	@echo "  - Baseline: kmeans_cuda_kernel.cu"
	@echo "  - Variants: kmeans_cuda_kernel_<model>_<trail>.cu"
	@echo ""
	@echo "Build Strategy:"
	@echo "  - Temporarily copy kernel file to main directory"
	@echo "  - Direct compilation (no intermediate .o files)"
	@echo "  - Compile kmeans_cuda.cu and all C files together"
	@echo "  - Clean up temporary files"
	@echo ""
	@echo "Output:"
	@echo "  - Executables are placed in $(OUTPUT_DIR)/"
