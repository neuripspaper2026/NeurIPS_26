include ../common.mk

# Output directory for executables
OUTPUT_DIR = EX3_optimized_codes

EXE = streamcluster

# Source files
MAIN_CPP = streamcluster.cpp
HEADER = streamcluster_header.cu

# Kernel file in main directory (temporary placeholder)
KERNEL_LOCAL = streamcluster_cuda.cu

# Suppress warnings during compilation
NVCCFLAGS += -w

# Find kernel variants in EX3_optimized_codes
KERNEL_VARIANTS = $(wildcard $(OUTPUT_DIR)/streamcluster_cuda_*.cu)
KERNEL_VARIANT_IDS = $(patsubst $(OUTPUT_DIR)/streamcluster_cuda_%.cu,%,$(KERNEL_VARIANTS))

# Check if baseline exists
KERNEL_BASELINE_EXISTS = $(shell [ -f $(OUTPUT_DIR)/streamcluster_cuda.cu ] && echo yes)

# Generate executable names
EXECUTABLES =
ifeq ($(KERNEL_BASELINE_EXISTS),yes)
    EXECUTABLES += $(OUTPUT_DIR)/$(EXE)
endif
EXECUTABLES += $(addprefix $(OUTPUT_DIR)/$(EXE)_,$(KERNEL_VARIANT_IDS))

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

# Build baseline version
ifeq ($(KERNEL_BASELINE_EXISTS),yes)
$(OUTPUT_DIR)/$(EXE): $(MAIN_CPP) $(OUTPUT_DIR)/streamcluster_cuda.cu $(HEADER)
	@echo ""
	@echo "=========================================="
	@echo "Building: $@ (baseline)"
	@echo "=========================================="
	@cp $(OUTPUT_DIR)/streamcluster_cuda.cu $(KERNEL_LOCAL)
	-@$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(NVCC_LDLIBS) -o $@ $(MAIN_CPP) $(KERNEL_LOCAL)
	@rm -f $(KERNEL_LOCAL)
	@if [ -f $@ ]; then echo "✓ Successfully built $@"; fi
endif

# Build variant versions
$(OUTPUT_DIR)/$(EXE)_%: $(MAIN_CPP) $(OUTPUT_DIR)/streamcluster_cuda_%.cu $(HEADER)
	@echo ""
	@echo "=========================================="
	@echo "Building: $@ (variant $*)"
	@echo "=========================================="
	@cp $(OUTPUT_DIR)/streamcluster_cuda_$*.cu $(KERNEL_LOCAL)
	-@$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(NVCC_LDLIBS) -o $@ $(MAIN_CPP) $(KERNEL_LOCAL)
	@rm -f $(KERNEL_LOCAL)
	@if [ -f $@ ]; then echo "✓ Successfully built $@"; fi

.PHONY: clean
clean:
	@echo "Cleaning up..."
	@# Only remove executables (no extension), NOT .cu/.h/.c/.cpp source files
	@cd $(OUTPUT_DIR) 2>/dev/null && rm -f $(EXE) $$(ls $(EXE)_* 2>/dev/null | grep -v '\.cu$$' | grep -v '\.h$$' | grep -v '\.c$$' | grep -v '\.cpp$$') || true
	$(RM) $(KERNEL_LOCAL)
	@echo "Clean complete."

.PHONY: list-variants
list-variants:
	@echo "Detected kernel files:"
	@echo "  Baseline: $(KERNEL_BASELINE_EXISTS)"
	@echo "  Variant IDs: $(KERNEL_VARIANT_IDS)"
	@echo ""
	@echo "Executables to build:"
	@for exe in $(EXECUTABLES); do echo "  - $$exe"; done

.PHONY: help
help:
	@echo "Streamcluster Multi-Variant Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all            - Build all variants (baseline + variants)"
	@echo "  clean          - Remove executables and temporary files"
	@echo "  list-variants  - Show detected variants"
	@echo "  help           - Show this help message"
	@echo ""
	@echo "Variant Detection:"
	@echo "  - Baseline: streamcluster_cuda.cu"
	@echo "  - Variant format: streamcluster_cuda_<model>_<trail>.cu"
	@echo ""
	@echo "Build Strategy:"
	@echo "  - Temporarily copy kernel file to main directory"
	@echo "  - Compile streamcluster.cpp (which includes the kernel via header)"
	@echo "  - Clean up temporary files"
	@echo ""
	@echo "Output:"
	@echo "  - Executables are placed in $(OUTPUT_DIR)/"
	@echo "  - streamcluster (baseline streamcluster_cuda)"
	@echo "  - streamcluster_<ID> (streamcluster_cuda_<ID>)"
