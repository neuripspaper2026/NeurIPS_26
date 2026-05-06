include ../common.mk

# Output directory for executables
OUTPUT_DIR = EX3_optimized_codes

EXE = particlefilter

# Kernel file in EX3_optimized_codes (this is a complete standalone program)
KERNEL_SRC_BASELINE = $(OUTPUT_DIR)/particlefilter_naive.cu

# Suppress warnings during compilation
NVCCFLAGS += -w

# Find kernel variants in EX3_optimized_codes (particlefilter_naive_*.cu)
KERNEL_VARIANTS = $(wildcard $(OUTPUT_DIR)/particlefilter_naive_*.cu)
# Extract variant IDs (remove "particlefilter_naive_" prefix, keep model_trail)
KERNEL_VARIANT_IDS = $(patsubst $(OUTPUT_DIR)/particlefilter_naive_%.cu,%,$(KERNEL_VARIANTS))

# Check if baseline exists
KERNEL_BASELINE_EXISTS = $(shell [ -f $(KERNEL_SRC_BASELINE) ] && echo yes)

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
$(OUTPUT_DIR)/$(EXE): $(KERNEL_SRC_BASELINE)
	@echo ""
	@echo "=========================================="
	@echo "Building: $@ (baseline)"
	@echo "=========================================="
	-@$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(NVCC_LDLIBS) -lm -o $@ $<
	@if [ -f $@ ]; then echo "✓ Successfully built $@"; fi
endif

# Build variant versions (source: particlefilter_naive_*.cu → output: particlefilter_*.cu)
$(OUTPUT_DIR)/$(EXE)_%: $(OUTPUT_DIR)/particlefilter_naive_%.cu
	@echo ""
	@echo "=========================================="
	@echo "Building: $@ (variant $*)"
	@echo "=========================================="
	-@$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(NVCC_LDLIBS) -lm -o $@ $<
	@if [ -f $@ ]; then echo "✓ Successfully built $@"; fi

.PHONY: clean
clean:
	@echo "Cleaning up..."
	@# Only remove executables, NOT source files
	@cd $(OUTPUT_DIR) 2>/dev/null && rm -f $(EXE) $$(ls $(EXE)_* 2>/dev/null | grep -v '\.cu$$' | grep -v '\.h$$') || true
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
	@echo "Particle Filter Multi-Variant Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all            - Build all variants (baseline + variants)"
	@echo "  clean          - Remove executables"
	@echo "  list-variants  - Show detected variants"
	@echo "  help           - Show this help message"
	@echo ""
	@echo "Variant Detection:"
	@echo "  - Baseline: particlefilter_naive.cu → particlefilter"
	@echo "  - Variant format: particlefilter_naive_<model>_<trail>.cu → particlefilter_<model>_<trail>"
	@echo ""
	@echo "Build Strategy:"
	@echo "  - Direct compilation (standalone file, no intermediate .o files)"
	@echo "  - Each variant is a complete CUDA program"
	@echo ""
	@echo "Output:"
	@echo "  - Executables are placed in $(OUTPUT_DIR)/"
	@echo "  - particlefilter (baseline, from particlefilter_naive.cu)"
	@echo "  - particlefilter_<model>_<trail> (variants, from particlefilter_naive_<model>_<trail>.cu)"
