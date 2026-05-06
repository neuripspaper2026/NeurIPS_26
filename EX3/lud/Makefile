include ../common.mk

# Output directory for executables
OUTPUT_DIR = EX3_optimized_codes

EXE = lud

# Source files
MAIN_SRC = lud.cu
COMMON_SRC = common/common.c

# Kernel file in EX3_optimized_codes
KERNEL_SRC_BASELINE = $(OUTPUT_DIR)/lud_kernel.cu
KERNEL_LOCAL = lud_kernel.cu

# Compiler flags
CPPFLAGS += -Icommon
NVCCFLAGS += -w

# Find kernel variants in EX3_optimized_codes
KERNEL_VARIANTS = $(wildcard $(OUTPUT_DIR)/lud_kernel_*.cu)
KERNEL_VARIANT_IDS = $(patsubst $(OUTPUT_DIR)/lud_kernel_%.cu,%,$(KERNEL_VARIANTS))

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
$(OUTPUT_DIR)/$(EXE): $(MAIN_SRC) $(KERNEL_SRC_BASELINE) $(COMMON_SRC)
	@echo ""
	@echo "=========================================="
	@echo "Building: $@ (baseline)"
	@echo "=========================================="
	@cp $(KERNEL_SRC_BASELINE) $(KERNEL_LOCAL)
	-@$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(NVCC_LDLIBS) -o $@ $(MAIN_SRC) $(KERNEL_LOCAL) $(COMMON_SRC)
	@rm -f $(KERNEL_LOCAL)
	@if [ -f $@ ]; then echo "✓ Successfully built $@"; fi
endif

# Build variant versions
$(OUTPUT_DIR)/$(EXE)_%: $(MAIN_SRC) $(OUTPUT_DIR)/lud_kernel_%.cu $(COMMON_SRC)
	@echo ""
	@echo "=========================================="
	@echo "Building: $@ (variant $*)"
	@echo "=========================================="
	@cp $(OUTPUT_DIR)/lud_kernel_$*.cu $(KERNEL_LOCAL)
	-@$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(NVCC_LDLIBS) -o $@ $(MAIN_SRC) $(KERNEL_LOCAL) $(COMMON_SRC)
	@rm -f $(KERNEL_LOCAL)
	@if [ -f $@ ]; then echo "✓ Successfully built $@"; fi

.PHONY: clean
clean:
	@echo "Cleaning up..."
	@# Only remove executables, NOT source files
	@cd $(OUTPUT_DIR) 2>/dev/null && rm -f $(EXE) $$(ls $(EXE)_* 2>/dev/null | grep -v '\.cu$$' | grep -v '\.h$$' | grep -v '\.c$$') || true
	$(RM) $(KERNEL_LOCAL)
	$(RM) common/common.o
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
	@echo "LUD Multi-Variant Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all            - Build all variants (baseline + variants)"
	@echo "  clean          - Remove executables and temporary files"
	@echo "  list-variants  - Show detected variants"
	@echo "  help           - Show this help message"
	@echo ""
	@echo "Variant Detection:"
	@echo "  - Baseline: lud_kernel.cu"
	@echo "  - Variant format: lud_kernel_<model>_<trail>.cu"
	@echo ""
	@echo "Build Strategy:"
	@echo "  - Temporarily copy kernel file to main directory"
	@echo "  - Compile lud.cu with the kernel file"
	@echo "  - Clean up temporary files"
	@echo ""
	@echo "Output:"
	@echo "  - Executables are placed in $(OUTPUT_DIR)/"
	@echo "  - lud (baseline lud_kernel)"
	@echo "  - lud_<ID> (lud_kernel_<ID>)"
