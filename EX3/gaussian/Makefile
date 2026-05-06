include ../common.mk

# Output directory for executables
OUTPUT_DIR = EX3_optimized_codes
BUILD_DIR = .build

EXE = gaussian

# Find gaussian kernel variants
GAUSSIAN_VARIANTS = $(wildcard $(OUTPUT_DIR)/gaussian_*.cu)
GAUSSIAN_IDS = $(patsubst $(OUTPUT_DIR)/gaussian_%.cu,%,$(GAUSSIAN_VARIANTS))

# Check if baseline exists
GAUSSIAN_BASELINE_EXISTS = $(shell [ -f $(OUTPUT_DIR)/gaussian.cu ] && echo yes)

# Generate executable names
EXECUTABLES =
ifeq ($(GAUSSIAN_BASELINE_EXISTS),yes)
    EXECUTABLES += $(OUTPUT_DIR)/$(EXE)
endif
EXECUTABLES += $(addprefix $(OUTPUT_DIR)/$(EXE)_,$(GAUSSIAN_IDS))

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

# Build baseline version
ifeq ($(GAUSSIAN_BASELINE_EXISTS),yes)
$(OUTPUT_DIR)/$(EXE): $(OUTPUT_DIR)/gaussian.cu
	@echo ""
	@echo "=========================================="
	@echo "Building: $@ (baseline)"
	@echo "=========================================="
	-@$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(NVCC_LDLIBS) -o $@ $<
	@if [ -f $@ ]; then echo "✓ Successfully built $@"; fi
endif

# Build variant versions
$(OUTPUT_DIR)/$(EXE)_%: $(OUTPUT_DIR)/gaussian_%.cu
	@echo ""
	@echo "=========================================="
	@echo "Building: $@ (variant $*)"
	@echo "=========================================="
	-@$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(NVCC_LDLIBS) -o $@ $<
	@if [ -f $@ ]; then echo "✓ Successfully built $@"; fi

.PHONY: clean
clean:
	@echo "Cleaning up..."
	@# Only remove executables (no extension), NOT .cu/.h/.c source files
	@cd $(OUTPUT_DIR) 2>/dev/null && rm -f $(EXE) $$(ls $(EXE)_* 2>/dev/null | grep -v '\.cu$$' | grep -v '\.h$$' | grep -v '\.c$$' | grep -v '\.cpp$$') || true
	$(RM) -r $(BUILD_DIR)
	@echo "Clean complete."

.PHONY: list-variants
list-variants:
	@echo "Detected variants:"
	@echo "  gaussian baseline: $(GAUSSIAN_BASELINE_EXISTS)"
	@echo "  gaussian variant IDs: $(GAUSSIAN_IDS)"
	@echo ""
	@echo "Executables to build:"
	@for exe in $(EXECUTABLES); do echo "  - $$exe"; done

.PHONY: help
help:
	@echo "Gaussian Multi-Variant Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all            - Build all variants (baseline + variants)"
	@echo "  clean          - Remove executables and temporary files"
	@echo "  list-variants  - Show detected variants"
	@echo "  help           - Show this help message"
	@echo ""
	@echo "Variant Detection:"
	@echo "  - gaussian baseline: gaussian.cu"
	@echo "  - gaussian variants: gaussian_<model>_<trail>.cu"
	@echo ""
	@echo "Output:"
	@echo "  - Executables are placed in $(OUTPUT_DIR)/"
