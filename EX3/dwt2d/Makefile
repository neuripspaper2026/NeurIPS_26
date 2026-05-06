include ../common.mk

# Output directory for executables
OUTPUT_DIR = EX3_optimized_codes
BUILD_DIR = .build

EXE = dwt2d

# Source files (excluding fdwt97.cu which will be handled separately)
COMMON_SRCS = dwt.cu components.cu dwt_cuda/common.cu dwt_cuda/fdwt53.cu dwt_cuda/rdwt53.cu dwt_cuda/rdwt97.cu

# Add OUTPUT flag to enable file writing
CXXFLAGS += -DOUTPUT
NVCCFLAGS += -DOUTPUT

# Add include paths for dwt_cuda headers
CPPFLAGS += -I. -I./dwt_cuda

# Suppress common warnings
NVCCFLAGS += -w

# Find fdwt97 kernel variants in EX3_optimized_codes/
FDWT97_VARIANTS = $(wildcard $(OUTPUT_DIR)/fdwt97_*.cu)
FDWT97_IDS = $(patsubst $(OUTPUT_DIR)/fdwt97_%.cu,%,$(FDWT97_VARIANTS))

# Check if baseline exists
FDWT97_BASELINE_EXISTS = $(shell [ -f $(OUTPUT_DIR)/fdwt97.cu ] && echo yes)

# Generate executable names
EXECUTABLES =
ifeq ($(FDWT97_BASELINE_EXISTS),yes)
    EXECUTABLES += $(OUTPUT_DIR)/$(EXE)
endif
EXECUTABLES += $(addprefix $(OUTPUT_DIR)/$(EXE)_,$(FDWT97_IDS))

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
ifeq ($(FDWT97_BASELINE_EXISTS),yes)
$(OUTPUT_DIR)/$(EXE): dwt2d.cu $(COMMON_SRCS) $(OUTPUT_DIR)/fdwt97.cu
	@echo ""
	@echo "=========================================="
	@echo "Building: $@ (baseline)"
	@echo "=========================================="
	-@$(NVCC) $(CPPFLAGS) $(CXXFLAGS) $(NVCCFLAGS) $(NVCC_LDLIBS) -o $@ dwt2d.cu $(COMMON_SRCS) $(OUTPUT_DIR)/fdwt97.cu
	@if [ -f $@ ]; then echo "✓ Successfully built $@"; fi
endif

# Build variant versions
$(OUTPUT_DIR)/$(EXE)_%: dwt2d.cu $(COMMON_SRCS) $(OUTPUT_DIR)/fdwt97_%.cu
	@echo ""
	@echo "=========================================="
	@echo "Building: $@ (variant $*)"
	@echo "=========================================="
	-@$(NVCC) $(CPPFLAGS) $(CXXFLAGS) $(NVCCFLAGS) $(NVCC_LDLIBS) -o $@ dwt2d.cu $(COMMON_SRCS) $(OUTPUT_DIR)/fdwt97_$*.cu
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
	@echo "  fdwt97 baseline: $(FDWT97_BASELINE_EXISTS)"
	@echo "  fdwt97 variant IDs: $(FDWT97_IDS)"
	@echo ""
	@echo "Executables to build:"
	@for exe in $(EXECUTABLES); do echo "  - $$exe"; done

.PHONY: help
help:
	@echo "DWT2D Multi-Variant Makefile (fdwt97 only)"
	@echo ""
	@echo "Targets:"
	@echo "  all            - Build all variants (baseline + variants)"
	@echo "  clean          - Remove executables and temporary files"
	@echo "  list-variants  - Show detected variants"
	@echo "  help           - Show this help message"
	@echo ""
	@echo "Variant Detection:"
	@echo "  - fdwt97 baseline: EX3_optimized_codes/fdwt97.cu"
	@echo "  - fdwt97 variants: EX3_optimized_codes/fdwt97_<model>_<trail>.cu"
	@echo ""
	@echo "Output:"
	@echo "  - Executables are placed in $(OUTPUT_DIR)/"
	@echo ""
	@echo "Note: Only forward 9/7 DWT (fdwt97) is compiled."
