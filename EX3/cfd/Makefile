include ../common.mk

# Output directory for executables
OUTPUT_DIR = EX3_optimized_codes
BUILD_DIR = .build

# Only compile euler3d.cu (not pre_euler3d.cu)
# Output executables are named 'cfd' and 'cfd_*'

# Find euler3d kernel variants (source files)
EULER3D_VARIANTS = $(wildcard $(OUTPUT_DIR)/euler3d_*.cu)
EULER3D_IDS = $(patsubst $(OUTPUT_DIR)/euler3d_%.cu,%,$(EULER3D_VARIANTS))

# Check if baseline exists
EULER3D_BASELINE_EXISTS = $(shell [ -f $(OUTPUT_DIR)/euler3d.cu ] && echo yes)

# Generate executable names (use 'cfd' instead of 'euler3d')
EULER3D_EXES =

ifeq ($(EULER3D_BASELINE_EXISTS),yes)
    EULER3D_EXES += $(OUTPUT_DIR)/cfd
endif
EULER3D_EXES += $(addprefix $(OUTPUT_DIR)/cfd_,$(EULER3D_IDS))

EXECUTABLES = $(EULER3D_EXES)

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

# Build baseline (source: euler3d.cu, output: cfd)
ifeq ($(EULER3D_BASELINE_EXISTS),yes)
$(OUTPUT_DIR)/cfd: $(OUTPUT_DIR)/euler3d.cu
	@echo ""
	@echo "=========================================="
	@echo "Building: $@ (baseline)"
	@echo "=========================================="
	-@$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(NVCC_LDLIBS) -o $@ $<
	@if [ -f $@ ]; then echo "✓ Successfully built $@"; fi
endif

# Build variants (source: euler3d_*.cu, output: cfd_*)
$(OUTPUT_DIR)/cfd_%: $(OUTPUT_DIR)/euler3d_%.cu
	@echo ""
	@echo "=========================================="
	@echo "Building: $@ (variant $*)"
	@echo "=========================================="
	-@$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(NVCC_LDLIBS) -o $@ $<
	@if [ -f $@ ]; then echo "✓ Successfully built $@"; fi


# Clean target - CRITICAL: Only remove executables, NOT .cu source files
.PHONY: clean
clean:
	@echo "Cleaning up..."
	@# Only remove executables (no extension), NOT .cu/.h/.c source files
	@cd $(OUTPUT_DIR) 2>/dev/null && rm -f cfd $$(ls cfd_* 2>/dev/null | grep -v '\.cu$$' | grep -v '\.h$$' | grep -v '\.c$$' | grep -v '\.cpp$$') || true
	$(RM) density momentum density_energy
	$(RM) -r $(BUILD_DIR)
	@echo "Clean complete."

# List all detected variants
.PHONY: list-variants
list-variants:
	@echo "Detected variants:"
	@echo "  Source baseline (euler3d.cu): $(EULER3D_BASELINE_EXISTS)"
	@echo "  Source variant IDs: $(EULER3D_IDS)"
	@echo ""
	@echo "Executables to build:"
	@for exe in $(EXECUTABLES); do echo "  - $$exe"; done

.PHONY: help
help:
	@echo "CFD Multi-Variant Makefile (float precision)"
	@echo ""
	@echo "Targets:"
	@echo "  all            - Build all variants (baseline + variants)"
	@echo "  clean          - Remove executables and temporary files"
	@echo "  list-variants  - Show detected variants"
	@echo "  help           - Show this help message"
	@echo ""
	@echo "Program:"
	@echo "  cfd            - CFD solver (float precision)"
	@echo ""
	@echo "Variant Detection:"
	@echo "  - Source baseline: euler3d.cu -> Output: cfd"
	@echo "  - Source variants: euler3d_<model>_<trail>.cu -> Output: cfd_<model>_<trail>"
	@echo ""
	@echo "Output:"
	@echo "  - Executables are placed in $(OUTPUT_DIR)/"
