include ../common.mk

# Output directory for executables
OUTPUT_DIR = EX3_optimized_codes
BUILD_DIR = .build

EXE = heartwall

# Main source file
MAIN_SRC = heartwall.cu

# AVI library dependencies
AVI_DIR = ../../common_rodinia/avi
OBJS = $(AVI_DIR)/avilib.o $(AVI_DIR)/avimod.o

# Find kernel variants
KERNEL_VARIANTS = $(wildcard $(OUTPUT_DIR)/kernel_*.cu)
KERNEL_IDS = $(patsubst $(OUTPUT_DIR)/kernel_%.cu,%,$(KERNEL_VARIANTS))

# Check if baseline exists
KERNEL_BASELINE_EXISTS = $(shell [ -f $(OUTPUT_DIR)/kernel.cu ] && echo yes)

# Generate executable names
EXECUTABLES =
ifeq ($(KERNEL_BASELINE_EXISTS),yes)
    EXECUTABLES += $(OUTPUT_DIR)/$(EXE)
endif
EXECUTABLES += $(addprefix $(OUTPUT_DIR)/$(EXE)_,$(KERNEL_IDS))

# Include AVI headers
CPPFLAGS += -I$(AVI_DIR)

# Export variables for recursive make
export

# Ensure output directory exists
$(shell mkdir -p $(OUTPUT_DIR) $(BUILD_DIR))

.PHONY: all
all: $(OBJS) $(EXECUTABLES)
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

# Build AVI library objects
$(AVI_DIR)/avilib.o $(AVI_DIR)/avimod.o:
	$(MAKE) -C $(AVI_DIR)

# Build baseline version
ifeq ($(KERNEL_BASELINE_EXISTS),yes)
$(OUTPUT_DIR)/$(EXE): $(MAIN_SRC) $(OUTPUT_DIR)/kernel.cu $(OBJS)
	@echo ""
	@echo "=========================================="
	@echo "Building: $@ (baseline)"
	@echo "=========================================="
	@cp $(OUTPUT_DIR)/kernel.cu kernel.cu
	-@$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(NVCC_LDLIBS) -o $@ $(MAIN_SRC) $(OBJS)
	@rm -f kernel.cu
	@if [ -f $@ ]; then echo "✓ Successfully built $@"; fi
endif

# Build variant versions
$(OUTPUT_DIR)/$(EXE)_%: $(MAIN_SRC) $(OUTPUT_DIR)/kernel_%.cu $(OBJS)
	@echo ""
	@echo "=========================================="
	@echo "Building: $@ (variant $*)"
	@echo "=========================================="
	@cp $(OUTPUT_DIR)/kernel_$*.cu kernel.cu
	-@$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(NVCC_LDLIBS) -o $@ $(MAIN_SRC) $(OBJS)
	@rm -f kernel.cu
	@if [ -f $@ ]; then echo "✓ Successfully built $@"; fi

.PHONY: clean
clean:
	@echo "Cleaning up..."
	@# Only remove executables (no extension), NOT .cu/.h/.c source files
	@cd $(OUTPUT_DIR) 2>/dev/null && rm -f $(EXE) $$(ls $(EXE)_* 2>/dev/null | grep -v '\.cu$$' | grep -v '\.h$$' | grep -v '\.c$$' | grep -v '\.cpp$$') || true
	$(RM) output.txt
	$(RM) -r $(BUILD_DIR)
	$(MAKE) -C $(AVI_DIR) clean
	@echo "Clean complete."

.PHONY: list-variants
list-variants:
	@echo "Detected variants:"
	@echo "  kernel baseline: $(KERNEL_BASELINE_EXISTS)"
	@echo "  kernel variant IDs: $(KERNEL_IDS)"
	@echo ""
	@echo "Executables to build:"
	@for exe in $(EXECUTABLES); do echo "  - $$exe"; done

.PHONY: help
help:
	@echo "Heartwall Multi-Variant Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all            - Build all variants (baseline + variants)"
	@echo "  clean          - Remove executables and temporary files"
	@echo "  list-variants  - Show detected variants"
	@echo "  help           - Show this help message"
	@echo ""
	@echo "Variant Detection:"
	@echo "  - kernel baseline: kernel.cu"
	@echo "  - kernel variants: kernel_<model>_<trail>.cu"
	@echo ""
	@echo "Output:"
	@echo "  - Executables are placed in $(OUTPUT_DIR)/"
