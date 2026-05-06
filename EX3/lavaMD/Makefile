include ../common.mk

# Output directory for executables
OUTPUT_DIR = EX3_optimized_codes
BUILD_DIR = .build

EXE = lavaMD

# Main source file and other source files
MAIN_SRC = lavaMD.cpp
KERNEL_WRAPPER_SRC = kernel/kernel_gpu_cuda_wrapper.cu
OTHER_SRCS = util/num/num.c util/timer/timer.c util/device/device.cu

# Suppress warnings during compilation
NVCCFLAGS += -w

# Find kernel_gpu_cuda variants in EX3_optimized_codes
KERNEL_VARIANTS = $(wildcard $(OUTPUT_DIR)/kernel_gpu_cuda_*.cu)
KERNEL_IDS = $(patsubst $(OUTPUT_DIR)/kernel_gpu_cuda_%.cu,%,$(KERNEL_VARIANTS))

# Check if baseline exists
KERNEL_BASELINE_EXISTS = $(shell [ -f $(OUTPUT_DIR)/kernel_gpu_cuda.cu ] && echo yes)

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
$(OUTPUT_DIR)/$(EXE): $(MAIN_SRC) $(KERNEL_WRAPPER_SRC) $(OUTPUT_DIR)/kernel_gpu_cuda.cu $(OTHER_SRCS)
	@echo ""
	@echo "=========================================="
	@echo "Building: $@ (baseline)"
	@echo "=========================================="
	@cp $(OUTPUT_DIR)/kernel_gpu_cuda.cu kernel/kernel_gpu_cuda.cu
	-@$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(NVCC_LDLIBS) -o $@ $(MAIN_SRC) $(KERNEL_WRAPPER_SRC) $(OTHER_SRCS)
	@rm -f kernel/kernel_gpu_cuda.cu
	@if [ -f $@ ]; then echo "✓ Successfully built $@"; fi
endif

# Build variant versions (direct compilation, no .o files)
$(OUTPUT_DIR)/$(EXE)_%: $(MAIN_SRC) $(KERNEL_WRAPPER_SRC) $(OUTPUT_DIR)/kernel_gpu_cuda_%.cu $(OTHER_SRCS)
	@echo ""
	@echo "=========================================="
	@echo "Building: $@ (variant $*)"
	@echo "=========================================="
	@cp $(OUTPUT_DIR)/kernel_gpu_cuda_$*.cu kernel/kernel_gpu_cuda.cu
	-@$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(NVCC_LDLIBS) -o $@ $(MAIN_SRC) $(KERNEL_WRAPPER_SRC) $(OTHER_SRCS)
	@rm -f kernel/kernel_gpu_cuda.cu
	@if [ -f $@ ]; then echo "✓ Successfully built $@"; fi

.PHONY: clean
clean:
	@echo "Cleaning up..."
	@# Only remove executables (no extension), NOT .cu/.h/.c source files
	@cd $(OUTPUT_DIR) 2>/dev/null && rm -f $(EXE) $$(ls $(EXE)_* 2>/dev/null | grep -v '\.cu$$' | grep -v '\.h$$' | grep -v '\.c$$' | grep -v '\.cpp$$') || true
	$(RM) kernel/kernel_gpu_cuda.cu output.txt
	$(RM) -r $(BUILD_DIR)
	@echo "Clean complete."

.PHONY: list-variants
list-variants:
	@echo "Detected variants:"
	@echo "  kernel_gpu_cuda baseline: $(KERNEL_BASELINE_EXISTS)"
	@echo "  kernel_gpu_cuda variant IDs: $(KERNEL_IDS)"
	@echo ""
	@echo "Executables to build:"
	@for exe in $(EXECUTABLES); do echo "  - $$exe"; done

.PHONY: help
help:
	@echo "LavaMD Multi-Variant Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all            - Build all variants (baseline + variants)"
	@echo "  clean          - Remove executables and temporary files"
	@echo "  list-variants  - Show detected variants"
	@echo "  help           - Show this help message"
	@echo ""
	@echo "Variant Detection:"
	@echo "  - Baseline: kernel_gpu_cuda.cu"
	@echo "  - Variants: kernel_gpu_cuda_<model>_<trail>.cu"
	@echo ""
	@echo "Build Strategy:"
	@echo "  - Temporarily copy kernel file to kernel/ directory"
	@echo "  - Direct compilation (no intermediate .o files)"
	@echo "  - Compile lavaMD.cpp, kernel_wrapper, and utility files together"
	@echo "  - Clean up temporary files"
	@echo ""
	@echo "Output:"
	@echo "  - Executables are placed in $(OUTPUT_DIR)/"
