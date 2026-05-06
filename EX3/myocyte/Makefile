include ../common.mk

# Output directory for executables
OUTPUT_DIR = EX3_optimized_codes
BUILD_DIR = .build

EXE = myocyte

# Main source file (mode=0 only)
# Note: myocyte.cu includes define.c, file.c, timer.c via #include
MAIN_SRC = myocyte.cu

# Files to include (mode=0 version)
WORK_SRC = work.cu
MASTER_SRC = master.cu
SOLVER_SRC = solver.cu
EMBEDDED_SRC = embedded_fehlberg_7_8.cu
KERNEL_MAIN_SRC = kernel.cu

# Kernel files in main directory (temporary placeholders for includes)
KERNEL_CAM_LOCAL = kernel_cam.cu
KERNEL_ECC_LOCAL = kernel_ecc.cu
KERNEL_FIN_LOCAL = kernel_fin.cu

# Suppress warnings during compilation
NVCCFLAGS += -w

# Find kernel variants in EX3_optimized_codes
KERNEL_CAM_VARIANTS = $(wildcard $(OUTPUT_DIR)/kernel_cam_*.cu)
KERNEL_CAM_IDS = $(patsubst $(OUTPUT_DIR)/kernel_cam_%.cu,%,$(KERNEL_CAM_VARIANTS))

KERNEL_ECC_VARIANTS = $(wildcard $(OUTPUT_DIR)/kernel_ecc_*.cu)
KERNEL_ECC_IDS = $(patsubst $(OUTPUT_DIR)/kernel_ecc_%.cu,%,$(KERNEL_ECC_VARIANTS))

KERNEL_FIN_VARIANTS = $(wildcard $(OUTPUT_DIR)/kernel_fin_*.cu)
KERNEL_FIN_IDS = $(patsubst $(OUTPUT_DIR)/kernel_fin_%.cu,%,$(KERNEL_FIN_VARIANTS))

# Check if baselines exist
KERNEL_CAM_BASELINE_EXISTS = $(shell [ -f $(OUTPUT_DIR)/kernel_cam.cu ] && echo yes)
KERNEL_ECC_BASELINE_EXISTS = $(shell [ -f $(OUTPUT_DIR)/kernel_ecc.cu ] && echo yes)
KERNEL_FIN_BASELINE_EXISTS = $(shell [ -f $(OUTPUT_DIR)/kernel_fin.cu ] && echo yes)

# Find common variant IDs (all three kernels must have the same variant ID)
COMMON_IDS_TEMP = $(filter $(KERNEL_CAM_IDS),$(KERNEL_ECC_IDS))
COMMON_VARIANT_IDS = $(filter $(COMMON_IDS_TEMP),$(KERNEL_FIN_IDS))

# Generate executable names
EXECUTABLES =
ifeq ($(KERNEL_CAM_BASELINE_EXISTS),yes)
ifeq ($(KERNEL_ECC_BASELINE_EXISTS),yes)
ifeq ($(KERNEL_FIN_BASELINE_EXISTS),yes)
    EXECUTABLES += $(OUTPUT_DIR)/$(EXE)
endif
endif
endif
EXECUTABLES += $(addprefix $(OUTPUT_DIR)/$(EXE)_,$(COMMON_VARIANT_IDS))

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

# Build baseline version (all three baseline kernels)
ifeq ($(KERNEL_CAM_BASELINE_EXISTS),yes)
ifeq ($(KERNEL_ECC_BASELINE_EXISTS),yes)
ifeq ($(KERNEL_FIN_BASELINE_EXISTS),yes)
$(OUTPUT_DIR)/$(EXE): $(MAIN_SRC) $(OUTPUT_DIR)/kernel_cam.cu $(OUTPUT_DIR)/kernel_ecc.cu $(OUTPUT_DIR)/kernel_fin.cu $(WORK_SRC) $(MASTER_SRC) $(SOLVER_SRC) $(EMBEDDED_SRC) $(KERNEL_MAIN_SRC)
	@echo ""
	@echo "=========================================="
	@echo "Building: $@ (baseline)"
	@echo "=========================================="
	@cp $(OUTPUT_DIR)/kernel_cam.cu $(KERNEL_CAM_LOCAL)
	@cp $(OUTPUT_DIR)/kernel_ecc.cu $(KERNEL_ECC_LOCAL)
	@cp $(OUTPUT_DIR)/kernel_fin.cu $(KERNEL_FIN_LOCAL)
	-@$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(NVCC_LDLIBS) -lm -o $@ $(MAIN_SRC)
	@rm -f $(KERNEL_CAM_LOCAL) $(KERNEL_ECC_LOCAL) $(KERNEL_FIN_LOCAL)
	@if [ -f $@ ]; then echo "✓ Successfully built $@"; fi
endif
endif
endif

# Build variant versions (all three kernels must have the same variant ID)
$(OUTPUT_DIR)/$(EXE)_%: $(MAIN_SRC) $(OUTPUT_DIR)/kernel_cam_%.cu $(OUTPUT_DIR)/kernel_ecc_%.cu $(OUTPUT_DIR)/kernel_fin_%.cu $(WORK_SRC) $(MASTER_SRC) $(SOLVER_SRC) $(EMBEDDED_SRC) $(KERNEL_MAIN_SRC)
	@echo ""
	@echo "=========================================="
	@echo "Building: $@ (variant $*)"
	@echo "=========================================="
	@cp $(OUTPUT_DIR)/kernel_cam_$*.cu $(KERNEL_CAM_LOCAL)
	@cp $(OUTPUT_DIR)/kernel_ecc_$*.cu $(KERNEL_ECC_LOCAL)
	@cp $(OUTPUT_DIR)/kernel_fin_$*.cu $(KERNEL_FIN_LOCAL)
	-@$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(NVCC_LDLIBS) -lm -o $@ $(MAIN_SRC)
	@rm -f $(KERNEL_CAM_LOCAL) $(KERNEL_ECC_LOCAL) $(KERNEL_FIN_LOCAL)
	@if [ -f $@ ]; then echo "✓ Successfully built $@"; fi

.PHONY: clean
clean:
	@echo "Cleaning up..."
	@# Only remove executables (no extension), NOT .cu/.h/.c source files
	@cd $(OUTPUT_DIR) 2>/dev/null && rm -f $(EXE) $$(ls $(EXE)_* 2>/dev/null | grep -v '\.cu$$' | grep -v '\.h$$' | grep -v '\.c$$' | grep -v '\.cpp$$') || true
	$(RM) $(KERNEL_CAM_LOCAL) $(KERNEL_ECC_LOCAL) $(KERNEL_FIN_LOCAL)
	$(RM) -r $(BUILD_DIR)
	@echo "Clean complete."

.PHONY: list-variants
list-variants:
	@echo "Detected kernel files:"
	@echo "  kernel_cam baseline: $(KERNEL_CAM_BASELINE_EXISTS)"
	@echo "  kernel_cam variant IDs: $(KERNEL_CAM_IDS)"
	@echo "  kernel_ecc baseline: $(KERNEL_ECC_BASELINE_EXISTS)"
	@echo "  kernel_ecc variant IDs: $(KERNEL_ECC_IDS)"
	@echo "  kernel_fin baseline: $(KERNEL_FIN_BASELINE_EXISTS)"
	@echo "  kernel_fin variant IDs: $(KERNEL_FIN_IDS)"
	@echo ""
	@echo "Common paired variant IDs: $(COMMON_VARIANT_IDS)"
	@echo ""
	@echo "Executables to build:"
	@for exe in $(EXECUTABLES); do echo "  - $$exe"; done

.PHONY: help
help:
	@echo "Myocyte Multi-Variant Makefile (mode=0 only)"
	@echo ""
	@echo "Targets:"
	@echo "  all            - Build all variants (baseline + variants)"
	@echo "  clean          - Remove executables and temporary files"
	@echo "  list-variants  - Show detected variants"
	@echo "  help           - Show this help message"
	@echo ""
	@echo "Variant Detection:"
	@echo "  - MUST have ALL THREE: kernel_cam_<ID>.cu, kernel_ecc_<ID>.cu, kernel_fin_<ID>.cu"
	@echo "  - Baseline: kernel_cam.cu + kernel_ecc.cu + kernel_fin.cu"
	@echo "  - Variant format: kernel_cam_<model>_<trail>.cu + kernel_ecc_<model>_<trail>.cu + kernel_fin_<model>_<trail>.cu"
	@echo ""
	@echo "Build Strategy:"
	@echo "  - Temporarily copy three kernel files to main directory"
	@echo "  - Direct compilation (no intermediate .o files)"
	@echo "  - Compile myocyte.cu (which includes work.cu, which includes the kernels)"
	@echo "  - Clean up temporary files"
	@echo ""
	@echo "Output:"
	@echo "  - Executables are placed in $(OUTPUT_DIR)/"
	@echo "  - myocyte (baseline kernel_cam + kernel_ecc + kernel_fin)"
	@echo "  - myocyte_<ID> (kernel_cam_<ID> + kernel_ecc_<ID> + kernel_fin_<ID>)"
