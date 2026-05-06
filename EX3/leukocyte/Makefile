include ../common.mk

# Output directory for executables
OUTPUT_DIR = EX3_optimized_codes
BUILD_DIR = .build

EXE = leukocyte

# Main source file and other source files
MAIN_SRC = leukocyte.cpp
FIND_ELLIPSE_KERNEL_SRC = find_ellipse_kernel.cu
TRACK_ELLIPSE_KERNEL_SRC = track_ellipse_kernel.cu
OTHER_SRCS = find_ellipse.c track_ellipse.c misc_math.c

# External library directories
MESCHACH_DIR = ../../common_rodinia/meschach
AVI_DIR = ../../common_rodinia/avi

# Suppress warnings during compilation
NVCCFLAGS += -w
CPPFLAGS += -I$(MESCHACH_DIR) -I$(AVI_DIR)
LDLIBS += -lm -lstdc++

# Find kernel variants in EX3_optimized_codes
FIND_ELLIPSE_VARIANTS = $(wildcard $(OUTPUT_DIR)/find_ellipse_kernel_*.cu)
FIND_ELLIPSE_IDS = $(patsubst $(OUTPUT_DIR)/find_ellipse_kernel_%.cu,%,$(FIND_ELLIPSE_VARIANTS))

TRACK_ELLIPSE_VARIANTS = $(wildcard $(OUTPUT_DIR)/track_ellipse_kernel_*.cu)
TRACK_ELLIPSE_IDS = $(patsubst $(OUTPUT_DIR)/track_ellipse_kernel_%.cu,%,$(TRACK_ELLIPSE_VARIANTS))

# Check if baselines exist
FIND_ELLIPSE_BASELINE_EXISTS = $(shell [ -f $(OUTPUT_DIR)/find_ellipse_kernel.cu ] && echo yes)
TRACK_ELLIPSE_BASELINE_EXISTS = $(shell [ -f $(OUTPUT_DIR)/track_ellipse_kernel.cu ] && echo yes)

# Find common variant IDs (both kernels must have the same variant ID)
COMMON_VARIANT_IDS = $(filter $(FIND_ELLIPSE_IDS),$(TRACK_ELLIPSE_IDS))

# Generate executable names
EXECUTABLES =
ifeq ($(FIND_ELLIPSE_BASELINE_EXISTS),yes)
ifeq ($(TRACK_ELLIPSE_BASELINE_EXISTS),yes)
    EXECUTABLES += $(OUTPUT_DIR)/$(EXE)
endif
endif
EXECUTABLES += $(addprefix $(OUTPUT_DIR)/$(EXE)_,$(COMMON_VARIANT_IDS))

# Ensure output directory exists and external libraries are built
$(shell mkdir -p $(OUTPUT_DIR) $(BUILD_DIR))

.PHONY: all
all: $(MESCHACH_DIR)/libmeschach.a $(AVI_DIR)/avilib.o $(EXECUTABLES)
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

# Build external libraries
export
$(MESCHACH_DIR)/makefile:
	cd $(MESCHACH_DIR); ./configure --with-all
$(MESCHACH_DIR)/libmeschach.a: $(MESCHACH_DIR)/makefile
	$(MAKE) -C $(MESCHACH_DIR)

$(AVI_DIR)/avilib.o:
	$(MAKE) -C $(AVI_DIR)

# Build baseline version (baseline find_ellipse_kernel + baseline track_ellipse_kernel)
ifeq ($(FIND_ELLIPSE_BASELINE_EXISTS),yes)
ifeq ($(TRACK_ELLIPSE_BASELINE_EXISTS),yes)
$(OUTPUT_DIR)/$(EXE): $(MAIN_SRC) $(OUTPUT_DIR)/find_ellipse_kernel.cu $(OUTPUT_DIR)/track_ellipse_kernel.cu $(OTHER_SRCS) $(MESCHACH_DIR)/libmeschach.a $(AVI_DIR)/avilib.o
	@echo ""
	@echo "=========================================="
	@echo "Building: $@ (baseline)"
	@echo "=========================================="
	@cp $(OUTPUT_DIR)/find_ellipse_kernel.cu $(FIND_ELLIPSE_KERNEL_SRC)
	@cp $(OUTPUT_DIR)/track_ellipse_kernel.cu $(TRACK_ELLIPSE_KERNEL_SRC)
	-@$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(NVCC_LDLIBS) $(LDLIBS) -o $@ $(MAIN_SRC) $(FIND_ELLIPSE_KERNEL_SRC) $(TRACK_ELLIPSE_KERNEL_SRC) $(OTHER_SRCS) $(MESCHACH_DIR)/libmeschach.a $(AVI_DIR)/avilib.o
	@rm -f $(FIND_ELLIPSE_KERNEL_SRC) $(TRACK_ELLIPSE_KERNEL_SRC)
	@rm -f find_ellipse.o find_ellipse_kernel.o track_ellipse.o track_ellipse_kernel.o misc_math.o
	@if [ -f $@ ]; then echo "✓ Successfully built $@"; fi
endif
endif

# Build variant versions (paired find_ellipse_kernel_ID + track_ellipse_kernel_ID)
$(OUTPUT_DIR)/$(EXE)_%: $(MAIN_SRC) $(OUTPUT_DIR)/find_ellipse_kernel_%.cu $(OUTPUT_DIR)/track_ellipse_kernel_%.cu $(OTHER_SRCS) $(MESCHACH_DIR)/libmeschach.a $(AVI_DIR)/avilib.o
	@echo ""
	@echo "=========================================="
	@echo "Building: $@ (variant $*)"
	@echo "=========================================="
	@cp $(OUTPUT_DIR)/find_ellipse_kernel_$*.cu $(FIND_ELLIPSE_KERNEL_SRC)
	@cp $(OUTPUT_DIR)/track_ellipse_kernel_$*.cu $(TRACK_ELLIPSE_KERNEL_SRC)
	-@$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(NVCC_LDLIBS) $(LDLIBS) -o $@ $(MAIN_SRC) $(FIND_ELLIPSE_KERNEL_SRC) $(TRACK_ELLIPSE_KERNEL_SRC) $(OTHER_SRCS) $(MESCHACH_DIR)/libmeschach.a $(AVI_DIR)/avilib.o
	@rm -f $(FIND_ELLIPSE_KERNEL_SRC) $(TRACK_ELLIPSE_KERNEL_SRC)
	@rm -f find_ellipse.o find_ellipse_kernel.o track_ellipse.o track_ellipse_kernel.o misc_math.o
	@if [ -f $@ ]; then echo "✓ Successfully built $@"; fi

.PHONY: clean
clean:
	@echo "Cleaning up..."
	@# Only remove executables (no extension), NOT .cu/.h/.c source files
	@cd $(OUTPUT_DIR) 2>/dev/null && rm -f $(EXE) $$(ls $(EXE)_* 2>/dev/null | grep -v '\.cu$$' | grep -v '\.h$$' | grep -v '\.c$$' | grep -v '\.cpp$$') || true
	$(RM) $(FIND_ELLIPSE_KERNEL_SRC) $(TRACK_ELLIPSE_KERNEL_SRC)
	$(RM) find_ellipse.o find_ellipse_kernel.o track_ellipse.o track_ellipse_kernel.o misc_math.o
	-$(MAKE) -C $(MESCHACH_DIR) realclean
	$(MAKE) -C $(AVI_DIR) clean
	$(RM) -r $(BUILD_DIR)
	@echo "Clean complete."

.PHONY: list-variants
list-variants:
	@echo "Detected kernel files:"
	@echo "  find_ellipse_kernel baseline: $(FIND_ELLIPSE_BASELINE_EXISTS)"
	@echo "  find_ellipse_kernel variant IDs: $(FIND_ELLIPSE_IDS)"
	@echo "  track_ellipse_kernel baseline: $(TRACK_ELLIPSE_BASELINE_EXISTS)"
	@echo "  track_ellipse_kernel variant IDs: $(TRACK_ELLIPSE_IDS)"
	@echo ""
	@echo "Common paired variant IDs: $(COMMON_VARIANT_IDS)"
	@echo ""
	@echo "Executables to build:"
	@for exe in $(EXECUTABLES); do echo "  - $$exe"; done

.PHONY: help
help:
	@echo "Leukocyte Multi-Variant Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all            - Build all variants (baseline + variants)"
	@echo "  clean          - Remove executables and temporary files"
	@echo "  list-variants  - Show detected variants"
	@echo "  help           - Show this help message"
	@echo ""
	@echo "Variant Detection:"
	@echo "  - MUST have BOTH find_ellipse_kernel_<ID>.cu AND track_ellipse_kernel_<ID>.cu"
	@echo "  - Baseline: find_ellipse_kernel.cu + track_ellipse_kernel.cu"
	@echo "  - Variant format: find_ellipse_kernel_<model>_<trail>.cu + track_ellipse_kernel_<model>_<trail>.cu"
	@echo ""
	@echo "Build Strategy:"
	@echo "  - Build external libraries (meschach, avi)"
	@echo "  - Temporarily copy paired kernel files to main directory"
	@echo "  - Direct compilation (no intermediate .o files)"
	@echo "  - Compile leukocyte.cpp, kernel files, and other sources together"
	@echo "  - Clean up temporary files"
	@echo ""
	@echo "Output:"
	@echo "  - Executables are placed in $(OUTPUT_DIR)/"
	@echo "  - leukocyte (baseline find_ellipse_kernel + baseline track_ellipse_kernel)"
	@echo "  - leukocyte_<ID> (find_ellipse_kernel_<ID> + track_ellipse_kernel_<ID>)"
