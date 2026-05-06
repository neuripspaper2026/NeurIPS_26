#!/usr/bin/make -f

# Makefile for 2mm benchmark
# Automatically detects compiler and builds all variants with different dataset sizes

# ============================================================================
# Configuration
# ============================================================================

# Detect compiler
CC := $(shell command -v gcc 2>/dev/null)
ifdef CC
    COMPILER_TAG := gcc
else
    CC := $(shell command -v clang 2>/dev/null)
    ifdef CC
        COMPILER_TAG := clang
    else
        $(error No compiler found. Please install gcc or clang)
    endif
endif

# Compiler flags
CFLAGS := -O3
INCLUDES := -I../../utilities -I.
LIBS := -lm
POLYBENCH_SRC := ../../utilities/polybench.c

# Dataset size macros
DATASET_FLAGS := MINI_DATASET SMALL_DATASET MEDIUM_DATASET LARGE_DATASET EXTRA_LARGE_DATASET

# Directories
SRC_DIR := EX1_optimized_codes
CORRECTNESS_DIR := EX5_correctness

# ============================================================================
# Source file detection
# ============================================================================

# Find all .c source files in EX1_optimized_codes
ALL_SOURCES := $(wildcard $(SRC_DIR)/*.c)

# Extract base names (without .c extension and without directory)
# Exclude 2mm_parameterized as it's handled separately
SOURCE_BASES := $(filter-out 2mm_parameterized,$(basename $(notdir $(ALL_SOURCES))))

# ============================================================================
# Binary generation
# ============================================================================

# Generate all binary names: <basename>_<compiler>_<size>
# For each source base, create 5 binaries (one per dataset size)
define make_binary_names
$(foreach size,mini small medium large extra-large,\
  $(SRC_DIR)/$(1)_$(COMPILER_TAG)_$(size))
endef

ALL_BINS := $(foreach base,$(SOURCE_BASES),$(call make_binary_names,$(base)))

# ============================================================================
# Phony targets
# ============================================================================

.PHONY: all baseline variants clean help test-mini test-small test-medium test-large test-extra-large

# Default target - build all variants with all dataset sizes
all: variants

# ============================================================================
# Main build targets
# ============================================================================


# Build baseline only (2mm.c) - legacy support for fixed-size binaries
baseline: $(call make_binary_names,2mm)
	@echo "✓ Baseline binaries built successfully with $(COMPILER_TAG)"

# Build all variant binaries
variants: baseline
	@echo "Building all variant binaries with $(COMPILER_TAG)..."
	@failed=0; \
	total=0; \
	for base in $(SOURCE_BASES); do \
		if [ "$$base" != "2mm" ]; then \
			total=$$((total+1)); \
		fi; \
	done; \
	current=0; \
	for base in $(SOURCE_BASES); do \
		if [ "$$base" != "2mm" ]; then \
			current=$$((current+1)); \
			echo "[$$current/$$total] Building $$base..."; \
			variant_failed=0; \
			for size in mini small medium large extra-large; do \
				if ! $(MAKE) --no-print-directory $(SRC_DIR)/$${base}_$(COMPILER_TAG)_$$size 2>/dev/null; then \
					variant_failed=1; \
					break; \
				fi; \
			done; \
			if [ $$variant_failed -eq 1 ]; then \
				failed=$$((failed+1)); \
				echo "  ✗ Failed to build $$base (continuing...)"; \
			fi; \
		fi; \
	done; \
	echo ""; \
	if [ $$failed -eq 0 ]; then \
		echo "✓ All variant binaries built successfully"; \
	else \
		echo "⚠ $$failed variant(s) failed to compile (continued with others)"; \
		echo "  Successfully built: $$((total - failed))/$$total variants"; \
	fi

# ============================================================================
# Pattern rules for building binaries
# ============================================================================

# Pattern rule for mini dataset
$(SRC_DIR)/%_$(COMPILER_TAG)_mini: $(SRC_DIR)/%.c $(POLYBENCH_SRC)
	@echo "  Compiling $* (mini)..."
	@$(CC) -DMINI_DATASET $(CFLAGS) $(INCLUDES) $< $(POLYBENCH_SRC) $(LIBS) -o $@ \
		|| (echo "Error: Failed to build $@"; exit 1)

# Pattern rule for small dataset
$(SRC_DIR)/%_$(COMPILER_TAG)_small: $(SRC_DIR)/%.c $(POLYBENCH_SRC)
	@echo "  Compiling $* (small)..."
	@$(CC) -DSMALL_DATASET $(CFLAGS) $(INCLUDES) $< $(POLYBENCH_SRC) $(LIBS) -o $@ \
		|| (echo "Error: Failed to build $@"; exit 1)

# Pattern rule for medium dataset
$(SRC_DIR)/%_$(COMPILER_TAG)_medium: $(SRC_DIR)/%.c $(POLYBENCH_SRC)
	@echo "  Compiling $* (medium)..."
	@$(CC) -DMEDIUM_DATASET $(CFLAGS) $(INCLUDES) $< $(POLYBENCH_SRC) $(LIBS) -o $@ \
		|| (echo "Error: Failed to build $@"; exit 1)

# Pattern rule for large dataset
$(SRC_DIR)/%_$(COMPILER_TAG)_large: $(SRC_DIR)/%.c $(POLYBENCH_SRC)
	@echo "  Compiling $* (large)..."
	@$(CC) -DLARGE_DATASET $(CFLAGS) $(INCLUDES) $< $(POLYBENCH_SRC) $(LIBS) -o $@ \
		|| (echo "Error: Failed to build $@"; exit 1)

# Pattern rule for extra-large dataset
$(SRC_DIR)/%_$(COMPILER_TAG)_extra-large: $(SRC_DIR)/%.c $(POLYBENCH_SRC)
	@echo "  Compiling $* (extra-large)..."
	@$(CC) -DEXTRA_LARGE_DATASET $(CFLAGS) $(INCLUDES) $< $(POLYBENCH_SRC) $(LIBS) -o $@ \
		|| (echo "Error: Failed to build $@"; exit 1)

# ============================================================================
# Testing targets
# ============================================================================

test-mini: baseline
	@echo "Testing mini dataset..."
	@./$(SRC_DIR)/2mm_$(COMPILER_TAG)_mini > /dev/null 2>&1 && echo "✓ Mini test passed"

test-small: baseline
	@echo "Testing small dataset..."
	@./$(SRC_DIR)/2mm_$(COMPILER_TAG)_small > /dev/null 2>&1 && echo "✓ Small test passed"

test-medium: baseline
	@echo "Testing medium dataset..."
	@./$(SRC_DIR)/2mm_$(COMPILER_TAG)_medium > /dev/null 2>&1 && echo "✓ Medium test passed"

test-large: baseline
	@echo "Testing large dataset..."
	@./$(SRC_DIR)/2mm_$(COMPILER_TAG)_large > /dev/null 2>&1 && echo "✓ Large test passed"

test-extra-large: baseline
	@echo "Testing extra-large dataset..."
	@./$(SRC_DIR)/2mm_$(COMPILER_TAG)_extra-large > /dev/null 2>&1 && echo "✓ Extra-large test passed"

test-all: test-mini test-small test-medium test-large test-extra-large
	@echo "✓ All tests passed"

# ============================================================================
# Cleanup
# ============================================================================

clean:
	@echo "Cleaning up..."
	@rm -f $(SRC_DIR)/*_gcc_mini $(SRC_DIR)/*_gcc_small $(SRC_DIR)/*_gcc_medium \
		$(SRC_DIR)/*_gcc_large $(SRC_DIR)/*_gcc_extra-large
	@rm -f $(SRC_DIR)/*_clang_mini $(SRC_DIR)/*_clang_small $(SRC_DIR)/*_clang_medium \
		$(SRC_DIR)/*_clang_large $(SRC_DIR)/*_clang_extra-large
	@rm -f $(CORRECTNESS_DIR)/output_*.data
	@echo "✓ Cleanup complete"

# ============================================================================
# Help
# ============================================================================

help:
	@echo "2mm Benchmark Makefile"
	@echo "====================="
	@echo ""
	@echo "Detected compiler: $(COMPILER_TAG)"
	@echo ""
	@echo "Available targets:"
	@echo "  make              - Build all variants with all dataset sizes"
	@echo "  make all          - Same as 'make'"
	@echo "  make variants     - Build all variant binaries (all sizes)"
	@echo "  make baseline     - Build only baseline (2mm.c) binaries"
	@echo "  make test-mini    - Run baseline with mini dataset"
	@echo "  make test-small   - Run baseline with small dataset"
	@echo "  make test-medium  - Run baseline with medium dataset"
	@echo "  make test-large   - Run baseline with large dataset"
	@echo "  make test-extra-large - Run baseline with extra-large dataset"
	@echo "  make test-all     - Run all tests"
	@echo "  make clean        - Remove all generated binaries"
	@echo "  make help         - Show this help message"
	@echo ""
	@echo "Dataset sizes:"
	@echo "  mini, small, medium, large, extra-large"
	@echo ""
	@echo "Binary naming convention:"
	@echo "  <source_base>_$(COMPILER_TAG)_<size>"
	@echo ""
	@echo "Examples:"
	@echo "  ./$(SRC_DIR)/2mm_$(COMPILER_TAG)_mini              # Run baseline mini"
	@echo "  ./$(SRC_DIR)/2mm_$(COMPILER_TAG)_large             # Run baseline large"
	@echo "  ./$(SRC_DIR)/2mm_claude_v1_$(COMPILER_TAG)_mini    # Run variant mini"
	@echo "  ./$(SRC_DIR)/2mm_llama4_v2_$(COMPILER_TAG)_large   # Run variant large"
	@echo ""
	@echo "Source files found: $(words $(SOURCE_BASES))"
	@echo "  Baseline: 2mm.c"
	@echo "  Variants: $(words $(filter-out 2mm,$(SOURCE_BASES)))"
	@echo ""
	@echo "Total binaries to build: $(words $(SOURCE_BASES)) sources × 5 sizes = $(shell echo $$(($(words $(SOURCE_BASES)) * 5))) binaries"

