#!/usr/bin/make -f

# ============================================================================
# Correlation Benchmark Makefile
# ============================================================================
# This Makefile compiles the correlation benchmark with multiple dataset sizes.
# Each source file is compiled for each dataset size into a separate binary.
# Binary naming: <source_base>_<compiler>_<size>
# Compilation strategy: Serial (no OpenMP)
# ============================================================================

# ============================================================================
# Compiler Detection
# ============================================================================

# Auto-detect available compiler (prefer gcc, fallback to clang)
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

# ============================================================================
# Compilation Flags (Serial Compilation)
# ============================================================================

# Basic compilation options (serial, no OpenMP)
CFLAGS := -O3
INCLUDES := -I../../utilities -I.
LIBS := -lm

# PolyBench framework source
POLYBENCH_SRC := ../../utilities/polybench.c

# Note: Do not add -fopenmp for serial execution
# Note: Do not add -lpthread, no multi-threading support needed

# ============================================================================
# Dataset Size Macros
# ============================================================================

# Define 5 standard dataset sizes
DATASET_FLAGS := MINI_DATASET SMALL_DATASET MEDIUM_DATASET LARGE_DATASET EXTRALARGE_DATASET

# ============================================================================
# Directories and Source Files
# ============================================================================

SRC_DIR := EX1_optimized_codes
CORRECTNESS_DIR := EX5_correctness

# Auto-detect all .c source files
ALL_SOURCES := $(wildcard $(SRC_DIR)/*.c)

# Extract base names (remove path and extension)
SOURCE_BASES := $(basename $(notdir $(ALL_SOURCES)))

# ============================================================================
# Binary Name Generation Function
# ============================================================================

# Generate all size binary names for a given source file base name
# Usage: $(call make_binary_names,correlation)
# Returns: EX1_optimized_codes/correlation_gcc_mini EX1_optimized_codes/correlation_gcc_small ...
define make_binary_names
$(foreach size,mini small medium large extra-large,\
  $(SRC_DIR)/$(1)_$(COMPILER_TAG)_$(size))
endef

# Generate all binary file list
ALL_BINS := $(foreach base,$(SOURCE_BASES),$(call make_binary_names,$(base)))

# ============================================================================
# Phony Targets
# ============================================================================

.PHONY: all baseline variants clean help test-mini test-small test-medium test-large test-extra-large

# Default target: compile all variants and all sizes
all: variants

# ============================================================================
# Main Compilation Targets
# ============================================================================

# Only compile baseline (first source file, usually unoptimized version)
baseline: $(call make_binary_names,correlation)
	@echo "✓ Baseline binaries built successfully with $(COMPILER_TAG)"

# Compile all variants (including baseline)
variants: baseline
	@echo "Building all variant binaries with $(COMPILER_TAG)..."
	@failed=0; \
	total=0; \
	for base in $(SOURCE_BASES); do \
		if [ "$$base" != "correlation" ]; then \
			total=$$((total+1)); \
		fi; \
	done; \
	current=0; \
	for base in $(SOURCE_BASES); do \
		if [ "$$base" != "correlation" ]; then \
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
# Compilation Rules (Pattern Matching)
# ============================================================================

# Define compilation rules for each dataset size
# Rule: EX1_optimized_codes/%_gcc_mini: EX1_optimized_codes/%.c

# Mini dataset
$(SRC_DIR)/%_$(COMPILER_TAG)_mini: $(SRC_DIR)/%.c $(POLYBENCH_SRC)
	@echo "  Compiling $* (mini)..."
	@$(CC) -DMINI_DATASET $(CFLAGS) $(INCLUDES) $< $(POLYBENCH_SRC) $(LIBS) -o $@ \
		|| (echo "Error: Failed to build $@"; exit 1)

# Small dataset
$(SRC_DIR)/%_$(COMPILER_TAG)_small: $(SRC_DIR)/%.c $(POLYBENCH_SRC)
	@echo "  Compiling $* (small)..."
	@$(CC) -DSMALL_DATASET $(CFLAGS) $(INCLUDES) $< $(POLYBENCH_SRC) $(LIBS) -o $@ \
		|| (echo "Error: Failed to build $@"; exit 1)

# Medium dataset
$(SRC_DIR)/%_$(COMPILER_TAG)_medium: $(SRC_DIR)/%.c $(POLYBENCH_SRC)
	@echo "  Compiling $* (medium)..."
	@$(CC) -DMEDIUM_DATASET $(CFLAGS) $(INCLUDES) $< $(POLYBENCH_SRC) $(LIBS) -o $@ \
		|| (echo "Error: Failed to build $@"; exit 1)

# Large dataset
$(SRC_DIR)/%_$(COMPILER_TAG)_large: $(SRC_DIR)/%.c $(POLYBENCH_SRC)
	@echo "  Compiling $* (large)..."
	@$(CC) -DLARGE_DATASET $(CFLAGS) $(INCLUDES) $< $(POLYBENCH_SRC) $(LIBS) -o $@ \
		|| (echo "Error: Failed to build $@"; exit 1)

# Extra-large dataset
$(SRC_DIR)/%_$(COMPILER_TAG)_extra-large: $(SRC_DIR)/%.c $(POLYBENCH_SRC)
	@echo "  Compiling $* (extra-large)..."
	@$(CC) -DEXTRALARGE_DATASET $(CFLAGS) $(INCLUDES) $< $(POLYBENCH_SRC) $(LIBS) -o $@ \
		|| (echo "Error: Failed to build $@"; exit 1)

# ============================================================================
# Test Targets
# ============================================================================

test-mini: baseline
	@echo "Testing mini dataset..."
	@./$(SRC_DIR)/correlation_$(COMPILER_TAG)_mini > /dev/null 2>&1 && echo "✓ Mini test passed"

test-small: baseline
	@echo "Testing small dataset..."
	@./$(SRC_DIR)/correlation_$(COMPILER_TAG)_small > /dev/null 2>&1 && echo "✓ Small test passed"

test-medium: baseline
	@echo "Testing medium dataset..."
	@./$(SRC_DIR)/correlation_$(COMPILER_TAG)_medium > /dev/null 2>&1 && echo "✓ Medium test passed"

test-large: baseline
	@echo "Testing large dataset..."
	@./$(SRC_DIR)/correlation_$(COMPILER_TAG)_large > /dev/null 2>&1 && echo "✓ Large test passed"

test-extra-large: baseline
	@echo "Testing extra-large dataset..."
	@./$(SRC_DIR)/correlation_$(COMPILER_TAG)_extra-large > /dev/null 2>&1 && echo "✓ Extra-large test passed"

test-all: test-mini test-small test-medium test-large test-extra-large
	@echo "✓ All tests passed"

# ============================================================================
# Cleanup
# ============================================================================

clean:
	@echo "Cleaning up..."
	@rm -f $(SRC_DIR)/*_gcc $(SRC_DIR)/*_clang
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
	@echo "Correlation Benchmark Makefile"
	@echo "==============================="
	@echo ""
	@echo "Detected compiler: $(COMPILER_TAG)"
	@echo ""
	@echo "Available targets:"
	@echo "  make              - Build all variants with all dataset sizes"
	@echo "  make all          - Same as 'make'"
	@echo "  make variants     - Build all variant binaries (all sizes)"
	@echo "  make baseline     - Build only baseline binaries"
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
	@echo "  ./$(SRC_DIR)/correlation_$(COMPILER_TAG)_mini"
	@echo "  ./$(SRC_DIR)/correlation_claude_v1_$(COMPILER_TAG)_large"
	@echo ""
	@echo "Source files found: $(words $(SOURCE_BASES))"
	@echo "Total binaries to build: $(shell echo $$(($(words $(SOURCE_BASES)) * 5)))"

