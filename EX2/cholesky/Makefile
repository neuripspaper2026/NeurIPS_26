#!/usr/bin/make -f
# Cholesky Benchmark Makefile - Standardized build system

CC := $(shell command -v gcc 2>/dev/null)
ifdef CC
    COMPILER_TAG := gcc
else
    CC := $(shell command -v clang 2>/dev/null)
    ifdef CC
        COMPILER_TAG := clang
    else
        $(error No compiler found)
    endif
endif

OMP_FLAGS ?= -fopenmp
CFLAGS := -O3 $(OMP_FLAGS)
INCLUDES := -I../../utilities -I.
LIBS := -lm
POLYBENCH_SRC := ../../utilities/polybench.c
DATASET_FLAGS := MINI_DATASET SMALL_DATASET MEDIUM_DATASET LARGE_DATASET EXTRALARGE_DATASET

SRC_DIR := EX2_optimized_codes
CORRECTNESS_DIR := EX5_correctness
ALL_SOURCES := $(wildcard $(SRC_DIR)/*.c)
SOURCE_BASES := $(basename $(notdir $(ALL_SOURCES)))

define make_binary_names
$(foreach size,mini small medium large extra-large,$(SRC_DIR)/$(1)_$(COMPILER_TAG)_$(size))
endef

ALL_BINS := $(foreach base,$(SOURCE_BASES),$(call make_binary_names,$(base)))

.PHONY: all baseline variants clean help test-mini test-small test-medium test-large test-extra-large test-all

all: variants

baseline: $(call make_binary_names,cholesky)
	@echo "✓ Baseline binaries built successfully with $(COMPILER_TAG)"

variants: baseline
	@echo "Building all variant binaries with $(COMPILER_TAG)..."
	@failed=0; total=0; \
	for base in $(SOURCE_BASES); do [ "$$base" != "cholesky" ] && total=$$((total+1)); done; \
	current=0; \
	for base in $(SOURCE_BASES); do \
		if [ "$$base" != "cholesky" ]; then \
			current=$$((current+1)); echo "[$$current/$$total] Building $$base..."; \
			variant_failed=0; \
			for size in mini small medium large extra-large; do \
				if ! $(MAKE) --no-print-directory $(SRC_DIR)/$${base}_$(COMPILER_TAG)_$$size 2>/dev/null; then \
					variant_failed=1; break; \
				fi; \
			done; \
			[ $$variant_failed -eq 1 ] && { failed=$$((failed+1)); echo "  ✗ Failed to build $$base (continuing...)"; }; \
		fi; \
	done; \
	echo ""; \
	[ $$failed -eq 0 ] && echo "✓ All variant binaries built successfully" || \
		{ echo "⚠ $$failed variant(s) failed to compile (continued with others)"; \
		  echo "  Successfully built: $$((total - failed))/$$total variants"; }

$(SRC_DIR)/%_$(COMPILER_TAG)_mini: $(SRC_DIR)/%.c $(POLYBENCH_SRC)
	@echo "  Compiling $* (mini)..."
	@$(CC) -DMINI_DATASET $(CFLAGS) $(INCLUDES) $< $(POLYBENCH_SRC) $(LIBS) -o $@ || { echo "Error: Failed to build $@"; exit 1; }

$(SRC_DIR)/%_$(COMPILER_TAG)_small: $(SRC_DIR)/%.c $(POLYBENCH_SRC)
	@echo "  Compiling $* (small)..."
	@$(CC) -DSMALL_DATASET $(CFLAGS) $(INCLUDES) $< $(POLYBENCH_SRC) $(LIBS) -o $@ || { echo "Error: Failed to build $@"; exit 1; }

$(SRC_DIR)/%_$(COMPILER_TAG)_medium: $(SRC_DIR)/%.c $(POLYBENCH_SRC)
	@echo "  Compiling $* (medium)..."
	@$(CC) -DMEDIUM_DATASET $(CFLAGS) $(INCLUDES) $< $(POLYBENCH_SRC) $(LIBS) -o $@ || { echo "Error: Failed to build $@"; exit 1; }

$(SRC_DIR)/%_$(COMPILER_TAG)_large: $(SRC_DIR)/%.c $(POLYBENCH_SRC)
	@echo "  Compiling $* (large)..."
	@$(CC) -DLARGE_DATASET $(CFLAGS) $(INCLUDES) $< $(POLYBENCH_SRC) $(LIBS) -o $@ || { echo "Error: Failed to build $@"; exit 1; }

$(SRC_DIR)/%_$(COMPILER_TAG)_extra-large: $(SRC_DIR)/%.c $(POLYBENCH_SRC)
	@echo "  Compiling $* (extra-large)..."
	@$(CC) -DEXTRALARGE_DATASET $(CFLAGS) $(INCLUDES) $< $(POLYBENCH_SRC) $(LIBS) -o $@ || { echo "Error: Failed to build $@"; exit 1; }

test-mini: baseline
	@echo "Testing mini dataset..." && ./$(SRC_DIR)/cholesky_$(COMPILER_TAG)_mini > /dev/null 2>&1 && echo "✓ Mini test passed"

test-small: baseline
	@echo "Testing small dataset..." && ./$(SRC_DIR)/cholesky_$(COMPILER_TAG)_small > /dev/null 2>&1 && echo "✓ Small test passed"

test-medium: baseline
	@echo "Testing medium dataset..." && ./$(SRC_DIR)/cholesky_$(COMPILER_TAG)_medium > /dev/null 2>&1 && echo "✓ Medium test passed"

test-large: baseline
	@echo "Testing large dataset..." && ./$(SRC_DIR)/cholesky_$(COMPILER_TAG)_large > /dev/null 2>&1 && echo "✓ Large test passed"

test-extra-large: baseline
	@echo "Testing extra-large dataset..." && ./$(SRC_DIR)/cholesky_$(COMPILER_TAG)_extra-large > /dev/null 2>&1 && echo "✓ Extra-large test passed"

test-all: test-mini test-small test-medium test-large test-extra-large
	@echo "✓ All tests passed"

clean:
	@echo "Cleaning up..."
	@rm -f $(SRC_DIR)/*_gcc $(SRC_DIR)/*_clang
	@rm -f $(SRC_DIR)/*_gcc_mini $(SRC_DIR)/*_gcc_small $(SRC_DIR)/*_gcc_medium \
		$(SRC_DIR)/*_gcc_large $(SRC_DIR)/*_gcc_extra-large $(SRC_DIR)/*_gcc_extralarge
	@rm -f $(SRC_DIR)/*_clang_mini $(SRC_DIR)/*_clang_small $(SRC_DIR)/*_clang_medium \
		$(SRC_DIR)/*_clang_large $(SRC_DIR)/*_clang_extra-large $(SRC_DIR)/*_clang_extralarge
	@rm -f $(CORRECTNESS_DIR)/output_*.data
	@echo "✓ Cleanup complete"

help:
	@echo "Cholesky Benchmark Makefile"
	@echo "============================"
	@echo ""
	@echo "Detected compiler: $(COMPILER_TAG)"
	@echo ""
	@echo "Available targets:"
	@echo "  make              - Build all variants (default)"
	@echo "  make baseline     - Build only baseline"
	@echo "  make test-all     - Run all tests"
	@echo "  make clean        - Remove binaries"
	@echo "  make help         - Show this message"
	@echo ""
	@echo "Dataset sizes: mini, small, medium, large, extra-large"
	@echo "Source files found: $(words $(SOURCE_BASES))"
	@echo "Total binaries to build: $(shell echo $$(($(words $(SOURCE_BASES)) * 5)))"
