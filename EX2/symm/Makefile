#!/usr/bin/make -f
# SYMM Benchmark Makefile
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
SRC_DIR := EX2_optimized_codes
CORRECTNESS_DIR := EX5_correctness
ALL_SOURCES := $(wildcard $(SRC_DIR)/*.c)
SOURCE_BASES := $(basename $(notdir $(ALL_SOURCES)))
define make_binary_names
$(foreach size,mini small medium large extra-large,$(SRC_DIR)/$(1)_$(COMPILER_TAG)_$(size))
endef
.PHONY: all baseline variants clean help test-mini test-small test-medium test-large test-extra-large
all: variants
baseline: $(call make_binary_names,symm)
	@echo "✓ Baseline binaries built successfully with $(COMPILER_TAG)"
variants: baseline
	@echo "Building all variant binaries with $(COMPILER_TAG)..."
	@failed=0; total=0; for base in $(SOURCE_BASES); do if [ "$$base" != "symm" ]; then total=$$((total+1)); fi; done; current=0; for base in $(SOURCE_BASES); do if [ "$$base" != "symm" ]; then current=$$((current+1)); echo "[$$current/$$total] Building $$base..."; variant_failed=0; for size in mini small medium large extra-large; do if ! $(MAKE) --no-print-directory $(SRC_DIR)/$${base}_$(COMPILER_TAG)_$$size 2>/dev/null; then variant_failed=1; break; fi; done; if [ $$variant_failed -eq 1 ]; then failed=$$((failed+1)); echo "  ✗ Failed to build $$base (continuing...)"; fi; fi; done; echo ""; if [ $$failed -eq 0 ]; then echo "✓ All variant binaries built successfully"; else echo "⚠ $$failed variant(s) failed to compile (continued with others)"; echo "  Successfully built: $$((total - failed))/$$total variants"; fi
$(SRC_DIR)/%_$(COMPILER_TAG)_mini: $(SRC_DIR)/%.c $(POLYBENCH_SRC)
	@echo "  Compiling $* (mini)..."
	@$(CC) -DMINI_DATASET $(CFLAGS) $(INCLUDES) $< $(POLYBENCH_SRC) $(LIBS) -o $@ || (echo "Error: Failed to build $@"; exit 1)
$(SRC_DIR)/%_$(COMPILER_TAG)_small: $(SRC_DIR)/%.c $(POLYBENCH_SRC)
	@echo "  Compiling $* (small)..."
	@$(CC) -DSMALL_DATASET $(CFLAGS) $(INCLUDES) $< $(POLYBENCH_SRC) $(LIBS) -o $@ || (echo "Error: Failed to build $@"; exit 1)
$(SRC_DIR)/%_$(COMPILER_TAG)_medium: $(SRC_DIR)/%.c $(POLYBENCH_SRC)
	@echo "  Compiling $* (medium)..."
	@$(CC) -DMEDIUM_DATASET $(CFLAGS) $(INCLUDES) $< $(POLYBENCH_SRC) $(LIBS) -o $@ || (echo "Error: Failed to build $@"; exit 1)
$(SRC_DIR)/%_$(COMPILER_TAG)_large: $(SRC_DIR)/%.c $(POLYBENCH_SRC)
	@echo "  Compiling $* (large)..."
	@$(CC) -DLARGE_DATASET $(CFLAGS) $(INCLUDES) $< $(POLYBENCH_SRC) $(LIBS) -o $@ || (echo "Error: Failed to build $@"; exit 1)
$(SRC_DIR)/%_$(COMPILER_TAG)_extra-large: $(SRC_DIR)/%.c $(POLYBENCH_SRC)
	@echo "  Compiling $* (extra-large)..."
	@$(CC) -DEXTRALARGE_DATASET $(CFLAGS) $(INCLUDES) $< $(POLYBENCH_SRC) $(LIBS) -o $@ || (echo "Error: Failed to build $@"; exit 1)
test-mini: baseline
	@echo "Testing mini dataset..."
	@./$(SRC_DIR)/symm_$(COMPILER_TAG)_mini > /dev/null 2>&1 && echo "✓ Mini test passed"
test-small: baseline
	@echo "Testing small dataset..."
	@./$(SRC_DIR)/symm_$(COMPILER_TAG)_small > /dev/null 2>&1 && echo "✓ Small test passed"
test-medium: baseline
	@echo "Testing medium dataset..."
	@./$(SRC_DIR)/symm_$(COMPILER_TAG)_medium > /dev/null 2>&1 && echo "✓ Medium test passed"
test-large: baseline
	@echo "Testing large dataset..."
	@./$(SRC_DIR)/symm_$(COMPILER_TAG)_large > /dev/null 2>&1 && echo "✓ Large test passed"
test-extra-large: baseline
	@echo "Testing extra-large dataset..."
	@./$(SRC_DIR)/symm_$(COMPILER_TAG)_extra-large > /dev/null 2>&1 && echo "✓ Extra-large test passed"
test-all: test-mini test-small test-medium test-large test-extra-large
	@echo "✓ All tests passed"
clean:
	@echo "Cleaning up..."
	@rm -f $(SRC_DIR)/*_gcc $(SRC_DIR)/*_clang $(SRC_DIR)/*_gcc_* $(SRC_DIR)/*_clang_* $(CORRECTNESS_DIR)/output_*.data
	@echo "✓ Cleanup complete"
help:
	@echo "SYMM Benchmark Makefile"
	@echo "Detected compiler: $(COMPILER_TAG)"
	@echo "Source files found: $(words $(SOURCE_BASES))"
	@echo "Total binaries to build: $(shell echo $$(($(words $(SOURCE_BASES)) * 5)))"
