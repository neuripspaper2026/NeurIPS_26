CC ?= gcc
CFLAGS ?= -O3 -w
LDFLAGS ?=

# Detect compiler for binary naming
ifneq (,$(findstring clang,$(CC)))
COMPILER_TAG := clang
else
COMPILER_TAG := gcc
endif

# Priority: use EX1_optimized_codes if exists, fallback to EX1_optimize_codes
ifneq (,$(wildcard EX1_optimized_codes))
OPT_DIR := EX1_optimized_codes
else
OPT_DIR := EX1_optimize_codes
endif

# Source basename (file is pbmsrch_parameterized.c)
SRC_BASENAME := pbmsrch_parameterized
# Benchmark/binary name expected by scripts
BENCH_NAME := stringsearch-mibench

# Parameterized source (baseline)
PARAM_SRC := $(OPT_DIR)/$(SRC_BASENAME).c
PARAM_BIN := $(OPT_DIR)/$(BENCH_NAME)_$(COMPILER_TAG)

# Variant sources and executables (variants: pbmsrch_parameterized_*.c)
VARIANT_SRCS := $(filter-out $(PARAM_SRC),$(wildcard $(OPT_DIR)/$(SRC_BASENAME)_*.c))
SUFFIXES := $(patsubst $(OPT_DIR)/$(SRC_BASENAME)_%.c,%,$(VARIANT_SRCS))
VARIANT_BINS := $(addprefix $(OPT_DIR)/$(BENCH_NAME)_$(COMPILER_TAG)_,${SUFFIXES})
ALL_BINS := $(PARAM_BIN) $(VARIANT_BINS)

# Output directory
OUT_DIR := EX5_correctness

.PHONY: all clean test-sizes bins

# Default target: build parameterized binary
all: bins

# Continue on error - try to build all binaries even if some fail
bins:
	@echo "Building all binaries..."
	@failed=0; \
	for bin in $(ALL_BINS); do \
		$(MAKE) --no-print-directory $$bin || failed=$$((failed+1)); \
	done; \
	if [ $$failed -gt 0 ]; then \
		echo "Warning: $$failed binaries failed to build"; \
	fi

# Parameterized binary (only binary we build)
$(PARAM_BIN): $(PARAM_SRC) Makefile
	@mkdir -p $(OPT_DIR)
	@echo "Building $@..."
	@$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@ || (echo "Error: Failed to build $@"; exit 1)

# Variant binaries: pbmsrch_parameterized_<suffix>.c -> stringsearch-mibench_<compiler>_<suffix>
$(OPT_DIR)/$(BENCH_NAME)_$(COMPILER_TAG)_%: $(OPT_DIR)/$(SRC_BASENAME)_%.c Makefile
	@mkdir -p $(OPT_DIR)
	@echo "Building $@..."
	@$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@ || (echo "Error: Failed to build $@"; exit 1)

# Test parameterized binary with all sizes
test-sizes: bins
	@echo "Testing all sizes with parameterized binary..."
	@for exe in $(ALL_BINS); do \
		echo ""; \
		echo "==========================================================="; \
		echo "Executable: $$exe"; \
		echo "==========================================================="; \
		for size in mini small medium large extra-large; do \
			echo "Running size: $$size"; \
			time $$exe $$size > /dev/null || echo "Failed: $$size"; \
		done; \
	done

clean:
	@rm -f $(ALL_BINS)
	@rm -f $(OPT_DIR)/pbmsrch_$(COMPILER_TAG) 2>/dev/null || true
	@rm -f $(OUT_DIR)/output_* 2>/dev/null || true
