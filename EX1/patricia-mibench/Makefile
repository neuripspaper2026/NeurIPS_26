CC ?= gcc
CFLAGS ?= -O3 -w -std=gnu89
LDFLAGS ?=

OPT_DIR := EX1_optimized_codes

# Source basename (files are patricia.c / patricia_*.c)
SRC_BASENAME := patricia
# Benchmark/binary name expected by scripts
BENCH_NAME := patricia-mibench

# Detect compiler for binary naming
ifneq (,$(findstring clang,$(CC)))
COMPILER_TAG := clang
else
COMPILER_TAG := gcc
endif

# Baseline and variant source files (baseline: patricia.c, variants: patricia_*.c)
BASE_SRC := $(OPT_DIR)/$(SRC_BASENAME).c
VARIANT_SRCS := $(filter-out $(BASE_SRC),$(wildcard $(OPT_DIR)/$(SRC_BASENAME)_*.c))
SUFFIXES := $(patsubst $(OPT_DIR)/$(SRC_BASENAME)_%.c,%,${VARIANT_SRCS})

# Test driver source file
TEST_SRC := patricia_test.c

# Executable files - baseline ends with _gcc (or _clang)
BASE_BIN := $(OPT_DIR)/$(BENCH_NAME)_$(COMPILER_TAG)
VARIANT_BINS := $(addprefix $(OPT_DIR)/$(BENCH_NAME)_$(COMPILER_TAG)_,${SUFFIXES})
ALL_BINS := $(BASE_BIN) ${VARIANT_BINS}

# Output directory and default input
OUT_DIR := EX5_correctness
INPUT ?= large.udp

.PHONY: all clean bins run

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

$(BASE_BIN): $(BASE_SRC) $(TEST_SRC) Makefile
	@mkdir -p $(OPT_DIR)
	@echo "Building $@..."
	@$(CC) $(CFLAGS) $(BASE_SRC) $(TEST_SRC) $(LDFLAGS) -o $@ || (echo "Error: Failed to build $@"; exit 1)

$(OPT_DIR)/$(BENCH_NAME)_$(COMPILER_TAG)_%: $(OPT_DIR)/$(SRC_BASENAME)_%.c $(TEST_SRC) Makefile
	@mkdir -p $(OPT_DIR)
	@echo "Building $@..."
	@$(CC) $(CFLAGS) $< $(TEST_SRC) $(LDFLAGS) -o $@ || (echo "Error: Failed to build $@"; exit 1)

# Run: redirect stdout to EX5_correctness/
run: bins
	@mkdir -p $(OUT_DIR)
	@echo "[RUN] $(BASE_BIN) $(INPUT)"
	@$(BASE_BIN) $(INPUT) > $(OUT_DIR)/output_$(COMPILER_TAG).data 2>&1 || true
	@for s in ${SUFFIXES}; do \
		bin=$(OPT_DIR)/$(BENCH_NAME)_$(COMPILER_TAG)_$$s; \
		out=$(OUT_DIR)/output_$(COMPILER_TAG)_$$s.data; \
		echo "[RUN] $$bin $(INPUT)"; \
		$$bin $(INPUT) > $$out 2>&1 || true; \
	done

clean:
	@rm -f $(ALL_BINS)
	@rm -f $(OPT_DIR)/$(SRC_BASENAME)_gcc $(OPT_DIR)/$(SRC_BASENAME)_gcc_* $(OPT_DIR)/$(SRC_BASENAME)_clang $(OPT_DIR)/$(SRC_BASENAME)_clang_* 2>/dev/null || true
	@rm -f $(OUT_DIR)/output_* 2>/dev/null || true
