CC ?= gcc
CFLAGS ?= -O3 -w -I .
LDFLAGS ?= -lm

OPT_DIR := EX1_optimized_codes

# Source basename (files are susan.c / susan_*.c)
SRC_BASENAME := susan
# Benchmark/binary name expected by scripts
BENCH_NAME := susan-s-mibench

# Detect compiler for binary naming
ifneq (,$(findstring clang,$(CC)))
COMPILER_TAG := clang
else
COMPILER_TAG := gcc
endif

# Baseline and variant source files (baseline: susan.c, variants: susan_*.c)
BASE_SRC := $(OPT_DIR)/$(SRC_BASENAME).c
VARIANT_SRCS := $(filter-out $(BASE_SRC),$(wildcard $(OPT_DIR)/$(SRC_BASENAME)_*.c))
SUFFIXES := $(patsubst $(OPT_DIR)/$(SRC_BASENAME)_%.c,%,${VARIANT_SRCS})

# Executable files - baseline ends with _gcc (or _clang)
BASE_BIN := $(OPT_DIR)/$(BENCH_NAME)_$(COMPILER_TAG)
VARIANT_BINS := $(addprefix $(OPT_DIR)/$(BENCH_NAME)_$(COMPILER_TAG)_,${SUFFIXES})
ALL_BINS := $(BASE_BIN) ${VARIANT_BINS}

# Output directory and fixed input
OUT_DIR := EX5_correctness
INPUT := input_large.pgm

.PHONY: all clean bins run run-variants

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

$(BASE_BIN): $(BASE_SRC) Makefile
	@mkdir -p $(OPT_DIR)
	@echo "Building $@..."
	@$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@ || (echo "Error: Failed to build $@"; exit 1)

$(OPT_DIR)/$(BENCH_NAME)_$(COMPILER_TAG)_%: $(OPT_DIR)/$(SRC_BASENAME)_%.c Makefile
	@mkdir -p $(OPT_DIR)
	@echo "Building $@..."
	@$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@ || (echo "Error: Failed to build $@"; exit 1)

# Run: input is PGM, output image written to EX5_correctness/
# Only run smoothing mode: susan input_large.pgm output_large.smoothing.pgm -s
run: bins
	@mkdir -p $(OUT_DIR)
	@echo "[RUN] $(BASE_BIN) $(INPUT) $(OUT_DIR)/output_large.smoothing.pgm -s"
	@$(BASE_BIN) $(INPUT) $(OUT_DIR)/output_large.smoothing.pgm -s > $(OUT_DIR)/output_$(COMPILER_TAG).log 2>&1 || true
	@for s in ${SUFFIXES}; do \
		bin=$(OPT_DIR)/$(BENCH_NAME)_$(COMPILER_TAG)_$$s; \
		out=$(OUT_DIR)/output_large.smoothing_$$s.pgm; \
		log=$(OUT_DIR)/output_$(COMPILER_TAG)_$$s.log; \
		echo "[RUN] $$bin $(INPUT) $$out -s"; \
		$$bin $(INPUT) $$out -s > $$log 2>&1 || true; \
	done

run-variants: bins
	@mkdir -p $(OUT_DIR)
	@for s in ${SUFFIXES}; do \
		bin=$(OPT_DIR)/$(BENCH_NAME)_$(COMPILER_TAG)_$$s; \
		out=$(OUT_DIR)/output_$(COMPILER_TAG)_$$s.pgm; \
		log=$(OUT_DIR)/output_$(COMPILER_TAG)_$$s.log; \
		echo "[RUN] $$bin $(INPUT) -> $$out"; \
		$$bin $(INPUT) $$out > $$log 2>&1 || true; \
	done

clean:
	@rm -f $(ALL_BINS)
	@rm -f $(OPT_DIR)/susan_gcc $(OPT_DIR)/susan_gcc_* $(OPT_DIR)/susan_clang $(OPT_DIR)/susan_clang_* 2>/dev/null || true
	@rm -f $(OUT_DIR)/output_* 2>/dev/null || true
