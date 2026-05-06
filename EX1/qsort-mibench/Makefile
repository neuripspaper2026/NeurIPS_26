CC ?= gcc
CFLAGS ?= -O3 -I .
LDFLAGS ?= -lm

OPT_DIR := EX1_optimized_codes

# Source basename (files are qsort_large.c / qsort_large_*.c)
SRC_BASENAME := qsort_large
# Benchmark/binary name expected by scripts
BENCH_NAME := qsort-mibench

# Detect compiler for binary naming
ifneq (,$(findstring clang,$(CC)))
COMPILER_TAG := clang
else
COMPILER_TAG := gcc
endif

# Baseline and variant source files (baseline: qsort_large.c, variants: qsort_large_*.c)
BASE_SRC := $(OPT_DIR)/$(SRC_BASENAME).c
VARIANT_SRCS := $(filter-out $(BASE_SRC),$(wildcard $(OPT_DIR)/$(SRC_BASENAME)_*.c))
SUFFIXES := $(patsubst $(OPT_DIR)/$(SRC_BASENAME)_%.c,%,${VARIANT_SRCS})

# Executable files - baseline ends with _gcc (or _clang)
BASE_BIN := $(OPT_DIR)/$(BENCH_NAME)_$(COMPILER_TAG)
VARIANT_BINS := $(addprefix $(OPT_DIR)/$(BENCH_NAME)_$(COMPILER_TAG)_,${SUFFIXES})
ALL_BINS := $(BASE_BIN) ${VARIANT_BINS}

# Output directory
OUT_DIR := EX5_correctness

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

$(BASE_BIN): $(BASE_SRC)
	@mkdir -p $(OPT_DIR)
	@echo "Building $@..."
	@$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@ || (echo "Error: Failed to build $@"; exit 1)

$(OPT_DIR)/$(BENCH_NAME)_$(COMPILER_TAG)_%: $(OPT_DIR)/$(SRC_BASENAME)_%.c
	@mkdir -p $(OPT_DIR)
	@echo "Building $@..."
	@$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@ || (echo "Error: Failed to build $@"; exit 1)

# Run and write stdout to EX5_correctness/output_<compiler>*.data
INPUT ?= input_large.dat
run: bins
	@mkdir -p $(OUT_DIR)
	@echo "[RUN] $(BASE_BIN) $(INPUT)"
	@$(BASE_BIN) $(INPUT) > $(OUT_DIR)/output_$(COMPILER_TAG).data 2>&1 || true
	@for s in ${SUFFIXES}; do \
		bin=$(OPT_DIR)/$(BENCH_NAME)_$(COMPILER_TAG)_$$s; \
		echo "[RUN] $$bin $(INPUT)"; \
		$$bin $(INPUT) > $(OUT_DIR)/output_$(COMPILER_TAG)_$$s.data 2>&1 || true; \
	done

run-variants: bins
	@mkdir -p $(OUT_DIR)
	@for s in ${SUFFIXES}; do \
		bin=$(OPT_DIR)/$(BENCH_NAME)_$(COMPILER_TAG)_$$s; \
		echo "[RUN] $$bin $(INPUT)"; \
		$$bin $(INPUT) > $(OUT_DIR)/output_$(COMPILER_TAG)_$$s.data 2>&1 || true; \
	done

clean:
	@rm -f $(ALL_BINS)
	@rm -f $(OPT_DIR)/$(SRC_BASENAME)_gcc $(OPT_DIR)/$(SRC_BASENAME)_gcc_* $(OPT_DIR)/$(SRC_BASENAME)_clang $(OPT_DIR)/$(SRC_BASENAME)_clang_* 2>/dev/null || true
