CC ?= gcc
CFLAGS ?= -O3 -I .
LDFLAGS ?=

OPT_DIR := EX1_optimized_codes

# Common sources (excluding driver bitcnts.c)
COMMON_SRCS := bitcnt_1.c bitcnt_2.c bitcnt_3.c bitcnt_4.c bitfiles.c bitstrng.c bstr_i.c

# Baseline and variant bitcnts sources
BASE_SRC := $(OPT_DIR)/bitcnts.c
VARIANT_SRCS := $(filter-out $(BASE_SRC),$(wildcard $(OPT_DIR)/bitcnts_*.c))
SUFFIXES := $(patsubst $(OPT_DIR)/bitcnts_%.c,%,${VARIANT_SRCS})

# Executable files - align with benchmark name (bitcount-mibench_gcc)
BASE_BIN := $(OPT_DIR)/bitcount-mibench_gcc
VARIANT_BINS := $(addprefix $(OPT_DIR)/bitcount-mibench_gcc_,${SUFFIXES})
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

$(BASE_BIN): $(COMMON_SRCS) $(BASE_SRC)
	@mkdir -p $(OPT_DIR)
	@echo "Building $@..."
	@$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@ || (echo "Error: Failed to build $@"; exit 1)

$(OPT_DIR)/bitcount-mibench_gcc_%: $(COMMON_SRCS) $(OPT_DIR)/bitcnts_%.c
	@mkdir -p $(OPT_DIR)
	@echo "Building $@..."
	@$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@ || (echo "Error: Failed to build $@"; exit 1)

# Run and write stdout to EX5_correctness/output_gcc*.data (only 7 lines of Bits for consistency check)
ITER ?= 1125000
RUN_FLAGS ?= --bits-only
run: bins
	@mkdir -p $(OUT_DIR)
	@echo "[RUN] $(BASE_BIN) $(ITER) $(RUN_FLAGS)"
	@$(BASE_BIN) $(ITER) $(RUN_FLAGS) > $(OUT_DIR)/output_gcc.data 2>&1 || true
	@for s in ${SUFFIXES}; do \
		bin=$(OPT_DIR)/bitcount-mibench_gcc_$$s; \
		echo "[RUN] $$bin $(ITER) $(RUN_FLAGS)"; \
		$$bin $(ITER) $(RUN_FLAGS) > $(OUT_DIR)/output_gcc_$$s.data 2>&1 || true; \
	done

run-variants: bins
	@mkdir -p $(OUT_DIR)
	@for s in ${SUFFIXES}; do \
		bin=$(OPT_DIR)/bitcount-mibench_gcc_$$s; \
		echo "[RUN] $$bin $(ITER) $(RUN_FLAGS)"; \
		$$bin $(ITER) $(RUN_FLAGS) > $(OUT_DIR)/output_gcc_$$s.data 2>&1 || true; \
	done

clean:
	@rm -f $(ALL_BINS)
