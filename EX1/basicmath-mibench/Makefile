CC ?= gcc
CFLAGS ?= -O3 -w -I .
LDFLAGS ?= -lm

OPT_DIR := EX1_optimized_codes

# Common sources (excluding cubic.c)
COMMON_SRCS := basicmath_large.c rad2deg.c isqrt.c
PARAM_COMMON_SRCS := basicmath_parameterized.c rad2deg.c isqrt.c

# Baseline and variant cubic sources
BASE_CUBIC := $(OPT_DIR)/cubic.c
VARIANT_CUBICS := $(filter-out $(BASE_CUBIC),$(wildcard $(OPT_DIR)/cubic_*.c))
SUFFIXES := $(patsubst $(OPT_DIR)/cubic_%.c,%,${VARIANT_CUBICS})

# Executable files (align name with scripts expectation: basicmath-mibench_gcc)
BASE_BIN := $(OPT_DIR)/basicmath-mibench_gcc
VARIANT_BINS := $(addprefix $(OPT_DIR)/basicmath-mibench_gcc_,${SUFFIXES})
PARAM_BIN := $(BASE_BIN)
ALL_BINS := $(BASE_BIN) ${VARIANT_BINS}

# Output directory
OUT_DIR := EX5_correctness

.PHONY: all clean bins run run-variants parameterized test-sizes

all: bins

bins: $(ALL_BINS)

$(BASE_BIN): $(PARAM_COMMON_SRCS) $(BASE_CUBIC)
	@mkdir -p $(OPT_DIR)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

$(OPT_DIR)/basicmath-mibench_gcc_%: $(PARAM_COMMON_SRCS) $(OPT_DIR)/cubic_%.c
	@mkdir -p $(OPT_DIR)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

# Run and write stdout to EX5_correctness/output_gcc*.data
run: bins
	@mkdir -p $(OUT_DIR)
	@echo "[RUN] $(BASE_BIN)"
	@$(BASE_BIN) > $(OUT_DIR)/output_gcc.data 2>&1 || true
	@for s in ${SUFFIXES}; do \
		bin=$(OPT_DIR)/basicmath-mibench_gcc_$$s; \
		echo "[RUN] $$bin"; \
		$$bin > $(OUT_DIR)/output_gcc_$$s.data 2>&1 || true; \
	done

run-variants: bins
	@mkdir -p $(OUT_DIR)
	@for s in ${SUFFIXES}; do \
		bin=$(OPT_DIR)/basicmath-mibench_gcc_$$s; \
		echo "[RUN] $$bin"; \
		$$bin > $(OUT_DIR)/output_gcc_$$s.data 2>&1 || true; \
	done

# Compile parameterized version
parameterized: $(PARAM_BIN)
	@echo "Built parameterized version: $(PARAM_BIN)"
	@echo "Usage: $(PARAM_BIN) [mini|small|medium|large|extra-large]"

# Test all sizes
test-sizes: $(PARAM_BIN)
	@echo "Testing all sizes..."
	@for size in mini small medium large extra-large; do \
		echo ""; \
		echo "========================================"; \
		echo "Testing size: $$size"; \
		echo "========================================"; \
		time $(PARAM_BIN) $$size > /dev/null; \
	done

clean:
	@rm -f $(ALL_BINS)
