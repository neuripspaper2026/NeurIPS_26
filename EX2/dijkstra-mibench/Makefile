CC ?= gcc
OMP_FLAGS ?= -fopenmp
CFLAGS ?= -O3 -w -std=gnu89
CFLAGS += $(OMP_FLAGS)
LDFLAGS ?=
LDFLAGS += $(OMP_FLAGS)

OPT_DIR := EX2_optimized_codes

# Detect compiler for binary naming
ifneq (,$(findstring clang,$(CC)))
COMPILER_TAG := clang
else
COMPILER_TAG := gcc
endif

# Source files and derived binary names (match correctness-run expectations)
BASE_SRC := $(OPT_DIR)/dijkstra_parameterized.c
VARIANT_SRCS := $(filter-out $(BASE_SRC),$(wildcard $(OPT_DIR)/dijkstra_parameterized_*.c))
SUFFIXES := $(patsubst $(OPT_DIR)/dijkstra_parameterized_%.c,%,${VARIANT_SRCS})

BASE_BIN := $(OPT_DIR)/dijkstra-mibench_$(COMPILER_TAG)
VARIANT_BINS := $(addprefix $(OPT_DIR)/dijkstra-mibench_$(COMPILER_TAG)_,$(SUFFIXES))
ALL_BINS := $(BASE_BIN) $(VARIANT_BINS)

# Output directory and default input
OUT_DIR := EX5_correctness
INPUT ?= input_data/mini/input/input.dat

.PHONY: all clean run test-sizes

# Default: build baseline + all variant binaries (ignore individual failures).
all: $(ALL_BINS)

$(BASE_BIN): $(BASE_SRC) Makefile
	@mkdir -p $(OPT_DIR)
	@echo "Building $@..."
	-@$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@ || echo "[WARN] Failed to build $@"

$(OPT_DIR)/dijkstra-mibench_$(COMPILER_TAG)_%: $(OPT_DIR)/dijkstra_parameterized_%.c Makefile
	@mkdir -p $(OPT_DIR)
	@echo "Building $@..."
	-@$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@ || echo "[WARN] Failed to build $@"

# Run: redirect stdout to EX5_correctness/
run: $(BASE_BIN)
	@mkdir -p $(OUT_DIR)
	@echo "[RUN] $(BASE_BIN) $(INPUT)"
	@$(BASE_BIN) $(INPUT) > $(OUT_DIR)/output_$(COMPILER_TAG).data 2>&1 || true

# Test all sizes
test-sizes: $(BASE_BIN)
	@echo "Testing all sizes with $(BASE_BIN)..."
	@for size in mini small medium large extra-large; do \
		input=input_data/$$size/input/input.dat; \
		if [ -f $$input ]; then \
			echo ""; \
			echo "=== Testing $$size ==="; \
			time $(BASE_BIN) $$input 2>&1 | head -5; \
		fi; \
	done

clean:
	@rm -f $(ALL_BINS)
	@rm -f $(OUT_DIR)/output_* 2>/dev/null || true
