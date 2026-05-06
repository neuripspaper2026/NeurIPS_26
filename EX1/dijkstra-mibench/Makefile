CC ?= gcc
CFLAGS ?= -O3 -w -std=gnu89
LDFLAGS ?=

OPT_DIR := EX1_optimized_codes

# Detect compiler for binary naming
ifneq (,$(findstring clang,$(CC)))
COMPILER_TAG := clang
else
COMPILER_TAG := gcc
endif

# Source and binary files (align exe name with benchmark)
SRC := $(OPT_DIR)/dijkstra_parameterized.c
BIN := $(OPT_DIR)/dijkstra-mibench_$(COMPILER_TAG)

# Variant detection (dijkstra_parameterized_<suffix>.c)
VARIANT_SRCS := $(filter-out $(SRC),$(wildcard $(OPT_DIR)/dijkstra_parameterized_*.c))
VARIANT_SUFFIXES := $(patsubst $(OPT_DIR)/dijkstra_parameterized_%.c,%, $(VARIANT_SRCS))
VARIANT_BINS := $(addprefix $(OPT_DIR)/dijkstra-mibench_$(COMPILER_TAG)_,$(VARIANT_SUFFIXES))
ALL_BINS := $(BIN) $(VARIANT_BINS)

# Output directory and default input
OUT_DIR := EX5_correctness
INPUT ?= input_data/mini/input/input.dat

.PHONY: all clean run test-sizes bins

all: bins

bins: $(ALL_BINS)

# Build binary
$(BIN): $(SRC) Makefile
	@mkdir -p $(OPT_DIR)
	@echo "Building $@..."
	-@$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@ || echo "Error: Failed to build $@"

$(OPT_DIR)/dijkstra-mibench_$(COMPILER_TAG)_%: $(OPT_DIR)/dijkstra_parameterized_%.c Makefile
	@mkdir -p $(OPT_DIR)
	@echo "Building $@..."
	-@$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@ || echo "Error: Failed to build $@"

# Run: redirect stdout to EX5_correctness/
run: $(BIN)
	@mkdir -p $(OUT_DIR)
	@echo "[RUN] $(BIN) $(INPUT)"
	@$(BIN) $(INPUT) > $(OUT_DIR)/output_$(COMPILER_TAG).data 2>&1 || true

# Test all sizes
test-sizes: $(BIN)
	@echo "Testing all sizes with $(BIN)..."
	@for size in mini small medium large extra-large; do \
		input=input_data/$$size/input/input.dat; \
		if [ -f $$input ]; then \
			echo ""; \
			echo "=== Testing $$size ==="; \
			time $(BIN) $$input 2>&1 | head -5; \
		fi; \
	done

clean:
	@rm -f $(ALL_BINS)
	@rm -f $(OUT_DIR)/output_* 2>/dev/null || true
