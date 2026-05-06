CC ?= gcc
CFLAGS ?= -O3 -I include
LDFLAGS ?= -lm

OPT_DIR := EX1_optimized_codes

# Compiler tag for naming
ifneq (,$(findstring clang,$(CC)))
COMPILER_TAG := clang
else
COMPILER_TAG := gcc
endif

# Base source and variants (基准为 pi.c，变体为 pi_*.c)
BASE_SRC := $(OPT_DIR)/pi.c
VARIANT_SRCS := $(filter-out $(BASE_SRC),$(wildcard $(OPT_DIR)/pi_*.c))
SUFFIXES := $(patsubst $(OPT_DIR)/pi_%.c,%,${VARIANT_SRCS})

# Executables
BASE_BIN := $(OPT_DIR)/pi_$(COMPILER_TAG)
VARIANT_BINS := $(addprefix $(OPT_DIR)/pi_$(COMPILER_TAG)_,${SUFFIXES})
ALL_BINS := $(BASE_BIN) ${VARIANT_BINS}

# Output directory (当前模块下)
OUT_DIR := EX5_correctness

.PHONY: all clean bins run

all: bins

bins: $(ALL_BINS)

$(BASE_BIN): $(BASE_SRC)
	-$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

$(OPT_DIR)/pi_$(COMPILER_TAG)_%: $(OPT_DIR)/pi_%.c
	-$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

# 运行并将标准输出写入 EX5_correctness/output_gcc*.data
STEPS ?= 100000000
run: bins
	@mkdir -p $(OUT_DIR)
	@echo "[RUN] $(BASE_BIN) $(STEPS)"
	@$(BASE_BIN) $(STEPS) > $(OUT_DIR)/output_gcc.data 2>&1 || true
	@for s in ${SUFFIXES}; do \
		bin=$(OPT_DIR)/pi_gcc_$$s; \
		echo "[RUN] $$bin $(STEPS)"; \
		$$bin $(STEPS) > $(OUT_DIR)/output_gcc_$$s.data 2>&1 || true; \
	done

clean:
	@rm -f $(OPT_DIR)/pi_gcc $(OPT_DIR)/pi_clang
	@rm -f $(OPT_DIR)/pi_gcc_* $(OPT_DIR)/pi_clang_*
