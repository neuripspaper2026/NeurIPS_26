CC ?= gcc
OMP_FLAGS ?= -fopenmp
CFLAGS ?= -std=c99 -O3 -I lib -D_POSIX_C_SOURCE=200809L
CFLAGS += $(OMP_FLAGS)
LDFLAGS ?= -lm -lrt
LDFLAGS += $(OMP_FLAGS)

OPT_DIR := EX2_optimized_codes
LIB_SRCS := lib/CRSMatrix.c lib/Matrix2D.c lib/Vector.c

# Compiler tag for naming
ifneq (,$(findstring clang,$(CC)))
COMPILER_TAG := clang
else
COMPILER_TAG := gcc
endif

# 基准与变体源文件（基准为 atmux.c，变体为 atmux_*.c）
BASE_SRC := $(OPT_DIR)/atmux.c
VARIANT_SRCS := $(filter-out $(BASE_SRC),$(wildcard $(OPT_DIR)/atmux_*.c))
SUFFIXES := $(patsubst $(OPT_DIR)/atmux_%.c,%,${VARIANT_SRCS})

# 可执行文件
BASE_BIN := $(OPT_DIR)/atmux_$(COMPILER_TAG)
VARIANT_BINS := $(addprefix $(OPT_DIR)/atmux_$(COMPILER_TAG)_,${SUFFIXES})
ALL_BINS := $(BASE_BIN) ${VARIANT_BINS}

# 输出目录（当前模块下）
OUT_DIR := EX5_correctness

.PHONY: all clean bins run

all: bins

bins: $(ALL_BINS)

$(BASE_BIN): $(BASE_SRC) $(LIB_SRCS)
	-$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

$(OPT_DIR)/atmux_$(COMPILER_TAG)_%: $(OPT_DIR)/atmux_%.c $(LIB_SRCS)
	-$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

# 运行并将标准输出写入 EX5_correctness/output_gcc*.data
SIZE ?= 25000
run: bins
	@mkdir -p $(OUT_DIR)
	@echo "[RUN] $(BASE_BIN) $(SIZE)"
	@$(BASE_BIN) $(SIZE) > $(OUT_DIR)/output_gcc.data 2>&1 || true
	@for s in ${SUFFIXES}; do \
		bin=$(OPT_DIR)/atmux_gcc_$$s; \
		echo "[RUN] $$bin $(SIZE)"; \
		$$bin $(SIZE) > $(OUT_DIR)/output_gcc_$$s.data 2>&1 || true; \
	done

clean:
	@rm -f $(OPT_DIR)/atmux_gcc $(OPT_DIR)/atmux_clang
	@rm -f $(OPT_DIR)/atmux_gcc_* $(OPT_DIR)/atmux_clang_*
