CC ?= gcc
CFLAGS ?= -O3 -I lib -I include

OPT_DIR := EX1_optimized_codes
LIB_SRCS := lib/matrix.c lib/clock.c

# Compiler tag for naming
ifneq (,$(findstring clang,$(CC)))
COMPILER_TAG := clang
else
COMPILER_TAG := gcc
endif

# Base source and variants (注意：变体文件为 matmul_*.c，基准为 matmul.c)
BASE_SRC := $(OPT_DIR)/matmul.c
VARIANT_SRCS := $(filter-out $(BASE_SRC),$(wildcard $(OPT_DIR)/matmul_*.c))
SUFFIXES := $(patsubst $(OPT_DIR)/matmul_%.c,%,${VARIANT_SRCS})

# Executables
BASE_BIN := $(OPT_DIR)/matmul_$(COMPILER_TAG)
VARIANT_BINS := $(addprefix $(OPT_DIR)/matmul_$(COMPILER_TAG)_,${SUFFIXES})

ALL_BINS := $(BASE_BIN) ${VARIANT_BINS}

# Output directory for correctness matrices (当前模块下)
OUT_DIR := EX5_correctness

.PHONY: all clean run bins

all: bins

bins: $(ALL_BINS)

$(BASE_BIN): $(BASE_SRC) ${LIB_SRCS}
	-$(CC) $(CFLAGS) $^ -o $@

$(OPT_DIR)/matmul_$(COMPILER_TAG)_%: $(OPT_DIR)/matmul_%.c ${LIB_SRCS}
	-$(CC) $(CFLAGS) $^ -o $@

# 运行并把矩阵结果文件移入 EX5_correctness
# 使用 SIZE 参数控制矩阵规模，默认 1500
SIZE ?= 1500
run: bins
	@mkdir -p $(OUT_DIR)
	@echo "[RUN] $(BASE_BIN) $(SIZE)"
	@$(BASE_BIN) $(SIZE) >/dev/null 2>&1 || true
	@if [ -f matmul_gcc_output.txt ]; then \
		mv matmul_gcc_output.txt $(OUT_DIR)/output_gcc.data; \
		echo "  -> wrote $(OUT_DIR)/output_gcc.data"; \
	else \
		echo "  !! missing matmul_gcc_output.txt (no matrix dumped?)"; \
	fi
	@for s in ${SUFFIXES}; do \
		bin=$(OPT_DIR)/matmul_gcc_$$s; \
		echo "[RUN] $$bin $(SIZE)"; \
		$$bin $(SIZE) >/dev/null 2>&1 || true; \
		if [ -f matmul_gcc_output.txt ]; then \
			mv matmul_gcc_output.txt $(OUT_DIR)/output_gcc_$$s.data; \
			echo "  -> wrote $(OUT_DIR)/output_gcc_$$s.data"; \
		else \
			echo "  !! missing matmul_gcc_output.txt for $$bin"; \
		fi; \
	done

clean:
	@rm -f $(OPT_DIR)/matmul_gcc $(OPT_DIR)/matmul_clang
	@rm -f $(OPT_DIR)/matmul_gcc_* $(OPT_DIR)/matmul_clang_*
	@rm -f matmul_gcc_output.txt matmul_clang_output.txt
