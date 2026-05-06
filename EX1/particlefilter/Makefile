.RECIPEPREFIX := >
CC      ?= gcc
CFLAGS  ?= -O3
LDFLAGS ?= -lm
INCLUDES = -I. -IEX1_optimized_codes

# 源码文件前缀（用于定位 EX1_optimized_codes/particle-filter.c 等）
SRC_BASENAME := particle-filter
# benchmark 名（用于生成脚本期望的可执行文件名 particlefilter_gcc）
BENCH_NAME   := particlefilter
KERNEL_DIR := EX1_optimized_codes

# 发现变体：EX1_optimized_codes/particle-filter_*.c
VARIANT_SRC := $(wildcard $(KERNEL_DIR)/$(SRC_BASENAME)_*.c)
VARIANTS    := $(patsubst $(KERNEL_DIR)/$(SRC_BASENAME)_%.c,%, $(VARIANT_SRC))

# 编译器标签（gcc/clang）
COMPILER_TAG := gcc
ifneq (,$(findstring clang,$(CC)))
COMPILER_TAG := clang
endif

# 基线源码：优先使用 EX1_optimized_codes/particle-filter.c，否则回退仓库根目录
BASE_SRC := $(SRC_BASENAME).c
ifneq (,$(wildcard $(KERNEL_DIR)/$(SRC_BASENAME).c))
BASE_SRC := $(KERNEL_DIR)/$(SRC_BASENAME).c
endif

# 产物全部输出到 EX1_optimized_codes/
# Baseline 使用带编译器标签的名字，如 particlefilter_gcc
BASE_EXE     := $(KERNEL_DIR)/$(BENCH_NAME)_$(COMPILER_TAG)
VARIANT_EXES := $(foreach v,$(VARIANTS),$(KERNEL_DIR)/$(BENCH_NAME)_$(COMPILER_TAG)_$(v))

.PHONY: all clean

all: $(BASE_EXE) $(VARIANT_EXES)

# 基线构建（单翻译单元，直接从 .c 编译到可执行文件，不生成 .o 文件）
$(BASE_EXE): $(BASE_SRC)
>-$(CC) $(CFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

# 变体构建：EX1_optimized_codes/particle-filter_%.c -> EX1_optimized_codes/particlefilter_<cc>_%（直接编译，不生成 .o 文件）
$(KERNEL_DIR)/$(BENCH_NAME)_$(COMPILER_TAG)_%: $(KERNEL_DIR)/$(SRC_BASENAME)_%.c
>-$(CC) $(CFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

clean:
>rm -f $(KERNEL_DIR)/$(BENCH_NAME)_gcc $(KERNEL_DIR)/$(BENCH_NAME)_gcc_* $(KERNEL_DIR)/$(BENCH_NAME)_clang $(KERNEL_DIR)/$(BENCH_NAME)_clang_* \
>      $(KERNEL_DIR)/$(SRC_BASENAME)_gcc $(KERNEL_DIR)/$(SRC_BASENAME)_gcc_* $(KERNEL_DIR)/$(SRC_BASENAME)_clang $(KERNEL_DIR)/$(SRC_BASENAME)_clang_* \
>      output.txt