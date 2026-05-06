.RECIPEPREFIX := >
CC      ?= gcc
CFLAGS  ?= -O3
LDFLAGS ?= -lm
INCLUDES= -I. -IEX1_optimized_codes

SRC_BASENAME := srad
BENCH_NAME := srad_v1
KERNEL_DIR := EX1_optimized_codes

# 发现变体：EX1_optimized_codes/srad_*.c
VARIANT_SRC := $(wildcard $(KERNEL_DIR)/$(SRC_BASENAME)_*.c)
VARIANTS    := $(patsubst $(KERNEL_DIR)/$(SRC_BASENAME)_%.c,%, $(VARIANT_SRC))

# 编译器标签（gcc/clang）
COMPILER_TAG := gcc
ifneq (,$(findstring clang,$(CC)))
COMPILER_TAG := clang
endif

# 基线源码：优先使用 EX1_optimized_codes/srad.c，否则回退仓库根目录的 srad.c
BASE_SRC := $(SRC_BASENAME).c
ifneq (,$(wildcard $(KERNEL_DIR)/$(SRC_BASENAME).c))
BASE_SRC := $(KERNEL_DIR)/$(SRC_BASENAME).c
endif

# 产物全部输出到 EX1_optimized_codes/
# Baseline 使用带编译器标签的名字，如 srad_v1_gcc
BASE_EXE     := $(KERNEL_DIR)/$(BENCH_NAME)_$(COMPILER_TAG)
VARIANT_EXES := $(foreach v,$(VARIANTS),$(KERNEL_DIR)/$(BENCH_NAME)_$(COMPILER_TAG)_$(v))

.PHONY: all clean

all: $(BASE_EXE) $(VARIANT_EXES)

# 基线构建（单翻译单元，srad.c 内部 #include 其它 .c 文件）
$(BASE_EXE): $(BASE_SRC)
>-$(CC) $(CFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

# 变体构建：EX1_optimized_codes/srad_%.c -> EX1_optimized_codes/srad_<cc>_%
$(KERNEL_DIR)/$(BENCH_NAME)_$(COMPILER_TAG)_%: $(KERNEL_DIR)/$(SRC_BASENAME)_%.c
>-$(CC) $(CFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

clean:
>rm -f $(KERNEL_DIR)/$(BENCH_NAME)_gcc $(KERNEL_DIR)/$(BENCH_NAME)_gcc_* $(KERNEL_DIR)/$(BENCH_NAME)_clang $(KERNEL_DIR)/$(BENCH_NAME)_clang_*
>rm -f $(KERNEL_DIR)/$(SRC_BASENAME)_gcc $(KERNEL_DIR)/$(SRC_BASENAME)_gcc_* $(KERNEL_DIR)/$(SRC_BASENAME)_clang $(KERNEL_DIR)/$(SRC_BASENAME)_clang_* 2>/dev/null || true