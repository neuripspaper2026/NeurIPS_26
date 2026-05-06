.RECIPEPREFIX = >
CC      ?= gcc
OMP_FLAGS ?= -fopenmp
CFLAGS  ?= -O3
CFLAGS  += $(OMP_FLAGS)
LDFLAGS ?= -lm
LDFLAGS += $(OMP_FLAGS)
INCLUDES = -Icommon -IEX2_optimized_codes

TARGET := lud
KERNEL_DIR := EX2_optimized_codes

# 发现变体：EX2_optimized_codes/lud-omp_*.c（替换并行实现）
VARIANT_SRC := $(wildcard $(KERNEL_DIR)/lud-omp_*.c)
VARIANTS    := $(patsubst $(KERNEL_DIR)/lud-omp_%.c,%, $(VARIANT_SRC))

# 编译器标签（gcc/clang）
COMPILER_TAG := gcc
ifneq (,$(findstring clang,$(CC)))
COMPILER_TAG := clang
endif

# 基线并行源：优先使用 EX2_optimized_codes/lud-omp.c，否则使用仓库根目录的 lud-omp.c
BASE_OMP_SRC := $(KERNEL_DIR)/lud-omp.c
ifeq ($(wildcard $(BASE_OMP_SRC)),)
BASE_OMP_SRC := lud-omp.c
endif

# 产物全部输出到 EX2_optimized_codes/
# Baseline 使用带编译器标签的名字，如 lud_gcc
BASE_EXE     := $(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)
VARIANT_EXES := $(foreach v,$(VARIANTS),$(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)_$(v))

.PHONY: all clean

all: $(BASE_EXE) $(VARIANT_EXES)

# 基线构建：lud.c + (基线)lud-omp.c + common/common.c（直接编译，不生成 .o 文件）
$(BASE_EXE): lud.c $(BASE_OMP_SRC) common/common.c
>-$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

# 变体构建：指定 lud-omp_%.c + 其余文件（直接编译，不生成 .o 文件）
$(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)_%: $(KERNEL_DIR)/lud-omp_%.c lud.c common/common.c
>-$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

clean:
>rm -f $(KERNEL_DIR)/$(TARGET)_gcc $(KERNEL_DIR)/$(TARGET)_gcc_* $(KERNEL_DIR)/$(TARGET)_clang $(KERNEL_DIR)/$(TARGET)_clang_*