CC      ?= gcc
OMP_FLAGS ?= -fopenmp
CFLAGS  ?= -O3
CFLAGS  += $(OMP_FLAGS)
LDFLAGS ?= -lm
LDFLAGS += $(OMP_FLAGS)
INCLUDES = -I. -IEX2_optimized_codes

TARGET := 3D
KERNEL_DIR := EX2_optimized_codes

# 发现变体：EX2_optimized_codes/3D_*.c
VARIANT_SRC := $(wildcard $(KERNEL_DIR)/3D_*.c)
VARIANTS    := $(patsubst $(KERNEL_DIR)/3D_%.c,%, $(VARIANT_SRC))

# 编译器标签（gcc/clang）
COMPILER_TAG := gcc
ifneq (,$(findstring clang,$(CC)))
COMPILER_TAG := clang
endif

# 基线源码：优先 EX2_optimized_codes/3D.c，否则回退根目录 3D.c
BASE_SRC := $(KERNEL_DIR)/3D.c
ifeq ($(wildcard $(BASE_SRC)),)
BASE_SRC := 3D.c
endif

# 产物全部输出到 EX2_optimized_codes/，可执行名 hotspot3D_<cc>
BASE_EXE     := $(KERNEL_DIR)/hotspot3D_$(COMPILER_TAG)
VARIANT_EXES := $(foreach v,$(VARIANTS),$(KERNEL_DIR)/hotspot3D_$(COMPILER_TAG)_$(v))

.RECIPEPREFIX = >

.PHONY: all clean

all: $(BASE_EXE) $(VARIANT_EXES)

# 基线构建
$(BASE_EXE): $(BASE_SRC)
>-$(CC) $(CFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

# 变体构建：3D_%.c -> hotspot3D_<cc>_%
$(KERNEL_DIR)/hotspot3D_$(COMPILER_TAG)_%: $(KERNEL_DIR)/3D_%.c
>-$(CC) $(CFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

clean:
>rm -f $(KERNEL_DIR)/hotspot3D_gcc $(KERNEL_DIR)/hotspot3D_clang $(KERNEL_DIR)/hotspot3D_gcc_* $(KERNEL_DIR)/hotspot3D_clang_* output.txt *.o