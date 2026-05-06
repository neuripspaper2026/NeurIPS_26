CC      ?= gcc
OMP_FLAGS ?= -fopenmp
CFLAGS  ?= -O3
CFLAGS  += $(OMP_FLAGS)
LDFLAGS ?= -lm
LDFLAGS += $(OMP_FLAGS)
INCLUDES = -I. -IEX2_optimized_codes

TARGET := lavaMD
KERNEL_DIR := EX2_optimized_codes

# 发现变体：EX2_optimized_codes/kernel_cpu_*.c
VARIANT_SRC := $(wildcard $(KERNEL_DIR)/kernel_cpu_*.c)
VARIANTS    := $(patsubst $(KERNEL_DIR)/kernel_cpu_%.c,%, $(VARIANT_SRC))

# 编译器标签（gcc/clang）
COMPILER_TAG := gcc
ifneq (,$(findstring clang,$(CC)))
COMPILER_TAG := clang
endif

# 基线 kernel 源：优先 EX2_optimized_codes/kernel_cpu.c，否则使用仓库根目录
BASE_KERNEL_SRC := $(KERNEL_DIR)/kernel_cpu.c
ifeq ($(wildcard $(BASE_KERNEL_SRC)),)
BASE_KERNEL_SRC := kernel_cpu.c
endif

# 产物全部输出到 EX2_optimized_codes/
# Baseline 使用带编译器标签的名字，如 lavaMD_gcc
BASE_EXE     := $(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)
VARIANT_EXES := $(foreach v,$(VARIANTS),$(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)_$(v))

.PHONY: all clean

all: $(BASE_EXE) $(VARIANT_EXES)

# 基线构建：lavaMD.c + (基线)kernel_cpu.c + utils
$(BASE_EXE): lavaMD.c $(BASE_KERNEL_SRC) util/num/num.c util/timer/timer.c
	-$(CC) $(CFLAGS) $(INCLUDES) lavaMD.c $(BASE_KERNEL_SRC) util/num/num.c util/timer/timer.c -o $@ $(LDFLAGS)

# 变体构建：指定 kernel_cpu_%.c + 其余文件
$(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)_%: lavaMD.c $(KERNEL_DIR)/kernel_cpu_%.c util/num/num.c util/timer/timer.c
	-$(CC) $(CFLAGS) $(INCLUDES) lavaMD.c $(KERNEL_DIR)/kernel_cpu_$*.c util/num/num.c util/timer/timer.c -o $@ $(LDFLAGS)

clean:
	rm -f $(KERNEL_DIR)/$(TARGET)_gcc $(KERNEL_DIR)/$(TARGET)_gcc_* $(KERNEL_DIR)/$(TARGET)_clang $(KERNEL_DIR)/$(TARGET)_clang_* output.txt *.o kernel_cpu.o