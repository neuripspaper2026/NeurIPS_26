CC      ?= gcc
OMP_FLAGS ?= -fopenmp
CFLAGS  ?= -O3
CFLAGS  += $(OMP_FLAGS)
LDFLAGS ?= -lm -lpthread
LDFLAGS += $(OMP_FLAGS)
INCLUDES = -I. -IEX2_optimized_codes

TARGET := tpacf

# 变体源与派生名称（从 EX1_optimized_codes 目录发现 model_compute_cpu_*.c）
KERNEL_DIR   := EX2_optimized_codes
VARIANT_SRC  := $(wildcard $(KERNEL_DIR)/model_compute_cpu_*.c)
VARIANTS     := $(patsubst $(KERNEL_DIR)/model_compute_cpu_%.c,%, $(VARIANT_SRC))

# 编译器标签（gcc/clang）
COMPILER_TAG := gcc
ifneq (,$(findstring clang,$(CC)))
COMPILER_TAG := clang
endif

# 输出可执行文件全部放在 EX1_optimized_codes/
# Baseline 使用带编译器标签的名字，如 tpacf_gcc
BASE_EXE     := $(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)
VARIANT_EXES := $(foreach v,$(VARIANTS),$(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)_$(v))

all: $(BASE_EXE) $(VARIANT_EXES)

#---------------------- 基线：直接编译链接，不生成中间 .o ----------------------
$(BASE_EXE): main.c args.c model_io.c parboil.c $(KERNEL_DIR)/model_compute_cpu.c
	-$(CC) $(CFLAGS) $(INCLUDES) $^ $(LDFLAGS) -o $@

#---------------------- 变体：直接编译链接对应 kernel ----------------------
$(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)_%: main.c args.c model_io.c parboil.c $(KERNEL_DIR)/model_compute_cpu_%.c
	-$(CC) $(CFLAGS) $(INCLUDES) $^ $(LDFLAGS) -o $@

clean:
	rm -f $(KERNEL_DIR)/$(TARGET)_gcc $(KERNEL_DIR)/$(TARGET)_gcc_* $(KERNEL_DIR)/$(TARGET)_clang $(KERNEL_DIR)/$(TARGET)_clang_*

.PHONY: all clean
