# 可用 GCC/Clang；命令行可通过 CC=clang 切换
CC      ?= gcc
OMP_FLAGS ?= -fopenmp
CFLAGS  ?= -O3
CFLAGS  += $(OMP_FLAGS)
LDFLAGS ?= -lm -lpthread
LDFLAGS += $(OMP_FLAGS)
TOOLS   := ./common_src/convert-dataset
INCLUDES= -I. -I$(TOOLS) -IEX2_optimized_codes

TARGET := spmv

# 公共对象（不含 main.o，便于为变体替换 main）
COMMON_OBJS := file.o convert_dataset.o mmio.o parboil.o
OBJS        := main.o $(COMMON_OBJS)

# 变体源与派生名称（从 EX1_optimized_codes 目录下发现 main_*.c）
KERNEL_DIR  := EX2_optimized_codes
VARIANT_SRC := $(wildcard $(KERNEL_DIR)/main_*.c)
VARIANTS    := $(patsubst $(KERNEL_DIR)/main_%.c,%, $(VARIANT_SRC))

# 编译器标签（gcc/clang）
COMPILER_TAG := gcc
ifneq (,$(findstring clang,$(CC)))
COMPILER_TAG := clang
endif

# 输出可执行文件全部放在 EX1_optimized_codes/
# Baseline 使用带编译器标签的名字，如 spmv_gcc
BASE_EXE     := $(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)
VARIANT_EXES := $(foreach v,$(VARIANTS),$(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)_$(v))

all: $(BASE_EXE) $(VARIANT_EXES)

# 基线构建：使用 EX1_optimized_codes/main.c
$(BASE_EXE): $(OBJS)
	-$(CC) $(OBJS) $(LDFLAGS) -o $@

# 变体 main：从 EX1_optimized_codes/main_%.c 编译
main_%.o: $(KERNEL_DIR)/main_%.c file.h $(TOOLS)/convert_dataset.h
	-$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# 每个变体单独链接为 spmv_<compiler>_% 可执行
$(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)_%: main_%.o $(COMMON_OBJS)
	-$(CC) $^ $(LDFLAGS) -o $@

# 公共对象规则
main.o: $(KERNEL_DIR)/main.c file.h $(TOOLS)/convert_dataset.h
	-$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

file.o: file.c file.h
	-$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

convert_dataset.o: $(TOOLS)/convert_dataset.c $(TOOLS)/convert_dataset.h $(TOOLS)/mmio.h
	-$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

mmio.o: $(TOOLS)/mmio.c $(TOOLS)/mmio.h
	-$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

parboil.o: parboil.c parboil.h
	-$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS) main_*.o
	rm -f $(KERNEL_DIR)/$(TARGET)_gcc $(KERNEL_DIR)/$(TARGET)_gcc_* $(KERNEL_DIR)/$(TARGET)_clang $(KERNEL_DIR)/$(TARGET)_clang_*

.PHONY: all clean
