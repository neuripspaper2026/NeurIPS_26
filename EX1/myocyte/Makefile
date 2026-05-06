CC      ?= gcc
CFLAGS  ?= -O3
LDFLAGS ?= -lm
INCLUDES = -I. -IEX1_optimized_codes

TARGET := myocyte
KERNEL_DIR := EX1_optimized_codes

# 发现变体：EX1_optimized_codes/myocyte_*.c（单翻译单元，myocyte.c 内部包含其他 .c）
VARIANT_SRC := $(wildcard $(KERNEL_DIR)/$(TARGET)_*.c)
VARIANTS    := $(patsubst $(KERNEL_DIR)/$(TARGET)_%.c,%, $(VARIANT_SRC))

# 支持 master.c 变体（通过 include 覆盖）
MASTER_VARIANT_SRC := $(wildcard $(KERNEL_DIR)/master_*.c)
MASTER_VARIANTS    := $(patsubst $(KERNEL_DIR)/master_%.c,%, $(MASTER_VARIANT_SRC))
BUILD_DIR := .variant_build

# 编译器标签（gcc/clang）
COMPILER_TAG := gcc
ifneq (,$(findstring clang,$(CC)))
COMPILER_TAG := clang
endif

# 基线源码：优先使用 EX1_optimized_codes/myocyte.c，否则回退仓库根目录
BASE_SRC := $(TARGET).c
ifneq (,$(wildcard $(KERNEL_DIR)/$(TARGET).c))
BASE_SRC := $(KERNEL_DIR)/$(TARGET).c
endif

# 产物全部输出到 EX1_optimized_codes/
# Baseline 使用带编译器标签的名字，如 myocyte_gcc
BASE_EXE     := $(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)
VARIANT_EXES := $(foreach v,$(VARIANTS),$(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)_$(v)) \
                $(foreach v,$(MASTER_VARIANTS),$(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)_$(v))

.PHONY: all clean

all: $(BASE_EXE) $(VARIANT_EXES)

# 基线构建（单翻译单元，直接从 .c 编译到可执行文件，不生成 .o 文件）
$(BASE_EXE): $(BASE_SRC)
	-$(CC) $(CFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

# 变体构建：EX1_optimized_codes/myocyte_%.c -> EX1_optimized_codes/myocyte_<cc>_%（直接编译，不生成 .o 文件）
$(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)_%: $(KERNEL_DIR)/$(TARGET)_%.c
	-$(CC) $(CFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

# master_%.c 变体：复制为 .variant_build/%/master.c，并用 include 覆盖构建（直接编译，不生成 .o 文件）
$(BUILD_DIR)/%/master.c: $(KERNEL_DIR)/master_%.c
	-@mkdir -p $(dir $@)
	-cp -f $< $@

$(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)_%: $(BUILD_DIR)/%/master.c $(BASE_SRC)
	-$(CC) $(CFLAGS) $(INCLUDES) -I$(BUILD_DIR)/$* $(BASE_SRC) -o $@ $(LDFLAGS)

clean:
	rm -f $(KERNEL_DIR)/$(TARGET)_gcc $(KERNEL_DIR)/$(TARGET)_gcc_* $(KERNEL_DIR)/$(TARGET)_clang $(KERNEL_DIR)/$(TARGET)_clang_*
	rm -rf $(BUILD_DIR)