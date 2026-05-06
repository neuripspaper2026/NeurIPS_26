CC      ?= gcc
OMP_FLAGS ?= -fopenmp
CFLAGS  ?= -O3
CFLAGS  += $(OMP_FLAGS)
LDFLAGS ?= -lm
LDFLAGS += $(OMP_FLAGS)
TIMING_DEFS := -DBENCHMARK_TIMING -DBENCHMARK_RESET_FUNC=reset_heartwall_kernel_time -DBENCHMARK_GET_FUNC=get_heartwall_kernel_time

AVI_DIR=../../common_rodinia/avi
INCLUDES_BASE = -I$(AVI_DIR)

TARGET := heartwall
KERNEL_DIR := EX2_optimized_codes
BUILD_DIR := .variant_build

# 发现变体：EX2_optimized_codes/kernel_*.c
VARIANT_SRC := $(wildcard $(KERNEL_DIR)/kernel_*.c)
VARIANTS    := $(patsubst $(KERNEL_DIR)/kernel_%.c,%, $(VARIANT_SRC))

# 编译器标签
COMPILER_TAG := gcc
ifneq (,$(findstring clang,$(CC)))
COMPILER_TAG := clang
endif

# 基线：若存在 EX2_optimized_codes/kernel.c，用其作为内联实现，否则用根目录 kernel.c
BASE_KERNEL := $(KERNEL_DIR)/kernel.c
ifeq ($(wildcard $(BASE_KERNEL)),)
BASE_KERNEL := kernel.c
endif

# 产物输出到 EX2_optimized_codes/
BASE_EXE     := $(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)
VARIANT_EXES := $(foreach v,$(VARIANTS),$(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)_$(v))

.RECIPEPREFIX =>

.PHONY: all clean

all: $(BASE_EXE) $(VARIANT_EXES)

# 基线构建：在变体构建目录复制 heartwall.c 与基线 kernel.c，并从该目录编译
$(BUILD_DIR)/base/heartwall.c: heartwall.c
>-mkdir -p $(dir $@)
>-cp -f $< $@

$(BUILD_DIR)/base/kernel.c: $(BASE_KERNEL)
>-mkdir -p $(dir $@)
>-cp -f $< $@

$(BASE_EXE): $(BUILD_DIR)/base/heartwall.c $(BUILD_DIR)/base/kernel.c $(AVI_DIR)/avilib.o $(AVI_DIR)/avimod.o
>-$(CC) $(CFLAGS) $(TIMING_DEFS) $(INCLUDES_BASE) -I. -I$(BUILD_DIR)/base $(BUILD_DIR)/base/heartwall.c $(AVI_DIR)/avilib.o $(AVI_DIR)/avimod.o -o $@ $(LDFLAGS)

# 变体构建：为每个 kernel_%.c 生成独立目录与二进制
$(BUILD_DIR)/%/heartwall.c: heartwall.c
>-mkdir -p $(dir $@)
>-cp -f $< $@

$(BUILD_DIR)/%/kernel.c: $(KERNEL_DIR)/kernel_%.c
>-mkdir -p $(dir $@)
>-cp -f $< $@

$(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)_%: $(BUILD_DIR)/%/heartwall.c $(BUILD_DIR)/%/kernel.c $(AVI_DIR)/avilib.o $(AVI_DIR)/avimod.o
>-$(CC) $(CFLAGS) $(TIMING_DEFS) $(INCLUDES_BASE) -I. -I$(BUILD_DIR)/$* $(BUILD_DIR)/$*/heartwall.c $(AVI_DIR)/avilib.o $(AVI_DIR)/avimod.o -o $@ $(LDFLAGS)

$(AVI_DIR)/avilib.o:
>$(MAKE) -C $(AVI_DIR) avilib.o

$(AVI_DIR)/avimod.o:
>$(MAKE) -C $(AVI_DIR) avimod.o

clean:
>rm -f $(KERNEL_DIR)/$(TARGET)_gcc $(KERNEL_DIR)/$(TARGET)_gcc_* \
>      $(KERNEL_DIR)/$(TARGET)_clang $(KERNEL_DIR)/$(TARGET)_clang_* \
>      output.txt
>rm -rf $(BUILD_DIR)
>$(MAKE) -C $(AVI_DIR) clean