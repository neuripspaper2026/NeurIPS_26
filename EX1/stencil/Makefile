# 默认使用 gcc；可在命令行 CC=clang 切换为 clang
CC      ?= gcc
CFLAGS  ?= -O3
LDFLAGS ?= -lm -lpthread
INCLUDES= -I. -IEX1_optimized_codes

TARGET := stencil

# 源文件
COMMON_SRCS := file.c parboil.c
# 基线 main 在当前目录，kernels 在 EX1_optimized_codes

# 变体发现：支持 main_*.c 与 kernels_*.c 两类（任选其一或同时存在）
KERNEL_DIR  := EX1_optimized_codes
MAIN_VARIANT_SRC    := $(wildcard $(KERNEL_DIR)/main_*.c)
KERNELS_VARIANT_SRC := $(wildcard $(KERNEL_DIR)/kernels_*.c)
MAIN_VARIANTS       := $(patsubst $(KERNEL_DIR)/main_%.c,%, $(MAIN_VARIANT_SRC))
KERNELS_VARIANTS    := $(patsubst $(KERNEL_DIR)/kernels_%.c,%, $(KERNELS_VARIANT_SRC))

# 编译器标签（gcc/clang）
COMPILER_TAG := gcc
ifneq (,$(findstring clang,$(CC)))
COMPILER_TAG := clang
endif

# 输出可执行文件全部放到 EX1_optimized_codes/
# Baseline 使用带编译器标签的名字，如 stencil_gcc
BASE_EXE := $(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)

# 变体统一输出为 stencil_<cc>_<variant>
VARIANT_NAMES := $(sort $(MAIN_VARIANTS) $(KERNELS_VARIANTS))
VARIANT_EXES  := $(foreach v,$(VARIANT_NAMES),$(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)_$(v))

all: $(BASE_EXE) $(VARIANT_EXES)

#---------------------- 基线（仓库自带 main.c + kernels.c） ----------------------
$(BASE_EXE): main.c $(KERNEL_DIR)/kernels.c file.c parboil.c
	-$(CC) $(CFLAGS) $(INCLUDES) $^ $(LDFLAGS) -o $@

#---------------------- 统一命名的变体规则 ----------------------
define declare_variant
$(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)_$(1): \
	$(if $(filter $(1),$(MAIN_VARIANTS)),$(KERNEL_DIR)/main_$(1).c,main.c) \
	$(if $(filter $(1),$(KERNELS_VARIANTS)),$(KERNEL_DIR)/kernels_$(1).c,$(KERNEL_DIR)/kernels.c) \
	$(COMMON_SRCS)
	-$(CC) $(CFLAGS) $(INCLUDES) $$^ $(LDFLAGS) -o $$@
endef

$(foreach v,$(VARIANT_NAMES),$(eval $(call declare_variant,$(v))))

clean:
	rm -f $(KERNEL_DIR)/$(TARGET)_gcc $(KERNEL_DIR)/$(TARGET)_gcc_* $(KERNEL_DIR)/$(TARGET)_clang $(KERNEL_DIR)/$(TARGET)_clang_*

.PHONY: all clean
