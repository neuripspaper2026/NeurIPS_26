# 默认使用 gcc；可在命令行 CC=clang 切换为 clang
CC      ?= gcc
OMP_FLAGS ?= -fopenmp
CFLAGS  ?= -O3
CFLAGS  += $(OMP_FLAGS)
LDFLAGS ?= -lm -lpthread
LDFLAGS += $(OMP_FLAGS)
INCLUDES= -I. -IEX2_optimized_codes

TARGET := stencil

# 源文件
COMMON_SRCS := file.c parboil.c
# 基线 main 在当前目录，kernels 在 EX1_optimized_codes

# 变体发现：支持 main_*.c 与 kernels_*.c 两类（任选其一或同时存在）
KERNEL_DIR  := EX2_optimized_codes
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

# 组合变体：
COMBO_VARIANTS        := $(filter $(MAIN_VARIANTS),$(KERNELS_VARIANTS))
MAIN_ONLY_VARIANTS    := $(filter-out $(COMBO_VARIANTS),$(MAIN_VARIANTS))
KERNELS_ONLY_VARIANTS := $(filter-out $(COMBO_VARIANTS),$(KERNELS_VARIANTS))

# 1) 仅 main_%.c -> stencil_<cc>_%
MAIN_EXES := $(foreach v,$(MAIN_ONLY_VARIANTS),$(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)_$(v))
# 2) 仅 kernels_%.c -> stencil_<cc>_%
KERNELS_EXES := $(foreach v,$(KERNELS_ONLY_VARIANTS),$(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)_$(v))
# 3) 同时存在同名 main_foo.c 与 kernels_foo.c -> stencil_<cc>_foo
COMBO_EXES := $(foreach v,$(COMBO_VARIANTS),$(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)_$(v))

all: $(BASE_EXE) $(MAIN_EXES) $(KERNELS_EXES) $(COMBO_EXES)

#---------------------- 基线（仓库自带 main.c + kernels.c） ----------------------
$(BASE_EXE): main.c $(KERNEL_DIR)/kernels.c file.c parboil.c
	-$(CC) $(CFLAGS) $(INCLUDES) $^ $(LDFLAGS) -o $@

#---------------------- 仅 main 变体 ----------------------
define make_main_only_exe
$(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)_$(1): $(KERNEL_DIR)/main_$(1).c $(KERNEL_DIR)/kernels.c file.c parboil.c
	-$$(CC) $$(CFLAGS) $$(INCLUDES) $$^ $$(LDFLAGS) -o $$@
endef
$(foreach v,$(MAIN_ONLY_VARIANTS),$(eval $(call make_main_only_exe,$(v))))

#---------------------- 仅 kernels 变体 ----------------------
define make_kernels_only_exe
$(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)_$(1): main.c $(KERNEL_DIR)/kernels_$(1).c file.c parboil.c
	-$$(CC) $$(CFLAGS) $$(INCLUDES) $$^ $$(LDFLAGS) -o $$@
endef
$(foreach v,$(KERNELS_ONLY_VARIANTS),$(eval $(call make_kernels_only_exe,$(v))))

#---------------------- main 与 kernels 同名组合变体 ----------------------
# 定义模式规则以匹配组合变体
define make_combo_exe
$(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)_$(1): $(KERNEL_DIR)/main_$(1).c $(KERNEL_DIR)/kernels_$(1).c file.c parboil.c
	-$$(CC) $$(CFLAGS) $$(INCLUDES) $$^ $$(LDFLAGS) -o $$@
endef
$(foreach v,$(COMBO_VARIANTS),$(eval $(call make_combo_exe,$(v))))

clean:
	rm -f $(KERNEL_DIR)/$(TARGET)_gcc $(KERNEL_DIR)/$(TARGET)_gcc_* $(KERNEL_DIR)/$(TARGET)_clang $(KERNEL_DIR)/$(TARGET)_clang_*

.PHONY: all clean
