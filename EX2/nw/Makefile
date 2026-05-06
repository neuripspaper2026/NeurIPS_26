CC       ?= g++
OMP_FLAGS ?= -fopenmp
CFLAGS   ?= -O3
CFLAGS   += $(OMP_FLAGS)
LDFLAGS  ?= -lm
LDFLAGS  += $(OMP_FLAGS)
INCLUDES := -I. -IEX2_optimized_codes

# 源文件前缀（注意：变量值末尾不要留空格，否则会把目标名拆成多个 token）
SRC_BASENAME := needle
# 基准名（生成的可执行名以此为准）
BENCH_NAME   := nw
KERNEL_DIR := EX2_optimized_codes

# 发现变体：EX1_optimized_codes/needle_*.cpp
VARIANT_SRC := $(wildcard $(KERNEL_DIR)/$(SRC_BASENAME)_*.cpp)
VARIANTS    := $(patsubst $(KERNEL_DIR)/$(SRC_BASENAME)_%.cpp,%, $(VARIANT_SRC))

# 编译器标签派生：命令行指定非空则沿用，否则按 CC 判定；目标命名使用 COMPILER_LABEL，避免命令行传空值时生成空标签目标
COMPILER_LABEL := $(strip $(COMPILER_TAG))
ifeq ($(COMPILER_LABEL),)
  ifneq (,$(findstring clang,$(notdir $(CC))))
    COMPILER_LABEL := clang
  else
    COMPILER_LABEL := gcc
  endif
endif

# 基线源码：优先使用 EX2_optimized_codes/needle.cpp，否则回退仓库根目录
BASE_SRC := $(KERNEL_DIR)/$(SRC_BASENAME).cpp
ifeq ($(wildcard $(BASE_SRC)),)
BASE_SRC := $(SRC_BASENAME).cpp
endif

# 产物全部输出到 EX2_optimized_codes/，可执行名对齐 benchmark（nw_<cc>）
BASE_EXE     := $(KERNEL_DIR)/$(BENCH_NAME)_$(COMPILER_LABEL)
VARIANT_EXES := $(foreach v,$(VARIANTS),$(KERNEL_DIR)/$(BENCH_NAME)_$(COMPILER_LABEL)_$(v))

.PHONY: all clean variants

all: $(BASE_EXE) $(VARIANT_EXES)

variants: $(VARIANT_EXES)

# 基线构建（直接从 .cpp 编译到可执行文件，不生成 .o 文件）
$(BASE_EXE): $(BASE_SRC)
	-$(CC) $(CFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

# 变体构建（单一模式规则，避免多重目标模式冲突）
$(KERNEL_DIR)/$(BENCH_NAME)_$(COMPILER_LABEL)_%: $(KERNEL_DIR)/$(SRC_BASENAME)_%.cpp
	-$(CC) $(CFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(KERNEL_DIR)/$(BENCH_NAME)_gcc $(KERNEL_DIR)/$(BENCH_NAME)_gcc_* $(KERNEL_DIR)/$(BENCH_NAME)_clang $(KERNEL_DIR)/$(BENCH_NAME)_clang_*