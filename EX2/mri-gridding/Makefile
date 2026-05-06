# 默认使用 gcc，可在命令行用 CC=clang 切换
CC        ?= gcc
OMP_FLAGS ?= -fopenmp
CFLAGS    ?= -O3
CFLAGS    += -w
CFLAGS    += $(OMP_FLAGS)
LDFLAGS   ?= -lm -lpthread
LDFLAGS   += $(OMP_FLAGS)
SRC_DIR    = EX2_optimized_codes
INCLUDES   = -I. -I$(SRC_DIR)

# 根据编译器确定目标名后缀：_gcc 或 _clang
CC_BASENAME := $(notdir $(CC))
CC_TAG      := gcc
ifeq ($(findstring clang,$(CC_BASENAME)),clang)
  CC_TAG := clang
endif

# 公共源文件（不生成 .o，直接参与编译/链接）
COMMON_SRCS := main.c parboil.c

# 原始版本：EX1_optimized_codes/CPU_kernels.c
BASE_SRC := $(SRC_DIR)/CPU_kernels.c

# 变体版本：EX1_optimized_codes/CPU_kernels_*.c（如 CPU_kernels_llama_v1.c）
VARIANT_SRCS := $(wildcard $(SRC_DIR)/CPU_kernels_*.c)
VARIANTS     := $(patsubst $(SRC_DIR)/CPU_kernels_%.c,%,$(VARIANT_SRCS))

# 目标：基础版本 + 各变体版本，产物放到 EX1_optimized_codes/
BASE_TARGET     := $(SRC_DIR)/mri-gridding_$(CC_TAG)
VARIANT_TARGETS := $(addprefix $(SRC_DIR)/mri-gridding_$(CC_TAG)_, $(VARIANTS))
TARGETS         := $(BASE_TARGET) $(VARIANT_TARGETS)

.PHONY: all clean
all: $(TARGETS)

# 基础版本：使用 EX1_optimized_codes/CPU_kernels.c
$(SRC_DIR)/mri-gridding_$(CC_TAG): $(COMMON_SRCS) $(BASE_SRC)
	-$(CC) $(CFLAGS) $(INCLUDES) $^ $(LDFLAGS) -o $@

# 变体版本：使用 EX1_optimized_codes/CPU_kernels_<variant>.c
$(SRC_DIR)/mri-gridding_$(CC_TAG)_%: $(COMMON_SRCS) $(SRC_DIR)/CPU_kernels_%.c
	-$(CC) $(CFLAGS) $(INCLUDES) $^ $(LDFLAGS) -o $@

clean:
	rm -f $(TARGETS)