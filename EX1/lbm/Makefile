# 默认使用 gcc，可在命令行 CC=clang 切换为 clang
CC        ?= gcc
CFLAGS    ?= -O3
LDFLAGS   ?= -lm -lpthread
SRC_DIR    = EX1_optimized_codes
INCLUDES   = -I. -I$(SRC_DIR)   # parboil.h 等头文件路径

# 根据编译器确定目标名后缀：_gcc 或 _clang
CC_BASENAME := $(notdir $(CC))
CC_TAG      := gcc
ifeq ($(findstring clang,$(CC_BASENAME)),clang)
  CC_TAG := clang
endif

# 公共源文件（不生成 .o，直接参与最终编译/链接）
# 原始版本是 lbm.c（在当前目录）
COMMON_SRCS := main.c parboil.c

# 变体源：EX1_optimized_codes/lbm_*.c（如 lbm_llama_v1.c）
VARIANT_SRCS := $(wildcard $(SRC_DIR)/lbm_*.c)
VARIANTS     := $(patsubst $(SRC_DIR)/lbm_%.c,%,$(VARIANT_SRCS))

# 目标：基础版本 + 各变体版本，产物都放到 EX1_optimized_codes/
BASE_TARGET     := $(SRC_DIR)/lbm_$(CC_TAG)
VARIANT_TARGETS := $(addprefix $(SRC_DIR)/lbm_$(CC_TAG)_, $(VARIANTS))
TARGETS         := $(BASE_TARGET) $(VARIANT_TARGETS)

.PHONY: all clean
all: $(TARGETS)

# 基础版本：使用 EX1_optimized_codes/lbm.c
$(SRC_DIR)/lbm_$(CC_TAG): $(COMMON_SRCS) $(SRC_DIR)/lbm.c
	-$(CC) $(CFLAGS) $(INCLUDES) $^ $(LDFLAGS) -o $@

# 变体版本：使用 EX1_optimized_codes/lbm_<variant>.c
$(SRC_DIR)/lbm_$(CC_TAG)_%: $(COMMON_SRCS) $(SRC_DIR)/lbm_%.c
	-$(CC) $(CFLAGS) $(INCLUDES) $^ $(LDFLAGS) -o $@

clean:
	rm -f $(TARGETS)