# 使用 gcc（默认），可在命令行 CC=clang 切换到 clang
CC         ?= gcc
CFLAGS     ?= -O3
LDFLAGS    ?= -lm -lpthread
SRC_DIR     = EX1_optimized_codes
INCLUDES    = -I. -I$(SRC_DIR)

# 编译器标签：_gcc 或 _clang，用于目标名后缀
CC_BASENAME := $(notdir $(CC))
CC_TAG      := gcc
ifeq ($(findstring clang,$(CC_BASENAME)),clang)
  CC_TAG := clang
endif

# 变体源与变体名（如 llama_v1）
VARIANT_SRCS := $(wildcard $(SRC_DIR)/cutcpu_*.c)
VARIANTS     := $(patsubst $(SRC_DIR)/cutcpu_%.c,%,$(VARIANT_SRCS))

# 公共源码（不生成 .o，直接参与最终编译/链接）
COMMON_SRCS := main.c excl.c readatom.c output.c parboil.c

# 目标：基础版本 + 各变体版本，产物都放到 EX1_optimized_codes/
BASE_TARGET     := $(SRC_DIR)/cutcp_$(CC_TAG)
VARIANT_TARGETS := $(addprefix $(SRC_DIR)/cutcp_$(CC_TAG)_, $(VARIANTS))
TARGETS         := $(BASE_TARGET) $(VARIANT_TARGETS)

.PHONY: all clean
all: $(TARGETS)

# 基础版本：使用 EX1_optimized_codes/cutcpu.c
$(SRC_DIR)/cutcp_$(CC_TAG): $(COMMON_SRCS) $(SRC_DIR)/cutcpu.c
	-$(CC) $(CFLAGS) $(INCLUDES) $^ $(LDFLAGS) -o $@

# 变体版本：使用 EX1_optimized_codes/cutcpu_<variant>.c
$(SRC_DIR)/cutcp_$(CC_TAG)_%: $(COMMON_SRCS) $(SRC_DIR)/cutcpu_%.c
	-$(CC) $(CFLAGS) $(INCLUDES) $^ $(LDFLAGS) -o $@

clean:
	rm -f $(TARGETS)