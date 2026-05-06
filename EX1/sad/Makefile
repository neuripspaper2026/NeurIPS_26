# 默认使用 gcc，可在命令行用 CC=clang 切换到 Clang
MAKEFLAGS += --keep-going
# 可选：输出按目标归组，便于看日志
MAKEFLAGS += --output-sync=target
CC        ?= gcc
CFLAGS    ?= -O3
LDFLAGS   ?= -lm -lpthread
SRC_DIR    = EX1_optimized_codes
INCLUDES   = -I. -I$(SRC_DIR)

# 根据编译器确定目标名后缀：_gcc 或 _clang
CC_BASENAME := $(notdir $(CC))
CC_TAG      := gcc
ifeq ($(findstring clang,$(CC_BASENAME)),clang)
  CC_TAG := clang
endif

# 公共源文件（不生成 .o，直接参与最终编译/链接）
COMMON_SRCS := file.c image.c parboil.c main.c

# 变体：EX1_optimized_codes/sad_cpu_*.c（如 sad_cpu_gpt_v1.c）
VARIANT_SRCS := $(wildcard $(SRC_DIR)/sad_cpu_*.c)
VARIANTS     := $(patsubst $(SRC_DIR)/sad_cpu_%.c,%,$(VARIANT_SRCS))

# 目标：基础版本 + 各变体版本，产物都放到 EX1_optimized_codes/
BASE_TARGET     := $(SRC_DIR)/sad_$(CC_TAG)
VARIANT_TARGETS := $(addprefix $(SRC_DIR)/sad_$(CC_TAG)_, $(VARIANTS))
TARGETS         := $(BASE_TARGET) $(VARIANT_TARGETS)

.PHONY: all clean
all: $(TARGETS)

# 基础版本：使用 EX1_optimized_codes/sad_cpu.c
$(SRC_DIR)/sad_$(CC_TAG): $(COMMON_SRCS) $(SRC_DIR)/sad_cpu.c
	$(CC) $(CFLAGS) $(INCLUDES) $^ $(LDFLAGS) -o $@

# 变体版本：使用 EX1_optimized_codes/sad_cpu_<variant>.c
$(SRC_DIR)/sad_$(CC_TAG)_%: $(COMMON_SRCS) $(SRC_DIR)/sad_cpu_%.c
	$(CC) $(CFLAGS) $(INCLUDES) $^ $(LDFLAGS) -o $@

clean:
	rm -f $(TARGETS)