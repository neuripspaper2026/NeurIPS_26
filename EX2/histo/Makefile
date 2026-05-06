# 默认使用 gcc，可在命令行用 CC=clang 切换到 Clang
CC        ?= gcc
OMP_FLAGS ?= -fopenmp
CFLAGS    ?= -O3
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

# 公共源文件（除 main.c/变体外）
COMMON_SRCS := util.c parboil.c

# 变体：EX1_optimized_codes/main_*.c（如 main_llama_v1.c）
VARIANT_SRCS := $(wildcard $(SRC_DIR)/main_*.c)
VARIANTS     := $(patsubst $(SRC_DIR)/main_%.c,%,$(VARIANT_SRCS))

# 目标：基础版本 + 各变体版本，产物都放到 EX1_optimized_codes/
BASE_TARGET     := $(SRC_DIR)/histo_$(CC_TAG)
VARIANT_TARGETS := $(addprefix $(SRC_DIR)/histo_$(CC_TAG)_, $(VARIANTS))
TARGETS         := $(BASE_TARGET) $(VARIANT_TARGETS)

.PHONY: all clean
all: $(TARGETS)

# 基础版本：使用 EX1_optimized_codes/main.c
$(SRC_DIR)/histo_$(CC_TAG): $(COMMON_SRCS) $(SRC_DIR)/main.c
	-$(CC) $(CFLAGS) $(INCLUDES) $^ $(LDFLAGS) -o $@

# 变体版本：使用 main_<variant>.c
$(SRC_DIR)/histo_$(CC_TAG)_%: $(COMMON_SRCS) $(SRC_DIR)/main_%.c
	-$(CC) $(CFLAGS) $(INCLUDES) $^ $(LDFLAGS) -o $@

clean:
	rm -f $(TARGETS)
