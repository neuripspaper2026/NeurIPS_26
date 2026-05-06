.RECIPEPREFIX := >
CC      ?= gcc
OMP_FLAGS ?= -fopenmp
CFLAGS  ?= -O3
CFLAGS  += $(OMP_FLAGS)
LDFLAGS ?= -lm
LDFLAGS += $(OMP_FLAGS)
INCLUDES = -IEX2_optimized_codes

TARGET := kmeans
KERNEL_DIR := EX2_optimized_codes

# 发现变体：EX2_optimized_codes/kmeans-clustering_*.c
VARIANT_SRC := $(wildcard $(KERNEL_DIR)/kmeans-clustering_*.c)
VARIANTS    := $(patsubst $(KERNEL_DIR)/kmeans-clustering_%.c,%, $(VARIANT_SRC))

# 编译器标签（gcc/clang）
COMPILER_TAG := gcc
ifneq (,$(findstring clang,$(CC)))
COMPILER_TAG := clang
endif

# 基线 clustering 源：优先 EX2_optimized_codes/kmeans-clustering.c，否则使用仓库根目录
BASE_CLUSTER_SRC := $(KERNEL_DIR)/kmeans-clustering.c
ifeq ($(wildcard $(BASE_CLUSTER_SRC)),)
BASE_CLUSTER_SRC := kmeans-clustering.c
endif

# 产物全部输出到 EX2_optimized_codes/
BASE_EXE     := $(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)
VARIANT_EXES := $(foreach v,$(VARIANTS),$(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)_$(v))

.PHONY: all clean

all: $(BASE_EXE) $(VARIANT_EXES)

# 基线构建：kmeans.c + cluster.c + getopt.c + (基线)kmeans-clustering.c
$(BASE_EXE): kmeans.c cluster.c getopt.c $(BASE_CLUSTER_SRC)
>-$(CC) $(CFLAGS) $(INCLUDES) kmeans.c cluster.c getopt.c $(BASE_CLUSTER_SRC) -o $@ $(LDFLAGS)

# 变体构建：指定 kmeans-clustering_%.c + 其余文件
$(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)_%: kmeans.c cluster.c getopt.c $(KERNEL_DIR)/kmeans-clustering_%.c
>-$(CC) $(CFLAGS) $(INCLUDES) kmeans.c cluster.c getopt.c $(KERNEL_DIR)/kmeans-clustering_$*.c -o $@ $(LDFLAGS)

clean:
>rm -f $(KERNEL_DIR)/$(TARGET)_gcc $(KERNEL_DIR)/$(TARGET)_clang $(KERNEL_DIR)/$(TARGET)_gcc_* $(KERNEL_DIR)/$(TARGET)_clang_* output.txt *.o



# .RECIPEPREFIX := >
# CC = gcc
# CFLAGS = -O3
# LDFLAGS = -lm

# SRCS = cluster.c getopt.c kmeans.c kmeans-clustering.c
# OBJS = $(SRCS:.c=.o)
# EXE  = kmeans

# all: $(EXE)

# $(EXE): $(OBJS)
# >$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# %.o: %.c
# >$(CC) $(CFLAGS) -c $< -o $@

# clean:
# >rm -f $(EXE) $(OBJS) output.txt