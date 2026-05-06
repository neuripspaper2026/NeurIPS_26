CC      ?= gcc
CFLAGS  ?= -O3
LDFLAGS ?= -lm

MESCHACH_DIR=../../common_rodinia/meschach
AVI_DIR=../../common_rodinia/avi

INCLUDES = -I. -I$(MESCHACH_DIR) -I$(AVI_DIR) -IEX1_optimized_codes

TARGET := leukocyte
KERNEL_DIR := EX1_optimized_codes

# 发现变体：EX1_optimized_codes/track-ellipse_*.c
VARIANT_SRC := $(wildcard $(KERNEL_DIR)/track-ellipse_*.c)
VARIANTS    := $(patsubst $(KERNEL_DIR)/track-ellipse_%.c,%, $(VARIANT_SRC))

# 编译器标签（gcc/clang）
COMPILER_TAG := gcc
ifneq (,$(findstring clang,$(CC)))
COMPILER_TAG := clang
endif

# 基线 track 源：优先 EX1_optimized_codes/track-ellipse.c，否则使用仓库根目录
BASE_TRACK_SRC := track-ellipse.c
ifneq (,$(wildcard $(KERNEL_DIR)/track-ellipse.c))
BASE_TRACK_SRC := $(KERNEL_DIR)/track-ellipse.c
endif

# 产物全部输出到 EX1_optimized_codes/
# Baseline 使用带编译器标签的名字，如 leukocyte_gcc
BASE_EXE     := $(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)
VARIANT_EXES := $(foreach v,$(VARIANTS),$(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)_$(v))

.PHONY: all clean

all: $(BASE_EXE) $(VARIANT_EXES)

# 基线构建：leukocyte.c + find_ellipse.c + (基线)track-ellipse.c + misc_math.c（直接编译，不生成 .o 文件）
$(BASE_EXE): leukocyte.c find_ellipse.c $(BASE_TRACK_SRC) misc_math.c $(MESCHACH_DIR)/libmeschach.a $(AVI_DIR)/avilib.o
	-$(CC) $(CFLAGS) $(INCLUDES) leukocyte.c find_ellipse.c $(BASE_TRACK_SRC) misc_math.c $(AVI_DIR)/avilib.o $(MESCHACH_DIR)/libmeschach.a -o $@ $(LDFLAGS)

# 变体构建：指定 track-ellipse_%.c + 其余文件（直接编译，不生成 .o 文件）
$(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)_%: leukocyte.c find_ellipse.c $(KERNEL_DIR)/track-ellipse_%.c misc_math.c $(MESCHACH_DIR)/libmeschach.a $(AVI_DIR)/avilib.o
	-$(CC) $(CFLAGS) $(INCLUDES) leukocyte.c find_ellipse.c $(KERNEL_DIR)/track-ellipse_$*.c misc_math.c $(AVI_DIR)/avilib.o $(MESCHACH_DIR)/libmeschach.a -o $@ $(LDFLAGS)

$(AVI_DIR)/avilib.o:
	$(MAKE) -C $(AVI_DIR) avilib.o

$(MESCHACH_DIR)/libmeschach.a:
	cd $(MESCHACH_DIR); ./configure --with-all
	$(MAKE) -C $(MESCHACH_DIR)

clean:
	rm -f $(KERNEL_DIR)/$(TARGET)_gcc $(KERNEL_DIR)/$(TARGET)_gcc_* $(KERNEL_DIR)/$(TARGET)_clang $(KERNEL_DIR)/$(TARGET)_clang_*
	-$(MAKE) -C $(MESCHACH_DIR) realclean
	$(MAKE) -C $(AVI_DIR) clean