# 默认使用 gcc/g++；可通过 CC=clang CXX=clang++ 切换
CC      ?= gcc
CXX     ?= g++
CFLAGS  ?= -O3 -w
CXXFLAGS?= -O3 -w
LDFLAGS ?= -lm -lstdc++
INCLUDES = -I. -IEX1_optimized_codes

TARGET = sgemm
OBJS   = io.o parboil.o

# 源目录与变体发现
KERNEL_DIR   := EX1_optimized_codes
VARIANT_SRC  := $(wildcard $(KERNEL_DIR)/sgemm_kernel_*.cc)
VARIANTS     := $(patsubst $(KERNEL_DIR)/sgemm_kernel_%.cc,%, $(VARIANT_SRC))
BUILD_DIR    := .variant_build

# 编译器标签自动识别
COMPILER_TAG := gcc
ifneq (,$(findstring clang,$(CXX)))
COMPILER_TAG := clang
endif

# 产物路径（全部放入 EX1_optimized_codes）
# Baseline 使用带编译器标签的名字，如 sgemm_gcc
BASE_EXE     := $(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)
VARIANT_EXES := $(foreach v,$(VARIANTS),$(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)_$(v))

all: $(BASE_EXE) $(VARIANT_EXES)

#---------------------- 基线构建（使用 EX1_optimized_codes/sgemm_kernel.cc） ----------------------
$(BUILD_DIR)/base/sgemm_kernel.cc: $(KERNEL_DIR)/sgemm_kernel.cc
	@mkdir -p $(dir $@)
	cp -f $< $@

main_base.o: main.cc $(BUILD_DIR)/base/sgemm_kernel.cc
	-$(CXX) $(CXXFLAGS) -I$(BUILD_DIR)/base -I. -c main.cc -o $@

$(BASE_EXE): main_base.o $(OBJS)
	-$(CXX) $^ $(LDFLAGS) -o $@

#---------------------- 变体构建（匹配 sgemm_kernel_%.cc） ----------------------
$(BUILD_DIR)/%/sgemm_kernel.cc: $(KERNEL_DIR)/sgemm_kernel_%.cc
	@mkdir -p $(dir $@)
	cp -f $< $@

main_%.o: main.cc $(BUILD_DIR)/%/sgemm_kernel.cc
	-$(CXX) $(CXXFLAGS) -I$(BUILD_DIR)/$* -I. -c main.cc -o $@

$(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)_%: main_%.o $(OBJS)
	-$(CXX) $^ $(LDFLAGS) -o $@

#---------------------- 公共对象 ----------------------
io.o: io.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c io.cc -o $@

parboil.o: parboil.c parboil.h
	$(CC) $(CFLAGS) $(INCLUDES) -c parboil.c -o $@

clean:
	rm -f main_base.o main_*.o $(OBJS)
	rm -f $(KERNEL_DIR)/$(TARGET)_gcc $(KERNEL_DIR)/$(TARGET)_gcc_* $(KERNEL_DIR)/$(TARGET)_clang $(KERNEL_DIR)/$(TARGET)_clang_*
	rm -rf $(BUILD_DIR)

.PHONY: all clean
