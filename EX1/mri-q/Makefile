## mri-q Makefile
## 说明：
## - main.c 使用 "computeQ.cc"，现从 EX1_optimized_codes 目录读取。
## - 支持多个 computeQ 变体（computeQ_*.cc），按编译器生成：
##   g++/gcc -> mri-q_gcc_<suffix>；clang++/clang -> mri-q_clang_<suffix>
## - 源码含有 `extern "C"`（见 file.c），统一用 C++ 编译器编译以避免兼容问题。

CXX      ?= g++
CXXFLAGS ?= -O3 -ffast-math -funroll-loops
# 屏蔽告警（如需开启请移除）
CXXFLAGS += -w
LDFLAGS  ?= -lm
INCLUDES := -IEX1_optimized_codes -I.

TARGET := mri-q
OBJS   := main.o file.o parboil.o

# 变体源与派生名称
COMPUTEQ_DIR := EX1_optimized_codes
VARIANT_SRC  := $(wildcard $(COMPUTEQ_DIR)/computeQ_*.cc)
VARIANTS     := $(patsubst $(COMPUTEQ_DIR)/computeQ_%.cc,%, $(VARIANT_SRC))
BUILD_DIR    := .variant_build

# 根据编译器推断标签（gcc/clang）
COMPILER_TAG := gcc
ifneq (,$(findstring clang,$(CXX)))
COMPILER_TAG := clang
endif

# Baseline 使用带编译器标签的名字，如 mri-q_gcc
BASE_EXE := $(COMPUTEQ_DIR)/$(TARGET)_$(COMPILER_TAG)
VARIANT_EXES := $(foreach v,$(VARIANTS),$(COMPUTEQ_DIR)/$(TARGET)_$(COMPILER_TAG)_$(v))

all: $(BASE_EXE) $(VARIANT_EXES)

$(BASE_EXE): $(OBJS)
	-$(CXX) $(OBJS) $(LDFLAGS) -o $@

# baseline：从 EX1_optimized_codes/computeQ.cc 读取
main.o: main.c file.h parboil.h $(COMPUTEQ_DIR)/computeQ.cc
	-$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

file.o: file.c file.h
	-$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

parboil.o: parboil.c parboil.h
	-$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# 变体构建：为每个 computeQ_%.cc 复制到独立目录，确保 include 命中对应 computeQ.cc
$(BUILD_DIR)/%/computeQ.cc: $(COMPUTEQ_DIR)/computeQ_%.cc
	@mkdir -p $(dir $@)
	cp -f $< $@

main_%.o: main.c file.h parboil.h $(BUILD_DIR)/%/computeQ.cc
	-$(CXX) $(CXXFLAGS) -I$(BUILD_DIR)/$* -I. -c main.c -o $@

$(COMPUTEQ_DIR)/$(TARGET)_$(COMPILER_TAG)_%: main_%.o file.o parboil.o
	-$(CXX) $^ $(LDFLAGS) -o $@

clean:
	rm -f $(OBJS) main_*.o
	rm -f $(COMPUTEQ_DIR)/$(TARGET)_gcc $(COMPUTEQ_DIR)/$(TARGET)_gcc_* $(COMPUTEQ_DIR)/$(TARGET)_clang $(COMPUTEQ_DIR)/$(TARGET)_clang_*
	rm -rf $(BUILD_DIR)

.PHONY: all clean
