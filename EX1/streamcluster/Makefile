CXX      ?= g++
CXXFLAGS ?= -O3
LDFLAGS  ?= -lpthread -lm
TARGET   := streamcluster

# 变体与输出目录
KERNEL_DIR   := EX1_optimized_codes
VARIANT_SRC  := $(wildcard $(KERNEL_DIR)/$(TARGET)_*.cpp)
VARIANTS     := $(patsubst $(KERNEL_DIR)/$(TARGET)_%.cpp,%, $(VARIANT_SRC))

# 编译器标签（gcc/clang）
COMPILER_TAG := gcc
ifneq (,$(findstring clang,$(CXX)))
COMPILER_TAG := clang
endif

# 基线源码：优先使用 EX1_optimized_codes/streamcluster.cpp，否则回退仓库根目录
BASE_SRC := streamcluster.cpp
ifneq (,$(wildcard $(KERNEL_DIR)/$(TARGET).cpp))
BASE_SRC := $(KERNEL_DIR)/$(TARGET).cpp
endif

# 产物全部输出到 EX1_optimized_codes/
# Baseline 使用带编译器标签的名字，如 streamcluster_gcc
BASE_EXE     := $(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)
VARIANT_EXES := $(foreach v,$(VARIANTS),$(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)_$(v))

.PHONY: all clean

all: $(BASE_EXE) $(VARIANT_EXES)

# 基线构建
$(BASE_EXE): $(BASE_SRC)
	-$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# 变体构建：EX1_optimized_codes/streamcluster_%.cpp -> EX1_optimized_codes/streamcluster_<cc>_%
$(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)_%: $(KERNEL_DIR)/$(TARGET)_%.cpp
	-$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(KERNEL_DIR)/$(TARGET)_gcc $(KERNEL_DIR)/$(TARGET)_gcc_* $(KERNEL_DIR)/$(TARGET)_clang $(KERNEL_DIR)/$(TARGET)_clang_*