CXX      ?= g++
CXXFLAGS ?= -O3 -w
LDFLAGS  ?= -lm
INCLUDES = -I. -IEX1_optimized_codes

TARGET := hotspot
KERNEL_DIR := EX1_optimized_codes

# 发现变体：EX1_optimized_codes/hotspot_*.cpp
VARIANT_SRC := $(wildcard $(KERNEL_DIR)/$(TARGET)_*.cpp)
VARIANTS    := $(patsubst $(KERNEL_DIR)/$(TARGET)_%.cpp,%, $(VARIANT_SRC))

# 编译器标签（gcc/clang）
COMPILER_TAG := gcc
ifneq (,$(findstring clang,$(CXX)))
COMPILER_TAG := clang
endif

# 基线源码：优先 EX1_optimized_codes/hotspot.cpp，否则回退根目录 hotspot.cpp
BASE_SRC := $(TARGET).cpp
ifneq (,$(wildcard $(KERNEL_DIR)/$(TARGET).cpp))
BASE_SRC := $(KERNEL_DIR)/$(TARGET).cpp
endif

# 产物全部输出到 EX1_optimized_codes/
BASE_EXE     := $(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)
VARIANT_EXES := $(foreach v,$(VARIANTS),$(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)_$(v))

.PHONY: all clean

all: $(BASE_EXE) $(VARIANT_EXES)

# 基线构建
$(BASE_EXE): $(BASE_SRC)
	-$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

# 变体构建：hotspot_%.cpp -> hotspot_<cc>_%
$(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)_%: $(KERNEL_DIR)/$(TARGET)_%.cpp
	-$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(KERNEL_DIR)/$(TARGET)_gcc $(KERNEL_DIR)/$(TARGET)_gcc_* \
	      $(KERNEL_DIR)/$(TARGET)_clang $(KERNEL_DIR)/$(TARGET)_clang_* \
	      output.txt *.o