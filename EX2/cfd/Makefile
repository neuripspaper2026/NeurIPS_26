CC      ?= gcc
CXX     ?= g++
OMP_FLAGS ?= -fopenmp
CFLAGS  ?= -O3
CFLAGS  += $(OMP_FLAGS)
CXXFLAGS ?= -O3
CXXFLAGS += $(OMP_FLAGS)
LDFLAGS ?= -lm
LDFLAGS += $(OMP_FLAGS)
INCLUDES = -I. -IEX2_optimized_codes

TARGET := euler3d-cpu
KERNEL_DIR := EX2_optimized_codes

# 发现变体：EX2_optimized_codes/euler3d-cpu_*.cpp
VARIANT_SRC := $(wildcard $(KERNEL_DIR)/$(TARGET)_*.cpp)
VARIANTS    := $(patsubst $(KERNEL_DIR)/$(TARGET)_%.cpp,%, $(VARIANT_SRC))

# 编译器标签（gcc/clang）
COMPILER_TAG := gcc
ifneq (,$(findstring clang,$(CXX)))
COMPILER_TAG := clang
endif

# 基线源码：优先 EX2_optimized_codes/euler3d-cpu.cpp，否则回退根目录
BASE_SRC := $(KERNEL_DIR)/$(TARGET).cpp
ifeq ($(wildcard $(BASE_SRC)),)
BASE_SRC := $(TARGET).cpp
endif

# 产物全部输出到 EX2_optimized_codes/，可执行名对齐 benchmark（cfd_<cc>）
BASE_EXE     := $(KERNEL_DIR)/cfd_$(COMPILER_TAG)
VARIANT_EXES := $(foreach v,$(VARIANTS),$(KERNEL_DIR)/cfd_$(COMPILER_TAG)_$(v))

.PHONY: all clean

all: $(BASE_EXE) $(VARIANT_EXES)

# 基线构建
$(BASE_EXE): $(BASE_SRC)
	-$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

# 变体构建：euler3d-cpu_%.cpp -> cfd_<cc>_%
$(KERNEL_DIR)/cfd_$(COMPILER_TAG)_%: $(KERNEL_DIR)/$(TARGET)_%.cpp
	-$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(KERNEL_DIR)/cfd_gcc $(KERNEL_DIR)/cfd_gcc_* \
	      $(KERNEL_DIR)/cfd_clang $(KERNEL_DIR)/cfd_clang_* \
	      density momentum density_energy