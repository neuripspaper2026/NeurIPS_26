.RECIPEPREFIX := >

CXX ?= g++
OMP_FLAGS ?= -fopenmp
CXXFLAGS ?=
CXXFLAGS += -O3
CXXFLAGS += $(OMP_FLAGS)
LDFLAGS = -lm
LDFLAGS += $(OMP_FLAGS)
INCLUDES = -I.

# Directories and targets
KERNEL_DIR := EX2_optimized_codes
TARGET := bilateral

# Baseline source selection (prefer optimized copy if present)
BASE_SRC_OPT := $(KERNEL_DIR)/bilateralFilter-cpu.cpp
BASE_SRC_LOCAL := bilateralFilter-cpu.cpp
BASE_SRC := $(if $(wildcard $(BASE_SRC_OPT)),$(BASE_SRC_OPT),$(BASE_SRC_LOCAL))

# Fixed utility sources
UTIL_SRCS := bmploader.cpp

# Variant discovery (optimized codes use 'bilateralFilter-cpu_*.cpp')
VARIANT_SRC := $(wildcard $(KERNEL_DIR)/bilateralFilter-cpu_*.cpp)
VARIANTS := $(patsubst $(KERNEL_DIR)/bilateralFilter-cpu_%.cpp,%,${VARIANT_SRC})

# Compiler tag for naming
ifneq (,$(findstring clang,$(CXX)))
COMPILER_TAG := clang
else
COMPILER_TAG := gcc
endif

# Executables
BASE_EXE := $(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)
VARIANT_EXES := $(foreach v,$(VARIANTS),$(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)_$(v))

.PHONY: all clean

# Build baseline and all variants (no .o files)
all: $(BASE_EXE) $(VARIANT_EXES)

# Baseline: compile and link in one step
$(BASE_EXE): $(BASE_SRC) $(UTIL_SRCS)
> -$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

# Variant executables
$(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)_%: $(KERNEL_DIR)/bilateralFilter-cpu_%.cpp $(UTIL_SRCS)
> -$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

clean:
> $(RM) $(KERNEL_DIR)/$(TARGET)_gcc $(KERNEL_DIR)/$(TARGET)_clang
> $(RM) $(KERNEL_DIR)/$(TARGET)_gcc_* $(KERNEL_DIR)/$(TARGET)_clang_*
> $(RM) output.dat