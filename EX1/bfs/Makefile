.RECIPEPREFIX = >

# Compiler and flags
CXX ?= g++
CXXFLAGS ?=
CXXFLAGS += -O3
LDFLAGS = -lm

# Original single-file build variables (kept for optional legacy target)
SRCS = bfs.cpp
OBJS = $(SRCS:.cpp=.o)
EXE = bfs

# Variant build settings
KERNEL_DIR := EX1_optimized_codes
TARGET := bfs

# Baseline source preference: use optimized copy if present, else local
BASE_SRC_LOCAL := bfs.cpp
BASE_SRC_OPT := $(KERNEL_DIR)/bfs.cpp
BASE_KERNEL_SRC := $(if $(wildcard $(BASE_SRC_OPT)),$(BASE_SRC_OPT),$(BASE_SRC_LOCAL))

# Discover variant sources like EX1_optimized_codes/bfs_*.cpp
VARIANT_SRC := $(wildcard $(KERNEL_DIR)/bfs_*.cpp)
VARIANTS := $(patsubst $(KERNEL_DIR)/bfs_%.cpp,%,${VARIANT_SRC})

# Determine compiler tag for naming
ifneq (,$(findstring clang,$(CXX)))
COMPILER_TAG := clang
else
COMPILER_TAG := gcc
endif

# Output executables
BASE_EXE := $(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)
VARIANT_EXES := $(foreach v,$(VARIANTS),$(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)_$(v))

# Default target builds baseline and all variants
all: $(BASE_EXE) $(VARIANT_EXES)

# Baseline build (single-file)
$(BASE_EXE): $(BASE_KERNEL_SRC)
>-$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

# Variant builds (single-file per variant)
$(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)_%: $(KERNEL_DIR)/bfs_%.cpp
>-$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

# Optional legacy targets (not part of 'all')
$(EXE): $(OBJS)
>$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
>$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: clean
clean:
>rm -f $(EXE) $(OBJS) output.txt
>rm -f $(KERNEL_DIR)/$(TARGET)_gcc $(KERNEL_DIR)/$(TARGET)_gcc_* \
>      $(KERNEL_DIR)/$(TARGET)_clang $(KERNEL_DIR)/$(TARGET)_clang_*