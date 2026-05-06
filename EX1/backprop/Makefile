.RECIPEPREFIX = >

CC      ?= gcc
CFLAGS  ?=
CFLAGS  += -O3
LDFLAGS = -lm

# Legacy single-target variables (kept for optional manual build)
EXE  = backprop
OBJS =

# Common sources (excluding backprop.c which is variant)
COMMON_SRCS := backprop_kernel.c facetrain.c imagenet.c

# Variant build settings
KERNEL_DIR := EX1_optimized_codes
TARGET := backprop

# Prefer optimized copy for baseline if exists
BASE_SRC_LOCAL := backprop.c
BASE_SRC_OPT := $(KERNEL_DIR)/backprop.c
BASE_KERNEL_SRC := $(if $(wildcard $(BASE_SRC_OPT)),$(BASE_SRC_OPT),$(BASE_SRC_LOCAL))

# Discover variant sources under EX1_optimized_codes
VARIANT_SRC := $(wildcard $(KERNEL_DIR)/backprop_*.c)
VARIANTS := $(patsubst $(KERNEL_DIR)/backprop_%.c,%,${VARIANT_SRC})

# Compiler tag for naming
ifneq (,$(findstring clang,$(CC)))
COMPILER_TAG := clang
else
COMPILER_TAG := gcc
endif

# Output executables
BASE_EXE := $(KERNEL_DIR)/$(TARGET)
VARIANT_EXES := $(foreach v,$(VARIANTS),$(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)_$(v))

# Default: build baseline and all variants
all: $(BASE_EXE) $(VARIANT_EXES)

# Baseline build: compile/link directly from sources
$(BASE_EXE): $(BASE_KERNEL_SRC) $(COMMON_SRCS)
>$(CC) $(CFLAGS) -o $@ $(COMMON_SRCS) $(BASE_KERNEL_SRC) $(LDFLAGS)

# Variant builds: one exe per variant source (no .o intermediates)
$(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)_%: $(KERNEL_DIR)/backprop_%.c $(COMMON_SRCS)
>$(CC) $(CFLAGS) -o $@ $(COMMON_SRCS) $< $(LDFLAGS)

# Optional legacy target (not part of 'all')
$(EXE): $(OBJS)
>$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)

.PHONY: clean
clean:
>rm -f $(EXE) output.dat
>rm -f $(BASE_EXE) $(VARIANT_EXES)