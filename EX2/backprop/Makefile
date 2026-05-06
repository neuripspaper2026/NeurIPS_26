.RECIPEPREFIX = >

CC      ?= gcc
OMP_FLAGS ?= -fopenmp
CFLAGS  ?=
CFLAGS  += -O3
CFLAGS  += $(OMP_FLAGS)
LDFLAGS ?= -lm
LDFLAGS += $(OMP_FLAGS)

# Legacy single-target variables (kept for optional manual build)
EXE  = backprop
OBJS = backprop.o backprop_kernel.o facetrain.o imagenet.o

# Common objects (excluding backprop.c which is variant)
COMMON_SRCS := backprop_kernel.c facetrain.c imagenet.c
COMMON_OBJS := $(COMMON_SRCS:.c=.o)

# Variant build settings
KERNEL_DIR := EX2_optimized_codes
TARGET := backprop

# Prefer optimized copy for baseline if exists
BASE_SRC_LOCAL := backprop.c
BASE_SRC_OPT := $(KERNEL_DIR)/backprop.c
BASE_KERNEL_SRC := $(if $(wildcard $(BASE_SRC_OPT)),$(BASE_SRC_OPT),$(BASE_SRC_LOCAL))

# Discover variant sources under EX2_optimized_codes
VARIANT_SRC := $(wildcard $(KERNEL_DIR)/backprop_*.c)
VARIANTS := $(patsubst $(KERNEL_DIR)/backprop_%.c,%,${VARIANT_SRC})

# Compiler tag for naming
ifneq (,$(findstring clang,$(CC)))
COMPILER_TAG := clang
else
COMPILER_TAG := gcc
endif

# Output executables
BASE_EXE := $(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)
VARIANT_EXES := $(foreach v,$(VARIANTS),$(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)_$(v))

# Default: build baseline and all variants
all: $(BASE_EXE) $(VARIANT_EXES)

# Build common objs once
$(COMMON_OBJS): %.o: %.c
>$(CC) $(CFLAGS) -c $< -o $@

# Baseline build: link common objs with chosen backprop.c
$(BASE_EXE): $(BASE_KERNEL_SRC) $(COMMON_OBJS)
>$(CC) $(CFLAGS) -o $@ $(COMMON_OBJS) $< $(LDFLAGS)

# Variant builds: one exe per variant source
$(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)_%: $(KERNEL_DIR)/backprop_%.c $(COMMON_OBJS)
>$(CC) $(CFLAGS) -o $@ $(COMMON_OBJS) $< $(LDFLAGS)

# Optional legacy target (not part of 'all')
$(EXE): $(OBJS)
>$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)

.PHONY: clean
clean:
>rm -f $(EXE) $(OBJS) output.dat
>rm -f $(BASE_EXE) $(VARIANT_EXES)