.RECIPEPREFIX = >
CC ?= gcc -w
CFLAGS ?=
CFLAGS += -O3 -w
LDFLAGS = -lm
INCLUDES = -I.

# Paths
KERNEL_DIR := EX1_optimized_codes
TARGET := b+tree
MAIN := b+tree.c
UTIL := util/timer/timer.c util/num/num.c

# Choose baseline kernel sources (prefer optimized copies if present)
K1_BASE := $(if $(wildcard $(KERNEL_DIR)/kernel-cpu.c),$(KERNEL_DIR)/kernel-cpu.c,kernel/kernel_cpu.c)
K2_BASE := $(if $(wildcard $(KERNEL_DIR)/kernel-cpu-2.c),$(KERNEL_DIR)/kernel-cpu-2.c,kernel/kernel_cpu_2.c)

# Variants discovery (pair by common suffix)
VAR1_SRC := $(wildcard $(KERNEL_DIR)/kernel-cpu_*.c)
VAR2_SRC := $(wildcard $(KERNEL_DIR)/kernel-cpu-2_*.c)
SUFFIX1 := $(patsubst $(KERNEL_DIR)/kernel-cpu_%.c,%,${VAR1_SRC})
SUFFIX2 := $(patsubst $(KERNEL_DIR)/kernel-cpu-2_%.c,%,${VAR2_SRC})
# intersection of SUFFIX1 and SUFFIX2
COMMON_SUFFIX := $(foreach s,$(SUFFIX1),$(if $(filter $(s),$(SUFFIX2)),$(s)))

# Compiler tag
ifneq (,$(findstring clang,$(CC)))
COMPILER_TAG := clang
else
COMPILER_TAG := gcc
endif

# Executables
BASE_EXE := $(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)
PAIRED_EXES := $(foreach s,$(COMMON_SUFFIX),$(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)_$(s))

.PHONY: all clean

# Build baseline and all paired variants (no .o files)
all: $(BASE_EXE) $(PAIRED_EXES)

# Baseline: compile and link in one step
$(BASE_EXE): $(MAIN) $(UTIL) $(K1_BASE) $(K2_BASE)
>-$(CC) $(CFLAGS) $(INCLUDES) $(MAIN) $(UTIL) $(K1_BASE) $(K2_BASE) -o $@ $(LDFLAGS)

# Variant rule: build when both sides have the same suffix
$(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)_%: $(KERNEL_DIR)/kernel-cpu_%.c $(KERNEL_DIR)/kernel-cpu-2_%.c $(MAIN) $(UTIL)
>-$(CC) $(CFLAGS) $(INCLUDES) $(MAIN) $(UTIL) $(KERNEL_DIR)/kernel-cpu_$*.c $(KERNEL_DIR)/kernel-cpu-2_$*.c -o $@ $(LDFLAGS)

clean:
>rm -f $(KERNEL_DIR)/$(TARGET)_gcc $(KERNEL_DIR)/$(TARGET)_gcc_* \
>      $(KERNEL_DIR)/$(TARGET)_clang $(KERNEL_DIR)/$(TARGET)_clang_* \
>      output.txt