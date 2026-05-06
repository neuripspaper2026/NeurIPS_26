KERN=backprop
ALG=backprop
TARGET=$(KERN)

# Compiler settings
CC ?= gcc
CFLAGS ?= -O3 -Wall -Wno-unused-label

# Directories
KERNEL_DIR := EX1_optimized_codes
OUT_DIR := EX5_correctness

# Source files
SRCS=$(KERN).c local_support.c ../../common_MachSuite/support.c
FILES=$(SRCS) $(KERN).h ../../common_MachSuite/support.h

# Compiler tag for naming
ifneq (,$(findstring clang,$(CC)))
COMPILER_TAG := clang
else
COMPILER_TAG := gcc
endif

# Base source selection (prefer optimized copy if present)
BASE_SRC_OPT := $(KERNEL_DIR)/$(KERN).c
BASE_SRC_LOCAL := $(KERN).c
BASE_SRC := $(if $(wildcard $(BASE_SRC_OPT)),$(BASE_SRC_OPT),$(BASE_SRC_LOCAL))

# Support files
SUPPORT_SRCS := local_support.c ../../common_MachSuite/support.c ../../common_MachSuite/harness.c

# Variant discovery (optimized codes use pattern: backprop_*.c)
VARIANT_SRC := $(wildcard $(KERNEL_DIR)/$(KERN)_*.c)
VARIANTS := $(patsubst $(KERNEL_DIR)/$(KERN)_%.c,%,$(VARIANT_SRC))

# Executables
BASE_EXE := $(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)
VARIANT_EXES := $(foreach v,$(VARIANTS),$(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)_$(v))

.PHONY: all clean bins run generate hls

# Build baseline and all variants
all: bins

bins: $(BASE_EXE) $(VARIANT_EXES)

# Baseline: compile and link in one step (no .o files)
$(BASE_EXE): $(BASE_SRC) $(SUPPORT_SRCS)
	@mkdir -p $(KERNEL_DIR)
	-$(CC) $(CFLAGS) -I../../common_MachSuite -o $@ $^ -lm

# Variant executables: compile and link in one step (no .o files)
$(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)_%: $(KERNEL_DIR)/$(KERN)_%.c $(SUPPORT_SRCS)
	-$(CC) $(CFLAGS) -I../../common_MachSuite -o $@ $^ -lm

# Legacy single-binary build (backward compatibility, not part of 'all')
$(KERN): $(FILES) ../../common_MachSuite/harness.c
	$(CC) $(CFLAGS) -I../../common_MachSuite -o $(KERN) $(SRCS) ../../common_MachSuite/harness.c -lm

run: $(BASE_EXE) input.data check.data
	$(BASE_EXE) input.data check.data

generate: $(FILES) generate.c
	$(CC) $(CFLAGS) -I../../common_MachSuite -o generate $(SRCS) generate.c -lm
	./generate

hls: $(KERN).c $(KERN).h
	vivado_hls hls.tcl

clean:
	rm -f $(KERN) generate output.data
	rm -f $(KERNEL_DIR)/$(TARGET)_gcc $(KERNEL_DIR)/$(TARGET)_gcc_* \
	      $(KERNEL_DIR)/$(TARGET)_clang $(KERNEL_DIR)/$(TARGET)_clang_*
