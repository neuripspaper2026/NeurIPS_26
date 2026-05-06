KERN=stencil
ALG=stencil3d
TARGET=stencil-3d-mach

CC ?= gcc
OMP_FLAGS ?= -fopenmp
CFLAGS?=-O3 -Wall -Wno-unused-label
CFLAGS += $(OMP_FLAGS)
LDFLAGS ?=
LDFLAGS += $(OMP_FLAGS)
TIMING_DEFS := -DBENCHMARK_TIMING -DBENCHMARK_RESET_FUNC=reset_stencil_3d_kernel_time -DBENCHMARK_GET_FUNC=get_stencil_3d_kernel_time

OPT_DIR=EX2_optimized_codes
KERNEL_SRC := $(OPT_DIR)/$(KERN).c
SUPPORT_SRCS := local_support.c ../../common_MachSuite/support.c ../../common_MachSuite/harness.c
SRCS=$(KERNEL_SRC) $(SUPPORT_SRCS)
FILES=$(SRCS) $(KERN).h ../../common_MachSuite/support.h

# Output directories
CORR_DIR=EX5_correctness

# Dynamic compiler detection
ifneq (,$(findstring clang,$(CC)))
COMPILER_TAG := clang
else
COMPILER_TAG := gcc
endif

# Baseline executable
BASE_EXE=$(OPT_DIR)/$(TARGET)_$(COMPILER_TAG)

# Variant sources
VARIANT_SRCS=$(wildcard $(OPT_DIR)/$(KERN)_*.c)
VARIANT_EXES=$(patsubst $(OPT_DIR)/$(KERN)_%.c,$(OPT_DIR)/$(TARGET)_$(COMPILER_TAG)_%,$(VARIANT_SRCS))

.PHONY: all variants baseline run generate hls clean

all: baseline variants

baseline: $(BASE_EXE)

variants: $(VARIANT_EXES)

# Ensure output directories exist
$(OPT_DIR):
	mkdir -p $(OPT_DIR)
$(CORR_DIR):
	mkdir -p $(CORR_DIR)

# Baseline build
$(BASE_EXE): $(OPT_DIR) $(CORR_DIR) $(FILES) ../../common_MachSuite/harness.c
	-$(CC) $(CFLAGS) $(TIMING_DEFS) -I../../common_MachSuite \
		-DOUTPUT_FILE_NAME=\"$(CORR_DIR)/output_$(COMPILER_TAG).data\" \
		-o $@ $(KERNEL_SRC) local_support.c ../../common_MachSuite/support.c ../../common_MachSuite/harness.c $(LDFLAGS)

# Pattern rule for variant files
$(OPT_DIR)/$(TARGET)_$(COMPILER_TAG)_%: $(OPT_DIR)/$(KERN)_%.c $(OPT_DIR) $(CORR_DIR) local_support.c ../../common_MachSuite/support.c ../../common_MachSuite/harness.c
	-$(CC) $(CFLAGS) $(TIMING_DEFS) -I../../common_MachSuite \
		-DOUTPUT_FILE_NAME=\"$(CORR_DIR)/output_$(COMPILER_TAG)_$*.data\" \
		-o $@ $< local_support.c ../../common_MachSuite/support.c ../../common_MachSuite/harness.c $(LDFLAGS)

run: baseline input.data check.data
	$(BASE_EXE) input.data check.data

generate: $(FILES) generate.c
	-$(CC) $(CFLAGS) -I../../common_MachSuite -o generate $(SRCS) generate.c $(LDFLAGS)
	-./generate

hls: $(KERNEL_SRC) $(KERN).h
	vivado_hls hls.tcl

clean:
	rm -f $(KERN) generate output.data
	rm -f $(BASE_EXE) $(VARIANT_EXES)
	rm -f $(CORR_DIR)/output_$(COMPILER_TAG).data $(CORR_DIR)/output_*_*.data
