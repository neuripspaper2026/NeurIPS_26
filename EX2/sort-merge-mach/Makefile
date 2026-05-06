KERN=sort
ALG=merge
TARGET=sort-merge-mach

CC ?= gcc
OMP_FLAGS ?= -fopenmp
CFLAGS?=-O3 -Wall -Wno-unused-label
CFLAGS += $(OMP_FLAGS)
LDFLAGS ?=
LDFLAGS += $(OMP_FLAGS)
TIMING_DEFS := -DBENCHMARK_TIMING -DBENCHMARK_RESET_FUNC=reset_sort_merge_kernel_time -DBENCHMARK_GET_FUNC=get_sort_merge_kernel_time

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

# Variant sources and executables (only sort_*.c; no plain opt)
VARIANT_SRCS=$(wildcard $(OPT_DIR)/$(KERN)_*.c)
VARIANT_EXES=$(patsubst $(OPT_DIR)/$(KERN)_%.c,$(OPT_DIR)/$(TARGET)_$(COMPILER_TAG)_%,$(VARIANT_SRCS))

.PHONY: all baseline variants run generate hls clean

all: baseline variants

baseline: $(BASE_EXE)

# Build baseline into OPT_DIR and inject output filename in CORR_DIR
$(BASE_EXE): $(KERNEL_SRC) $(SUPPORT_SRCS)
	@mkdir -p $(OPT_DIR) $(CORR_DIR)
	-$(CC) $(CFLAGS) $(TIMING_DEFS) -I../../common_MachSuite \
		-DOUTPUT_FILE_NAME=\"$(CORR_DIR)/output_$(COMPILER_TAG).data\" \
		-o $@ $(KERNEL_SRC) $(SUPPORT_SRCS) $(LDFLAGS)

variants: $(VARIANT_EXES)

# Build each variant: EX2_optimized_codes/sort_<suffix>.c -> EX2_optimized_codes/sort_gcc_<suffix>
$(OPT_DIR)/$(TARGET)_$(COMPILER_TAG)_%: $(OPT_DIR)/$(KERN)_%.c local_support.c ../../common_MachSuite/support.c ../../common_MachSuite/harness.c
	@mkdir -p $(OPT_DIR) $(CORR_DIR)
	-$(CC) $(CFLAGS) $(TIMING_DEFS) -I../../common_MachSuite \
		-DOUTPUT_FILE_NAME=\"$(CORR_DIR)/output_$(COMPILER_TAG)_$*.data\" \
		-o $@ $< local_support.c ../../common_MachSuite/support.c ../../common_MachSuite/harness.c $(LDFLAGS)

run: $(BASE_EXE) input.data check.data
	@echo "Running baseline executable: $(BASE_EXE)"
	./$(BASE_EXE) input.data check.data
	@echo "Output written to $(CORR_DIR)/output_$(COMPILER_TAG).data"

generate: $(FILES) generate.c
	-$(CC) $(CFLAGS) -I../../common_MachSuite -o generate $(SRCS) generate.c $(LDFLAGS)
	-./generate

hls: $(KERNEL_SRC) $(KERN).h
	vivado_hls hls.tcl

clean:
	rm -f $(BASE_EXE) $(VARIANT_EXES)
	rm -f $(CORR_DIR)/output_$(COMPILER_TAG).data $(CORR_DIR)/output_$(COMPILER_TAG)_*.data
	rm -f $(KERN) generate output.data
