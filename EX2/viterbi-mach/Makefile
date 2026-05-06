# Compiler config
KERN=viterbi
ALG=viterbi
TARGET=viterbi-mach

CC ?= gcc
OMP_FLAGS ?= -fopenmp
CFLAGS?=-O3 -Wall -Wno-unused-label
CFLAGS += $(OMP_FLAGS)
LDFLAGS ?=
LDFLAGS += $(OMP_FLAGS)
TIMING_DEFS := -DBENCHMARK_TIMING -DBENCHMARK_RESET_FUNC=reset_viterbi_kernel_time -DBENCHMARK_GET_FUNC=get_viterbi_kernel_time

# Output directories
OPT_DIR=EX2_optimized_codes
CORR_DIR=EX5_correctness

# Sources
SRCS=$(OPT_DIR)/$(KERN).c local_support.c ../../common_MachSuite/support.c
FILES=$(SRCS) $(KERN).h ../../common_MachSuite/support.h

# Dynamic compiler detection
ifneq (,$(findstring clang,$(CC)))
COMPILER_TAG := clang
else
COMPILER_TAG := gcc
endif

# Baseline executable (built from root viterbi.c)
BASE_EXE=$(OPT_DIR)/$(TARGET)_$(COMPILER_TAG)

# Variant sources in OUT_DIR (only suffixed files)
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
		-o $@ $(OPT_DIR)/$(KERN).c local_support.c ../../common_MachSuite/support.c ../../common_MachSuite/harness.c $(LDFLAGS)

# Pattern rule for variant files: EX1_optimized_codes/viterbi_<suffix>.c
$(OPT_DIR)/$(TARGET)_$(COMPILER_TAG)_%: $(OPT_DIR)/$(KERN)_%.c $(OPT_DIR) $(CORR_DIR) local_support.c ../../common_MachSuite/support.c ../../common_MachSuite/harness.c
	-$(CC) $(CFLAGS) $(TIMING_DEFS) -I../../common_MachSuite \
		-DOUTPUT_FILE_NAME=\"$(CORR_DIR)/output_$(COMPILER_TAG)_$*.data\" \
		-o $@ $< local_support.c ../../common_MachSuite/support.c ../../common_MachSuite/harness.c $(LDFLAGS)

run: baseline input.data check.data
	$(BASE_EXE) input.data check.data

generate: $(FILES) generate.c
	-$(CC) $(CFLAGS) -I../../common_MachSuite -o generate $(SRCS) generate.c -lm $(LDFLAGS)
	-./generate

hls: $(OPT_DIR)/$(KERN).c $(KERN).h
	vivado_hls hls.tcl

clean:
	rm -f $(KERN) generate output.data
	rm -f $(BASE_EXE) $(VARIANT_EXES)
	rm -f $(CORR_DIR)/output_$(COMPILER_TAG).data $(CORR_DIR)/output_*_*.data
