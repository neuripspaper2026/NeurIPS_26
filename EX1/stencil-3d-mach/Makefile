KERN=stencil
ALG=stencil3d
TARGET=stencil-3d-mach

CFLAGS?=-O3 -Wall -Wno-unused-label
TIMING_DEFS := -DBENCHMARK_TIMING -DBENCHMARK_RESET_FUNC=reset_stencil_3d_kernel_time -DBENCHMARK_GET_FUNC=get_stencil_3d_kernel_time

SRCS=$(OUT_DIR)/$(KERN).c local_support.c ../../common_MachSuite/support.c
FILES=$(SRCS) $(KERN).h ../../common_MachSuite/support.h

# Output directories
OUT_DIR=EX1_optimized_codes
CORR_DIR=EX5_correctness

# Dynamic compiler detection
ifneq (,$(findstring clang,$(CC)))
COMPILER_TAG := clang
else
COMPILER_TAG := gcc
endif

# Baseline executable (built from root stencil.c)
BASE_EXE=$(OUT_DIR)/$(TARGET)_$(COMPILER_TAG)

# Variant sources in OUT_DIR (only suffixed files, no special 'opt')
VARIANT_SRCS=$(wildcard $(OUT_DIR)/$(KERN)_*.c)
VARIANT_EXES=$(patsubst $(OUT_DIR)/$(KERN)_%.c,$(OUT_DIR)/$(TARGET)_$(COMPILER_TAG)_%,$(VARIANT_SRCS))

.PHONY: all variants baseline run generate hls clean

all: baseline variants

baseline: $(BASE_EXE)

variants: $(VARIANT_EXES)

# Ensure output directories exist
$(OUT_DIR):
	mkdir -p $(OUT_DIR)
$(CORR_DIR):
	mkdir -p $(CORR_DIR)

# Baseline build
$(BASE_EXE): $(OUT_DIR) $(CORR_DIR) $(FILES) ../../common_MachSuite/harness.c
	-$(CC) $(CFLAGS) $(TIMING_DEFS) -I../../common_MachSuite \
		-DOUTPUT_FILE_NAME=\"$(CORR_DIR)/output_$(COMPILER_TAG).data\" \
		-o $@ $(OUT_DIR)/$(KERN).c local_support.c ../../common_MachSuite/support.c ../../common_MachSuite/harness.c

# Pattern rule for variant files: EX1_optimized_codes/stencil_<suffix>.c
$(OUT_DIR)/$(TARGET)_$(COMPILER_TAG)_%: $(OUT_DIR)/$(KERN)_%.c $(OUT_DIR) $(CORR_DIR) local_support.c ../../common_MachSuite/support.c ../../common_MachSuite/harness.c
	-$(CC) $(CFLAGS) $(TIMING_DEFS) -I../../common_MachSuite \
		-DOUTPUT_FILE_NAME=\"$(CORR_DIR)/output_$(COMPILER_TAG)_$*.data\" \
		-o $@ $< local_support.c ../../common_MachSuite/support.c ../../common_MachSuite/harness.c

run: baseline input.data check.data
	$(BASE_EXE) input.data check.data

generate: $(FILES) generate.c
	-$(CC) $(CFLAGS) -I../../common_MachSuite -o generate $(SRCS) generate.c
	-./generate

hls: $(OUT_DIR)/$(KERN).c $(KERN).h
	vivado_hls hls.tcl

clean:
	rm -f $(KERN) generate output.data
	rm -f $(BASE_EXE) $(VARIANT_EXES)
	rm -f $(CORR_DIR)/output_$(COMPILER_TAG).data $(CORR_DIR)/output_*_*.data
