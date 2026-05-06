KERN=spmv
ALG=ellpack
TARGET=spmv-ellpack-mach

CFLAGS?=-O3 -Wall -Wno-unused-label
TIMING_DEFS := -DBENCHMARK_TIMING -DBENCHMARK_RESET_FUNC=reset_spmv_ellpack_kernel_time -DBENCHMARK_GET_FUNC=get_spmv_ellpack_kernel_time

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

# Baseline executable
BASE_EXE=$(OUT_DIR)/$(TARGET)_$(COMPILER_TAG)

# Variant sources and executables (only spmv_*.c; no plain opt)
VARIANT_SRCS=$(wildcard $(OUT_DIR)/$(KERN)_*.c)
VARIANT_EXES=$(patsubst $(OUT_DIR)/$(KERN)_%.c,$(OUT_DIR)/$(TARGET)_$(COMPILER_TAG)_%,$(VARIANT_SRCS))

.PHONY: all baseline variants run generate hls clean

all: baseline variants

baseline: $(BASE_EXE)

# Build baseline into OUT_DIR and inject output filename in CORR_DIR
$(BASE_EXE): $(SRCS) ../../common_MachSuite/harness.c
	@mkdir -p $(OUT_DIR) $(CORR_DIR)
	-$(CC) $(CFLAGS) $(TIMING_DEFS) -I../../common_MachSuite \
		-DOUTPUT_FILE_NAME=\"$(CORR_DIR)/output_$(COMPILER_TAG).data\" \
		-o $@ $(SRCS) ../../common_MachSuite/harness.c

variants: $(VARIANT_EXES)

# Build each variant: EX1_optimized_codes/spmv_<suffix>.c -> EX1_optimized_codes/spmv_gcc_<suffix>
$(OUT_DIR)/$(TARGET)_$(COMPILER_TAG)_%: $(OUT_DIR)/$(KERN)_%.c local_support.c ../../common_MachSuite/support.c ../../common_MachSuite/harness.c
	@mkdir -p $(OUT_DIR) $(CORR_DIR)
	-$(CC) $(CFLAGS) $(TIMING_DEFS) -I../../common_MachSuite \
		-DOUTPUT_FILE_NAME=\"$(CORR_DIR)/output_$(COMPILER_TAG)_$*.data\" \
		-o $@ $< local_support.c ../../common_MachSuite/support.c ../../common_MachSuite/harness.c

run: $(BASE_EXE) input.data check.data
	@echo "Running baseline executable: $(BASE_EXE)"
	./$(BASE_EXE) input.data check.data
	@echo "Output written to $(CORR_DIR)/output_$(COMPILER_TAG).data"

generate: $(FILES) generate.c
	-$(CC) $(CFLAGS) -I../../common_MachSuite -o generate $(SRCS) generate.c
	-./generate

hls: $(OUT_DIR)/$(KERN).c $(KERN).h
	vivado_hls hls.tcl

clean:
	rm -f $(BASE_EXE) $(VARIANT_EXES)
	rm -f $(CORR_DIR)/output_$(COMPILER_TAG).data $(CORR_DIR)/output_$(COMPILER_TAG)_*.data
	rm -f $(KERN) generate output.data
