KERN=fft
ALG=strided
TARGET=fft-strided-mach

# Compiler settings
CC ?= gcc
CFLAGS ?= -O3 -Wall -Wno-unused-label
TIMING_DEFS := -DBENCHMARK_TIMING -DBENCHMARK_RESET_FUNC=reset_fft_strided_kernel_time -DBENCHMARK_GET_FUNC=get_fft_strided_kernel_time

# Common sources
SRCS=$(KERN).c local_support.c ../../common_MachSuite/support.c
FILES=$(SRCS) $(KERN).h ../../common_MachSuite/support.h

# Output directories
OUT_DIR := EX1_optimized_codes
CORR_DIR := EX5_correctness

# Compiler tag for naming
ifneq (,$(findstring clang,$(CC)))
COMPILER_TAG := clang
else
COMPILER_TAG := gcc
endif

# Base source selection (prefer optimized copy if present)
BASE_SRC_OPT := $(OUT_DIR)/$(KERN).c
BASE_SRC_LOCAL := $(KERN).c
BASE_SRC := $(if $(wildcard $(BASE_SRC_OPT)),$(BASE_SRC_OPT),$(BASE_SRC_LOCAL))

# Support files
SUPPORT_SRCS := local_support.c ../../common_MachSuite/support.c ../../common_MachSuite/harness.c

# Variant discovery (optimized codes use pattern: fft_*.c)
VARIANT_SRC := $(wildcard $(OUT_DIR)/$(KERN)_*.c)
VARIANTS := $(patsubst $(OUT_DIR)/$(KERN)_%.c,%,$(VARIANT_SRC))

# Executables
BASE_EXE := $(OUT_DIR)/$(TARGET)_$(COMPILER_TAG)
VARIANT_EXES := $(foreach v,$(VARIANTS),$(OUT_DIR)/$(TARGET)_$(COMPILER_TAG)_$(v))

.PHONY: all clean baseline variants run generate hls

# Build baseline and all variants
all: baseline variants

baseline: $(BASE_EXE)

variants: $(VARIANT_EXES)

# Ensure output directories exist
$(OUT_DIR):
	@mkdir -p $(OUT_DIR)

$(CORR_DIR):
	@mkdir -p $(CORR_DIR)

# Baseline: compile and link in one step (no .o files)
$(BASE_EXE): $(BASE_SRC) $(SUPPORT_SRCS) | $(OUT_DIR) $(CORR_DIR)
	-$(CC) $(CFLAGS) $(TIMING_DEFS) -I../../common_MachSuite \
		-DOUTPUT_FILE_NAME=\"$(CORR_DIR)/output_$(COMPILER_TAG).data\" \
		-o $@ $(BASE_SRC) $(SUPPORT_SRCS) -lm

# Variant executables: compile and link in one step (no .o files)
$(OUT_DIR)/$(TARGET)_$(COMPILER_TAG)_%: $(OUT_DIR)/$(KERN)_%.c $(SUPPORT_SRCS) | $(OUT_DIR) $(CORR_DIR)
	-$(CC) $(CFLAGS) $(TIMING_DEFS) -I../../common_MachSuite \
		-DOUTPUT_FILE_NAME=\"$(CORR_DIR)/output_$(COMPILER_TAG)_$*.data\" \
		-o $@ $< $(SUPPORT_SRCS) -lm

# Generator target (uses -lm for cos/sin)
generate: $(FILES) generate.c
	$(CC) $(CFLAGS) -I../../common_MachSuite -o generate $(SRCS) generate.c -lm
	./generate

# Run baseline with current input/check files
run: baseline input.data check.data
	$(BASE_EXE) input.data check.data

hls: $(KERN).c $(KERN).h
	vivado_hls hls.tcl

clean:
	rm -f $(KERN) generate output.data
	rm -f $(OUT_DIR)/$(TARGET)_gcc $(OUT_DIR)/$(TARGET)_gcc_* \
	      $(OUT_DIR)/$(TARGET)_clang $(OUT_DIR)/$(TARGET)_clang_*
	rm -f $(CORR_DIR)/output_gcc.data $(CORR_DIR)/output_gcc_*.data \
	      $(CORR_DIR)/output_clang.data $(CORR_DIR)/output_clang_*.data
