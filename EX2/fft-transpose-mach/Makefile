KERN=fft
ALG=transpose
TARGET=fft-transpose-mach

# Compiler settings
CC ?= gcc
OMP_FLAGS ?= -fopenmp
CFLAGS ?= -O3 -Wall -Wno-unused-label
CFLAGS += $(OMP_FLAGS)
LDFLAGS ?= -lm
LDFLAGS += $(OMP_FLAGS)
TIMING_DEFS := -DBENCHMARK_TIMING -DBENCHMARK_RESET_FUNC=reset_fft_transpose_kernel_time -DBENCHMARK_GET_FUNC=get_fft_transpose_kernel_time

# Common sources
OPT_DIR := EX2_optimized_codes
KERNEL_SRC := $(OPT_DIR)/$(KERN).c
SRCS=$(KERNEL_SRC) local_support.c ../../common_MachSuite/support.c
FILES=$(SRCS) $(KERN).h ../../common_MachSuite/support.h

# Output directories
CORR_DIR := EX5_correctness

# Compiler tag for naming
ifneq (,$(findstring clang,$(CC)))
COMPILER_TAG := clang
else
COMPILER_TAG := gcc
endif

# Base source selection
BASE_SRC := $(KERNEL_SRC)

# Support files
SUPPORT_SRCS := local_support.c ../../common_MachSuite/support.c ../../common_MachSuite/harness.c

# Variant discovery (optimized codes use pattern: fft_*.c)
VARIANT_SRC := $(wildcard $(OPT_DIR)/$(KERN)_*.c)
VARIANTS := $(patsubst $(OPT_DIR)/$(KERN)_%.c,%,$(VARIANT_SRC))

# Executables
BASE_EXE := $(OPT_DIR)/$(TARGET)_$(COMPILER_TAG)
VARIANT_EXES := $(foreach v,$(VARIANTS),$(OPT_DIR)/$(TARGET)_$(COMPILER_TAG)_$(v))

.PHONY: all clean baseline variants run generate hls

# Build baseline and all variants
all: baseline variants

baseline: $(BASE_EXE)

variants: $(VARIANT_EXES)

# Ensure output directories exist
$(OPT_DIR):
	@mkdir -p $(OPT_DIR)

$(CORR_DIR):
	@mkdir -p $(CORR_DIR)

# Baseline: compile and link in one step (no .o files)
$(BASE_EXE): $(BASE_SRC) $(SUPPORT_SRCS) | $(OPT_DIR) $(CORR_DIR)
	-$(CC) $(CFLAGS) $(TIMING_DEFS) -I../../common_MachSuite \
		-DOUTPUT_FILE_NAME=\"$(CORR_DIR)/output_$(COMPILER_TAG).data\" \
		-o $@ $(BASE_SRC) $(SUPPORT_SRCS) $(LDFLAGS)

# Variant executables: compile and link in one step (no .o files)
$(OPT_DIR)/$(TARGET)_$(COMPILER_TAG)_%: $(OPT_DIR)/$(KERN)_%.c $(SUPPORT_SRCS) | $(OPT_DIR) $(CORR_DIR)
	-$(CC) $(CFLAGS) $(TIMING_DEFS) -I../../common_MachSuite \
		-DOUTPUT_FILE_NAME=\"$(CORR_DIR)/output_$(COMPILER_TAG)_$*.data\" \
		-o $@ $< $(SUPPORT_SRCS) $(LDFLAGS)

# Generator target (uses -lm for cos/sin)
generate: $(FILES) generate.c
	$(CC) $(CFLAGS) -I../../common_MachSuite -o generate $(SRCS) generate.c $(LDFLAGS)
	./generate

# Run baseline with current input/check files
run: baseline input.data check.data
	$(BASE_EXE) input.data check.data

hls: $(KERNEL_SRC) $(KERN).h
	vivado_hls hls.tcl

clean:
	rm -f $(KERN) generate output.data
	rm -f $(OPT_DIR)/$(TARGET)_gcc $(OPT_DIR)/$(TARGET)_gcc_* \
	      $(OPT_DIR)/$(TARGET)_clang $(OPT_DIR)/$(TARGET)_clang_*
	rm -f $(CORR_DIR)/output_gcc.data $(CORR_DIR)/output_gcc_*.data \
	      $(CORR_DIR)/output_clang.data $(CORR_DIR)/output_clang_*.data
