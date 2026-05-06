KERN=aes
ALG=aes

CFLAGS?=-O3 -Wall -Wno-unused-label
TIMING_DEFS := -DBENCHMARK_TIMING -DBENCHMARK_RESET_FUNC=reset_aes_kernel_time -DBENCHMARK_GET_FUNC=get_aes_kernel_time

# Directories and targets
KERNEL_DIR := EX1_optimized_codes
TARGET := $(KERN)
OUT_DIR := EX5_correctness

# Baseline source selection (prefer optimized copy if present)
BASE_SRC_OPT := $(KERNEL_DIR)/aes.c
BASE_SRC_LOCAL := aes.c
BASE_SRC := $(if $(wildcard $(BASE_SRC_OPT)),$(BASE_SRC_OPT),$(BASE_SRC_LOCAL))

# Utility and harness sources (fixed)
UTIL_SRCS := local_support.c ../../common_MachSuite/support.c ../../common_MachSuite/harness.c
INCLUDES := -I../../common_MachSuite

# Variant discovery
VARIANT_SRC := $(wildcard $(KERNEL_DIR)/aes_*.c)
VARIANTS := $(patsubst $(KERNEL_DIR)/aes_%.c,%,${VARIANT_SRC})

# Compiler tag for naming
ifneq (,$(findstring clang,$(CC)))
COMPILER_TAG := clang
else
COMPILER_TAG := gcc
endif

# Executables
BASE_EXE := $(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)
VARIANT_EXES := $(foreach v,$(VARIANTS),$(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)_$(v))

.PHONY: all clean run legacy

# Build baseline and all variants (no .o files)
all: $(BASE_EXE) $(VARIANT_EXES)

# Baseline: compile and link in one step
$(BASE_EXE): $(BASE_SRC) $(UTIL_SRCS)
	@mkdir -p $(OUT_DIR)
	-$(CC) $(CFLAGS) $(TIMING_DEFS) $(INCLUDES) -DOUTPUT_FILE_NAME=\"$(OUT_DIR)/aes_output.data\" -o $@ $^

# Variant executables
$(KERNEL_DIR)/$(TARGET)_$(COMPILER_TAG)_%: $(KERNEL_DIR)/aes_%.c $(UTIL_SRCS)
	@mkdir -p $(OUT_DIR)
	-$(CC) $(CFLAGS) $(TIMING_DEFS) $(INCLUDES) -DOUTPUT_FILE_NAME=\"$(OUT_DIR)/aes_output_$*.data\" -o $@ $^

# Optional legacy single-binary build in repo root (not part of 'all')
legacy: $(KERN)

$(KERN): $(BASE_SRC_LOCAL) $(UTIL_SRCS)
	@mkdir -p $(OUT_DIR)
	$(CC) $(CFLAGS) $(TIMING_DEFS) $(INCLUDES) -DOUTPUT_FILE_NAME=\"$(OUT_DIR)/aes_output.data\" -o $(KERN) $^

run: legacy input.data check.data
	./$(KERN) input.data check.data

clean:
	rm -f $(KERN) *.o
	rm -f $(KERNEL_DIR)/$(TARGET)_gcc $(KERNEL_DIR)/$(TARGET)_clang
	rm -f $(KERNEL_DIR)/$(TARGET)_gcc_* $(KERNEL_DIR)/$(TARGET)_clang_*