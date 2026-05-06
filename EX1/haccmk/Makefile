CC ?= gcc
CFLAGS_MAIN ?= -O2 -fopenmp -I lib
CFLAGS_KERNEL ?= -O3 -I lib
LDFLAGS ?= -lm -fopenmp

OPT_DIR := EX1_optimized_codes
LIB_COMMON := lib/mysecond.c
BASE_KERNEL := lib/Step10_orig.c

# Compiler tag for naming
ifneq (,$(findstring clang,$(CC)))
COMPILER_TAG := clang
else
COMPILER_TAG := gcc
endif

# Base和变体：主程序为根目录 haccmk.c，kernel 基线为 lib/Step10_orig.c，
# 变体为 EX1_optimized_codes/Step10_orig_*.c
BASE_SRC := haccmk.c
VARIANT_SRCS := $(wildcard $(OPT_DIR)/Step10_orig_*.c)
SUFFIXES := $(patsubst $(OPT_DIR)/Step10_orig_%.c,%,${VARIANT_SRCS})

# Executables
BASE_BIN := $(OPT_DIR)/haccmk_$(COMPILER_TAG)
VARIANT_BINS := $(foreach s,${SUFFIXES},$(OPT_DIR)/haccmk_$(COMPILER_TAG)_$(s))
ALL_BINS := $(BASE_BIN) ${VARIANT_BINS}

# Output directory (当前模块下)
OUT_DIR := EX5_correctness

.PHONY: all clean bins run

all: bins

bins: $(ALL_BINS)

$(BASE_BIN): $(BASE_SRC) $(LIB_COMMON) $(BASE_KERNEL)
	-$(CC) $(CFLAGS_MAIN) $^ $(LDFLAGS) -o $@

# Pattern rule for variants: replace kernel source with Step10_orig_%.c
$(OPT_DIR)/haccmk_$(COMPILER_TAG)_%: $(BASE_SRC) $(LIB_COMMON) $(OPT_DIR)/Step10_orig_%.c
	-$(CC) $(CFLAGS_MAIN) $^ $(LDFLAGS) -o $@

# Run and save output to EX5_correctness
run: bins
	@mkdir -p $(OUT_DIR)
	@echo "[RUN] $(BASE_BIN)"
	@$(BASE_BIN) > $(OUT_DIR)/output_$(COMPILER_TAG).data 2>&1 || true
	@for s in ${SUFFIXES}; do \
		bin=$(OPT_DIR)/haccmk_$(COMPILER_TAG)_$$s; \
		echo "[RUN] $$bin"; \
		$$bin > $(OUT_DIR)/output_$(COMPILER_TAG)_$$s.data 2>&1 || true; \
	done

clean:
	@rm -f $(OPT_DIR)/haccmk_gcc $(OPT_DIR)/haccmk_clang
	@rm -f $(OPT_DIR)/haccmk_gcc_* $(OPT_DIR)/haccmk_clang_*

