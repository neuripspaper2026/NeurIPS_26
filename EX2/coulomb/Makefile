CC ?= gcc
OMP_FLAGS ?= -fopenmp
CFLAGS ?= -O3 -I lib -I include
CFLAGS += $(OMP_FLAGS)
LDFLAGS ?= -lm
LDFLAGS += $(OMP_FLAGS)

OPT_DIR := EX2_optimized_codes
LIB_SRCS := lib/Matrix2D.c lib/Vector.c

# Compiler tag for naming
ifneq (,$(findstring clang,$(CC)))
COMPILER_TAG := clang
else
COMPILER_TAG := gcc
endif

# Base and variants under EX2_optimized_codes (基准为 coulomb.c，变体 coulomb_*.c)
BASE_SRC := $(OPT_DIR)/coulomb.c
VARIANT_SRCS := $(filter-out $(BASE_SRC),$(wildcard $(OPT_DIR)/coulomb_*.c))
SUFFIXES := $(patsubst $(OPT_DIR)/coulomb_%.c,%,${VARIANT_SRCS})

# Executables
BASE_BIN := $(OPT_DIR)/coulomb_$(COMPILER_TAG)
VARIANT_BINS := $(addprefix $(OPT_DIR)/coulomb_$(COMPILER_TAG)_,${SUFFIXES})
ALL_BINS := $(BASE_BIN) ${VARIANT_BINS}

# Output directory for correctness data (当前模块下)
OUT_DIR := EX5_correctness

.PHONY: all clean bins run

all: bins

bins: $(ALL_BINS)

$(BASE_BIN): $(BASE_SRC) ${LIB_SRCS}
	-$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

$(OPT_DIR)/coulomb_$(COMPILER_TAG)_%: $(OPT_DIR)/coulomb_%.c ${LIB_SRCS}
	-$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

# 运行并将标准输出重定向保存到 EX5_correctness/output_gcc*.data
# 运行参数：SIZE(=n) [ITERS] [DENSITY]
SIZE ?= 350
ITERS ?= 1
DENSITY ?= 0.1
run: bins
	@mkdir -p $(OUT_DIR)
	@echo "[RUN] $(BASE_BIN) $(SIZE) $(ITERS) $(DENSITY)"
	@$(BASE_BIN) $(SIZE) $(ITERS) $(DENSITY) > $(OUT_DIR)/output_gcc.data 2>&1 || true
	@for s in ${SUFFIXES}; do \
		bin=$(OPT_DIR)/coulomb_gcc_$$s; \
		echo "[RUN] $$bin $(SIZE) $(ITERS) $(DENSITY)"; \
		$$bin $(SIZE) $(ITERS) $(DENSITY) > $(OUT_DIR)/output_gcc_$$s.data 2>&1 || true; \
	done

clean:
	@rm -f $(OPT_DIR)/coulomb_gcc $(OPT_DIR)/coulomb_clang
	@rm -f $(OPT_DIR)/coulomb_gcc_* $(OPT_DIR)/coulomb_clang_*
