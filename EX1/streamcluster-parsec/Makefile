CXX ?= g++
CXXFLAGS ?= -O3 -std=c++11
LDFLAGS ?=
LIBS ?= -lm

# Output directories
OUT_DIR = EX1_optimized_codes
CORR_DIR = EX5_correctness

# Dynamically detect compiler tag from CXX
ifneq (,$(findstring clang,$(CXX)))
    COMPILER_TAG = clang
else ifneq (,$(findstring g++,$(CXX)))
    COMPILER_TAG = gcc
else
    COMPILER_TAG = gcc
endif

# Baseline executable (built from root streamcluster.cpp)
BASE_EXE = $(OUT_DIR)/streamcluster_$(COMPILER_TAG)

# Variant sources in OUT_DIR (only suffixed files, no special 'opt')
VARIANT_SRCS = $(wildcard $(OUT_DIR)/streamcluster_*.cpp)
VARIANT_EXES = $(patsubst $(OUT_DIR)/streamcluster_%.cpp,$(OUT_DIR)/streamcluster_$(COMPILER_TAG)_%,$(VARIANT_SRCS))

.PHONY: all baseline variants run run-variants clean

all: baseline variants

baseline: $(BASE_EXE)

# Build all variants with error handling - continue on error
variants: $(OUT_DIR)
	@echo "Building variants..."
	@failed=0; \
	for src in $(VARIANT_SRCS); do \
		base=$$(basename $$src .cpp); \
		suffix=$${base#streamcluster_}; \
		exe=$(OUT_DIR)/streamcluster_$(COMPILER_TAG)_$$suffix; \
		echo "  Building $$exe from $$src..."; \
		$(MAKE) --no-print-directory $$exe || failed=$$((failed+1)); \
	done; \
	if [ $$failed -gt 0 ]; then \
		echo "Warning: $$failed variant(s) failed to compile"; \
	fi

# Ensure output directories exist
$(OUT_DIR):
	mkdir -p $(OUT_DIR)
$(CORR_DIR):
	mkdir -p $(CORR_DIR)

# Baseline build
$(BASE_EXE): $(OUT_DIR) streamcluster.cpp
	$(CXX) $(CXXFLAGS) -I. -o $@ streamcluster.cpp $(LDFLAGS) $(LIBS)

# Pattern rule for variant files: EX1_optimized_codes/streamcluster_<suffix>.cpp
$(OUT_DIR)/streamcluster_$(COMPILER_TAG)_%: $(OUT_DIR)/streamcluster_%.cpp $(OUT_DIR)
	$(CXX) $(CXXFLAGS) -I. -o $@ $< $(LDFLAGS) $(LIBS)

# Default run parameters (use synthetic input by setting N > 0)
KMIN ?= 10
KMAX ?= 20
DIM ?= 32
N ?= 10000
CHUNK ?= 1000
CLUSTERS ?= 1000
NPROC ?= 1

# For synthetic runs, infile is ignored but argv requires it; outfile is where results go
IGNORED_IN ?= inputs/ignored

RUN_ARGS = $(KMIN) $(KMAX) $(DIM) $(N) $(CHUNK) $(CLUSTERS) $(IGNORED_IN)

# Run baseline and write to EX5_correctness/output_gcc.data
run: $(BASE_EXE) $(CORR_DIR)
	$(BASE_EXE) $(RUN_ARGS) $(CORR_DIR)/output_$(COMPILER_TAG).data $(NPROC)

# Run all variants and write to EX5_correctness/output_gcc_<suffix>.data
run-variants: variants $(CORR_DIR)
	@for exe in $(VARIANT_EXES); do \
		suf=$${exe##*_}; \
		$$exe $(RUN_ARGS) $(CORR_DIR)/output_$(COMPILER_TAG)_$$suf.data $(NPROC); \
	done

clean:
	rm -f $(BASE_EXE) $(VARIANT_EXES)
	rm -f $(CORR_DIR)/output_$(COMPILER_TAG).data $(CORR_DIR)/output_*_*.data
