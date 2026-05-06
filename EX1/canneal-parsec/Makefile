CXX ?= g++
CXXFLAGS ?= -O3 -std=c++11
LDFLAGS ?=
LIBS ?= -lm

# Detect compiler for binary naming
ifneq (,$(findstring clang,$(CXX)))
COMPILER_TAG := clang
else ifneq (,$(findstring g++,$(CXX)))
COMPILER_TAG := gcc
else
COMPILER_TAG := gcc
endif

# Common sources (exclude annealer_thread.cpp so we can swap variants)
COMMON_SRCS = rng.cpp netlist.cpp netlist_elem.cpp main.cpp

# Output directories
OUT_DIR = EX1_optimized_codes
CORR_DIR = EX5_correctness

# Baseline executable (built from root annealer_thread.cpp)
BASE_EXE = $(OUT_DIR)/canneal-parsec_$(COMPILER_TAG)

# Variant sources in OUT_DIR (only suffixed files, no special 'opt')
VARIANT_SRCS = $(wildcard $(OUT_DIR)/annealer_thread_*.cpp)
VARIANT_EXES = $(patsubst $(OUT_DIR)/annealer_thread_%.cpp,$(OUT_DIR)/canneal-parsec_$(COMPILER_TAG)_%,$(VARIANT_SRCS))

.PHONY: all baseline variants run run-variants clean

all: baseline variants

baseline: $(BASE_EXE)

# Build all variants with error handling - continue on failure
variants:
	@echo "Building all variants..."
	@failed=0; \
	for exe in $(VARIANT_EXES); do \
		$(MAKE) --no-print-directory $$exe || failed=$$((failed+1)); \
	done; \
	if [ $$failed -gt 0 ]; then \
		echo "Warning: $$failed variants failed to build"; \
	fi

# Ensure output directories exist
$(OUT_DIR):
	mkdir -p $(OUT_DIR)
$(CORR_DIR):
	mkdir -p $(CORR_DIR)

# Baseline build - direct compilation without .o files (source in OUT_DIR)
$(BASE_EXE): $(OUT_DIR) $(COMMON_SRCS) $(OUT_DIR)/annealer_thread.cpp
	@mkdir -p $(OUT_DIR)
	@echo "Building baseline: $@..."
	@$(CXX) $(CXXFLAGS) -I. -o $@ $(OUT_DIR)/annealer_thread.cpp $(COMMON_SRCS) $(LDFLAGS) $(LIBS) || (echo "Error: Failed to build $@"; exit 1)

# Pattern rule for variant files - direct compilation without .o files
$(OUT_DIR)/canneal-parsec_$(COMPILER_TAG)_%: $(OUT_DIR)/annealer_thread_%.cpp $(OUT_DIR) $(COMMON_SRCS)
	@mkdir -p $(OUT_DIR)
	@echo "Building variant: $@..."
	@$(CXX) $(CXXFLAGS) -I. -o $@ $< $(COMMON_SRCS) $(LDFLAGS) $(LIBS) || (echo "Error: Failed to build $@"; exit 1)

# Default run parameters (can be overridden)
NTHREADS ?= 1
NSWAPS ?= 1000
TEMP ?= 2000
NETLIST ?= inputs/10.nets
NSTEPS ?= 50

RUN_ARGS = $(NTHREADS) $(NSWAPS) $(TEMP) $(NETLIST) $(NSTEPS)

# Run baseline and write stdout to EX5_correctness/output_gcc.data
run: $(BASE_EXE) $(CORR_DIR)
	$(BASE_EXE) $(RUN_ARGS) > $(CORR_DIR)/output_$(COMPILER_TAG).data

# Run all variants and write stdout to EX5_correctness/output_gcc_<suffix>.data
run-variants: variants $(CORR_DIR)
	@for exe in $(VARIANT_EXES); do \
		suf=$${exe##*_}; \
		$(exe) $(RUN_ARGS) > $(CORR_DIR)/output_$(COMPILER_TAG)_$$suf.data; \
	done

clean:
	rm -f $(BASE_EXE) $(VARIANT_EXES)
	rm -f $(CORR_DIR)/output_$(COMPILER_TAG).data $(CORR_DIR)/output_*_*.data


