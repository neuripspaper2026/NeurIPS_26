# HPC-Bench

A performance-grounded benchmark for evaluating LLMs as optimizers of computational kernels in realistic HPC workloads. HPC-Bench covers **114 workloads spanning 16 computational motifs** (Dense Linear Algebra, Sparse Linear Algebra, Stencils / Scientific Computing (SC), Graph Algorithms, Dynamic Programming, N-body Methods, etc.) drawn from seven well-established benchmark suites.

> Anonymous submission for **NeurIPS 2026 Evaluations & Datasets Track** (double-blind).

The pipeline:

1. Construct optimization prompts from annotated kernel regions of each benchmark.
2. Sample N candidate implementations from each LLM under test.
3. Compile every candidate.
4. Verify correctness against reference outputs.
5. Measure kernel and total runtime on a controlled HPC environment.

Headline metrics (sampling-aware, correctness-gated):

* **Fast@k** — probability of obtaining a correct speed-up ≥ p<sub>thr</sub> within k samples.
* **Speedup@k** — expected best speed-up of obtaining a correct speed-up ≥ p<sub>thr</sub> with k samples.

Three optimization scenarios are evaluated:

* **EX1** — Serial CPU (loop restructuring, redundant-computation elimination, locality optimizations; no parallel constructs).
* **EX2** — OpenMP CPU parallelization (thread counts {1, 2, 8, 16, 32}).
* **EX3** — CUDA GPU optimization (NVIDIA A100, sm_80).

---

## Repository Layout

```
HPC-Bench/
├── EX1/                    # Serial CPU benchmark variants (per-suite subdirs)
├── EX2/                    # OpenMP CPU benchmark variants
├── EX3/                    # CUDA GPU benchmark variants
├── common.mk
├── common_parboil/         # Suite-specific Makefile infra
├── common_rodinia/
├── common_MachSuite/
├── driver_parboil/         # Parboil-suite driver utilities
├── scripts/
│   ├── arg.py              # Unified CLI entry point (subcommands listed below)
│   ├── benchmarks.py       # Core execution engine
│   ├── benchmark_catalog.py
│   ├── llm/                # Per-provider LLM clients (env-var driven)
│   ├── checkers/           # Suite-specific correctness validators
│   └── analysis/           # Plotting / aggregation / metric computation
├── configs/                # Per-benchmark runtime strategies (JSON)
├── results/                # Pre-computed time_measurements + correctness outputs
├── analysis_summaries/     # Paper-supporting figures and tables
├── final_data/             # Unified result CSVs
├── input_data_archives/    # Bundled input_data zips (small benchmarks)
├── tools/                  # Setup and entry-point shell scripts
└── docs/
    ├── ARCHITECTURE.md
    ├── BENCHMARKS.md
    └── REPRODUCING_PAPER.md
```

---

## Installation

### System requirements

| Component | Version (tested) |
|---|---|
| Linux | RHEL 8 (kernel 4.18) — any modern x86-64 distro should work |
| GCC / G++ | 14.2.0 |
| Clang / Clang++ | 19.1.3 |
| CUDA Toolkit | 12.4 (only required for `EX3`) |
| Python | 3.10+ |
| SLURM | optional, only needed for the batch wrappers in `tools/` |

### Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then source the environment helper:

```bash
source tools/setup_env.sh
```

### LLM API keys (only for `auto-run`)

Re-running the LLM generation stage requires API keys, which **must be exported as environment variables — never committed to source**:

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export TOGETHER_API_KEY="..."
```

---

## Data

`input_data/` directories are not committed inline because of their volume (≈ 117 GB raw; ≈ 23 GB zipped). Data are split between two locations:

* **Bundled** — small benchmarks (≈ 80 of 114) ship as `input_data_archives/*.zip` inside this repo.
* **Hugging Face** — 49 large benchmark archives (covering `stringsearch-mibench`, `basicmath-mibench`, `hotspot`, `leukocyte`, `bfs`, `patricia-mibench`, `qsort-mibench`, `mri-gridding`, `kmeans`, `pathfinder`, `backprop`, `susan-{c,e,s}-mibench`, `b+tree`, `hotspot3D`, `lud`, `sgemm`, `stencil`, `srad_v2`, `histo`, `huffman`) are hosted at https://huggingface.co/datasets/neuripspaper26/NeurIPS_26_Large_Data .

Populate everything with:

```bash
tools/setup_input_data.sh                # extract bundled small zips
tools/setup_input_data.sh --with-large   # also pull large archives via huggingface-cli
```

`EX5_correctness/` reference outputs are regenerable from the source via `python scripts/arg.py benchmark correctness-run` and are not included.

---

## Quickstart

```bash
# 1) Setup
source tools/setup_env.sh
tools/setup_input_data.sh

# 2) List all benchmarks
python scripts/arg.py benchmark list --benchmarks all

# 3) Build one benchmark across all LLM variants (uses pre-shipped optimized code)
python scripts/arg.py benchmark build --root . --benchmarks 2mm --ex-versions EX1

# 4) Verify correctness of all variants
tools/run_correctness.sh EX1 2mm

# 5) Measure runtime
tools/run_time_measurement.sh EX1 2mm --models baseline,claude --datasets mini,small
```

---

## End-to-end workflow

```bash
# (Optional) Re-run LLM code generation. Requires API keys.
tools/run_auto_run.sh EX1 2mm anthropic claude-sonnet-4-5-20250929 claude

# Build LLM-generated variants
python scripts/arg.py benchmark build --root . --benchmarks 2mm --ex-versions EX1

# Capture reference outputs (baseline first)
python scripts/arg.py benchmark correctness-run --root . --benchmarks 2mm \
  --ex-versions EX1 --models baseline,claude --datasets mini,small

# Verify correctness
tools/run_correctness.sh EX1 2mm

# Measure runtime
tools/run_time_measurement.sh EX1 2mm
```

See [`docs/REPRODUCING_PAPER.md`](docs/REPRODUCING_PAPER.md) for the exact commands used to produce every figure and table in the paper.

---

## Documentation

* [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — code organization and how to add a new benchmark / LLM client / suite.
* [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) — catalog of the seven suites with citations.
* [`docs/REPRODUCING_PAPER.md`](docs/REPRODUCING_PAPER.md) — step-by-step paper reproduction.

---

## License

Code in this repository is released under the MIT License (see [`LICENSE`](LICENSE)).

The bundled benchmark sources retain the licenses of their respective upstream suites (PolyBench, Rodinia, Parboil, MachSuite, MiBench, PARSEC, CODEE) — see `docs/BENCHMARKS.md` and the per-suite `COPYING` / `README` files for details.

---

## Citation

```
@inproceedings{hpcbench2026,
  title  = {HPC-Bench: A Performance-Grounded Benchmark for LLM-driven HPC Kernel Optimization},
  author = {Anonymous Authors},
  booktitle = {NeurIPS 2026 Evaluations \& Datasets Track},
  year   = {2026},
  note   = {Under review (double-blind)}
}
```
