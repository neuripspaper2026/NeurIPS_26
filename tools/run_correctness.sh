#!/usr/bin/env bash
# Run correctness checks for one or more benchmarks.
#
# Usage:
#   tools/run_correctness.sh <EX>                       # all benchmarks in that EX
#   tools/run_correctness.sh <EX> <bench>[,<bench>...]  # specific benchmarks
#
# Examples:
#   tools/run_correctness.sh EX1
#   tools/run_correctness.sh EX2 2mm,3mm,bfs
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"

if [[ $# -lt 1 ]]; then
  sed -n '2,10p' "$0"
  exit 2
fi

EX="$1"
BENCH_ARG="${2:-all}"

cd "${REPO_ROOT}"
python scripts/arg.py benchmark check \
  --root . \
  --ex-versions "${EX}" \
  --benchmarks "${BENCH_ARG}"
