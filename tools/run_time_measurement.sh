#!/usr/bin/env bash
# Run wall-clock time measurements for a benchmark.
#
# Usage:
#   tools/run_time_measurement.sh <EX> <bench> [--models claude,gpt5.1,qwen,baseline] [--datasets mini,small,medium,large]
#
# Examples:
#   tools/run_time_measurement.sh EX1 2mm
#   tools/run_time_measurement.sh EX3 bfs --models baseline,claude --datasets medium
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"

if [[ $# -lt 2 ]]; then
  sed -n '2,10p' "$0"
  exit 2
fi

EX="$1"
BENCH="$2"
shift 2
EXTRA_ARGS=("$@")

cd "${REPO_ROOT}"
python scripts/arg.py time_measurement \
  --root . \
  --ex-version "${EX}" \
  --benchmark "${BENCH}" \
  --runtime-hints configs/runtime_hints.json \
  --no-dry-run \
  "${EXTRA_ARGS[@]}"
