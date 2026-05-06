#!/usr/bin/env bash
# Drive an LLM to generate optimized variants for a benchmark.
#
# Requires API keys in the environment (export them before running):
#   export OPENAI_API_KEY="..."        # for openai/gpt-5.1
#   export ANTHROPIC_API_KEY="..."     # for anthropic/claude
#   export TOGETHER_API_KEY="..."      # for together/qwen
#
# Usage:
#   tools/run_auto_run.sh <EX> <bench> <model_family> <model_name> <model_abbrev> [--max-attempts N]
#
# Examples:
#   tools/run_auto_run.sh EX1 2mm anthropic claude-sonnet-4-5-20250929 claude
#   tools/run_auto_run.sh EX2 bfs openai gpt-5.1 gpt5.1
#   tools/run_auto_run.sh EX3 cfd together Qwen/Qwen3-Coder-480B-A35B-Instruct qwen
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"

if [[ $# -lt 5 ]]; then
  sed -n '2,15p' "$0"
  exit 2
fi

EX="$1"
BENCH="$2"
FAMILY="$3"
MODEL="$4"
ABBREV="$5"
shift 5
EXTRA=("$@")

cd "${REPO_ROOT}"
python scripts/arg.py benchmark auto-run \
  --root . \
  --ex-version "${EX}" \
  --benchmarks "${BENCH}" \
  --model-family "${FAMILY}" \
  --llm-model "${MODEL}" \
  --llm-abbrev "${ABBREV}" \
  --max-attempts 10 \
  --write-state \
  "${EXTRA[@]}"
