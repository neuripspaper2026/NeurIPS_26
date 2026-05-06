#!/usr/bin/env bash
# HPC-Bench environment setup helper.
#
# Source this script to set REPO_ROOT and verify required env vars.
#
#   source tools/setup_env.sh
#
# Then optionally export your LLM keys (only needed for `benchmark auto-run`):
#
#   export OPENAI_API_KEY="..."
#   export ANTHROPIC_API_KEY="..."
#   export TOGETHER_API_KEY="..."

# Resolve repo root no matter where the script is sourced from.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Note: this script is intended to be 'source'd, not executed."
fi
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export REPO_ROOT="$(cd "${_HERE}/.." && pwd)"

# Optional: HF cache. Defaults are fine for most setups.
: "${HF_HOME:=${HOME}/.cache/huggingface}"
export HF_HOME

# Optional: CUDA. Override if your CUDA is in a non-standard location.
# export CUDA_HOME=/usr/local/cuda
# export PATH="${CUDA_HOME}/bin:${PATH}"
# export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

echo "REPO_ROOT=${REPO_ROOT}"
echo "HF_HOME=${HF_HOME}"
[[ -n "${OPENAI_API_KEY:-}" ]] && echo "OPENAI_API_KEY: set" || echo "OPENAI_API_KEY: NOT set (only needed for auto-run)"
[[ -n "${ANTHROPIC_API_KEY:-}" ]] && echo "ANTHROPIC_API_KEY: set" || echo "ANTHROPIC_API_KEY: NOT set (only needed for auto-run)"
[[ -n "${TOGETHER_API_KEY:-}" ]] && echo "TOGETHER_API_KEY: set" || echo "TOGETHER_API_KEY: NOT set (only needed for auto-run)"
