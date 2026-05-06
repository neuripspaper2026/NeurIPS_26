#!/usr/bin/env bash
# Populate EX{1,2,3}/<bench>/input_data/ directories.
#
# Steps performed:
#   1. Extract every input_data_archives/*.zip bundled in this repo.
#   2. Optionally fetch large benchmarks from the companion Hugging Face dataset
#      (those exceed GitHub's 100 MB single-file limit so they live elsewhere).
#
# Usage:
#   tools/setup_input_data.sh                  # extract bundled small zips only
#   tools/setup_input_data.sh --with-large     # also download large ones from HF
#
# Set HPC_BENCH_HF_REPO to override the HF dataset path. The default is the
# anonymous repo published with this submission.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"
ARCHIVES_DIR="${REPO_ROOT}/input_data_archives"

WITH_LARGE=0
for arg in "$@"; do
  case "$arg" in
    --with-large) WITH_LARGE=1 ;;
    -h|--help)
      sed -n '2,15p' "$0"
      exit 0
      ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

# 1) Extract bundled zips.
if [[ -d "${ARCHIVES_DIR}" ]]; then
  echo "Extracting bundled input_data zips from ${ARCHIVES_DIR}..."
  cd "${REPO_ROOT}"
  for z in "${ARCHIVES_DIR}"/*.zip; do
    [[ -f "$z" ]] || continue
    # Each zip is named like input_data_<EX>_<bench>.zip and contains a single
    # `input_data/` dir, intended to be extracted into EX<>/<bench>/.
    fname="$(basename "$z" .zip)"
    rest="${fname#input_data_}"          # strip "input_data_"
    ex="${rest%%_*}"                     # EX1 / EX2 / EX3
    bench="${rest#*_}"                   # benchmark name (may contain underscores)
    target_dir="${REPO_ROOT}/${ex}/${bench}"
    if [[ ! -d "${target_dir}" ]]; then
      echo "  skip: target dir ${target_dir} does not exist (zip ${fname})"
      continue
    fi
    echo "  -> ${ex}/${bench}/"
    (cd "${target_dir}" && unzip -oq "$z")
  done
  echo "Done extracting bundled archives."
else
  echo "Note: ${ARCHIVES_DIR} not present — skipping bundled extraction."
fi

# 2) Optional: pull large archives from Hugging Face.
if [[ "${WITH_LARGE}" -eq 1 ]]; then
  : "${HPC_BENCH_HF_REPO:=anonymous/HPC-Bench-input-data}"
  echo ""
  echo "Fetching large input_data archives from Hugging Face: ${HPC_BENCH_HF_REPO}"
  if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "  huggingface-cli not found. Install with: pip install huggingface_hub[cli]"
    exit 3
  fi
  TMPDIR="${REPO_ROOT}/.hf_cache_input_data"
  mkdir -p "${TMPDIR}"
  huggingface-cli download "${HPC_BENCH_HF_REPO}" --repo-type dataset \
    --local-dir "${TMPDIR}" --local-dir-use-symlinks False
  for z in "${TMPDIR}"/*.zip; do
    [[ -f "$z" ]] || continue
    fname="$(basename "$z" .zip)"
    rest="${fname#input_data_}"
    ex="${rest%%_*}"
    bench="${rest#*_}"
    target_dir="${REPO_ROOT}/${ex}/${bench}"
    if [[ ! -d "${target_dir}" ]]; then
      echo "  skip: ${target_dir} missing (zip ${fname})"
      continue
    fi
    echo "  -> ${ex}/${bench}/"
    (cd "${target_dir}" && unzip -oq "$z")
  done
  echo "Done fetching large archives."
fi

echo ""
echo "input_data setup complete."
