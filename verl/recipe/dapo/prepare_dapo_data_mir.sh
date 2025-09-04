#!/usr/bin/env bash
set -uxo pipefail

# ───────────── configurable variables ──────────────────────────
: "${VERL_HOME:=${HOME}/verl}"
: "${HF_ENDPOINT:=https://hf-mirror.com}"      # <- mirror domain
: "${OVERWRITE:=0}"

# local file targets
: "${TRAIN_FILE:=${VERL_HOME}/data/dapo-math-17k.parquet}"
: "${TEST_FILE:=${VERL_HOME}/data/aime-2024.parquet}"

mkdir -p "${VERL_HOME}/data"

# helper that prefers curl (resume + retries) but falls back to wget
download () {
  local url="$1"  dst="$2"
  echo ">> downloading $(basename "$dst") from $HF_ENDPOINT"
  if command -v curl >/dev/null 2>&1; then
    curl -L --retry 5 --fail -o "$dst" "$url"
  else
    wget -q --show-progress -O "$dst" "$url"
  fi
}

BASE="${HF_ENDPOINT}/datasets"   # common prefix for mirror

if [[ ! -f "$TRAIN_FILE" || "$OVERWRITE" -eq 1 ]]; then
  download \
    "${BASE}/BytedTsinghua-SIA/DAPO-Math-17k/resolve/main/data/dapo-math-17k.parquet" \
    "$TRAIN_FILE"
fi

if [[ ! -f "$TEST_FILE" || "$OVERWRITE" -eq 1 ]]; then
  download \
    "${BASE}/BytedTsinghua-SIA/AIME-2024/resolve/main/data/aime-2024.parquet" \
    "$TEST_FILE"
fi
echo "✓ Finished. Files are in ${VERL_HOME}/data"

