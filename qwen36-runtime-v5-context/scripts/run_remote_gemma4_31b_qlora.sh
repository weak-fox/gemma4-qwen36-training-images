#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-/root/gemma4-qlora}"
VENV_DIR="${VENV_DIR:-$WORKDIR/.venv}"
MODEL_DIR="${MODEL_DIR:-$WORKDIR/models/gemma-4-31b-it-unsloth-bnb-4bit}"
DATASET_PATH="${DATASET_PATH:-$WORKDIR/examples/smoke_messages.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORKDIR/runs/gemma4_31b_qlora_run}"
SCRIPT_PATH="${SCRIPT_PATH:-$WORKDIR/scripts/train_gemma4_31b_qlora.py}"

if [[ -n "${HTTPS_PROXY_URL:-}" ]]; then
  export https_proxy="${https_proxy:-$HTTPS_PROXY_URL}"
fi
if [[ -n "${HTTP_PROXY_URL:-}" ]]; then
  export http_proxy="${http_proxy:-$HTTP_PROXY_URL}"
fi
if [[ -n "${ALL_PROXY_URL:-}" ]]; then
  export all_proxy="${all_proxy:-$ALL_PROXY_URL}"
fi
export HF_HOME="${HF_HOME:-$WORKDIR/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$WORKDIR"
. "$VENV_DIR/bin/activate"

python -u "$SCRIPT_PATH" \
  --dataset-path "$DATASET_PATH" \
  --model-dir "$MODEL_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --local-files-only \
  "$@"
