#!/usr/bin/env bash
set -euo pipefail
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY no_proxy NO_PROXY
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/gemma4-qlora/hf
export HUGGINGFACE_HUB_CACHE=/root/gemma4-qlora/hf/hub
export HF_HUB_DISABLE_XET=1
exec python3 - <<"PY"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="unsloth/Qwen3.6-27B-UD-MLX-4bit",
    endpoint="https://hf-mirror.com",
    local_dir="/root/gemma4-qlora/models/qwen3.6-27b-ud-mlx-4bit",
    resume_download=True,
    max_workers=4,
)
PY
