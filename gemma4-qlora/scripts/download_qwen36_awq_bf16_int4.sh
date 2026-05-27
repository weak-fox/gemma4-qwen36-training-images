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
    repo_id="cyankiwi/Qwen3.6-27B-AWQ-BF16-INT4",
    endpoint="https://hf-mirror.com",
    local_dir="/root/gemma4-qlora/models/qwen3.6-27b-awq-bf16-int4",
    resume_download=True,
    max_workers=4,
)
PY
