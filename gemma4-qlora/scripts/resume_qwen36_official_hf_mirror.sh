#!/usr/bin/env bash
set -euo pipefail
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY no_proxy NO_PROXY
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/gemma4-qlora/hf
export HUGGINGFACE_HUB_CACHE=/root/gemma4-qlora/hf/hub
export HF_HUB_DISABLE_XET=1
exec python3 /root/gemma4-qlora/scripts/resume_qwen36_official_hf_mirror.py
