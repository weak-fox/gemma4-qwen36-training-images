#!/usr/bin/env bash
set -euo pipefail
export https_proxy=http://105.100.31.173:7897
export http_proxy=http://105.100.31.173:7897
export all_proxy=socks5://105.100.31.173:7897
export HF_HOME=/root/gemma4-qlora/hf
export HUGGINGFACE_HUB_CACHE=/root/gemma4-qlora/hf/hub
export HF_HUB_DISABLE_XET=1
exec python3 /root/gemma4-qlora/scripts/resume_qwen36_download_single.py
