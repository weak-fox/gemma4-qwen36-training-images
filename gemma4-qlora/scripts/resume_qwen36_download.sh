#!/usr/bin/env bash
set -euo pipefail
export https_proxy=http://105.100.31.173:7897
export http_proxy=http://105.100.31.173:7897
export all_proxy=socks5://105.100.31.173:7897
export HF_HOME=/root/gemma4-qlora/hf
export HUGGINGFACE_HUB_CACHE=/root/gemma4-qlora/hf/hub
export HF_HUB_DISABLE_XET=1
mkdir -p /root/gemma4-qlora/models/qwen3.6-27b-bnb-4bit /root/gemma4-qlora/logs
exec huggingface-cli download unsloth/Qwen3.6-27B \
  --local-dir /root/gemma4-qlora/models/qwen3.6-27b-bnb-4bit \
  --resume-download
