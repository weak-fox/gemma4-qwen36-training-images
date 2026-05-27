#!/bin/bash
set -euo pipefail
export https_proxy=http://105.100.31.173:7897
export http_proxy=http://105.100.31.173:7897
export all_proxy=socks5://105.100.31.173:7897
export HTTPS_PROXY=http://105.100.31.173:7897
export HTTP_PROXY=http://105.100.31.173:7897
export ALL_PROXY=socks5://105.100.31.173:7897
cd /root/gemma4-qlora-image
./scripts/prefetch_unsloth_base_image.sh
