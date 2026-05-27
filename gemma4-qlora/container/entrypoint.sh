#!/usr/bin/env bash
set -euo pipefail

mkdir -p /workspace/input
mkdir -p /workspace/output
mkdir -p /workspace/artifacts
mkdir -p /workspace/hf

export PYTHONPATH="/app${PYTHONPATH:+:$PYTHONPATH}"
cd /workspace

exec python -m runtime.main "$@"
