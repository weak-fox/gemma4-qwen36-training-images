#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-gemma4-qlora:train-nomodel}"
BASE_IMAGE="${BASE_IMAGE:-unsloth/unsloth:2026.4.6-pt2.10.0-vllm-0.16.0-cu12.8-studio-release-v0.1.36-beta-fixes}"
BUILD_ARGS=()

for proxy_name in http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY no_proxy NO_PROXY; do
  proxy_value="${!proxy_name:-}"
  if [[ -n "${proxy_value}" ]]; then
    BUILD_ARGS+=(--build-arg "${proxy_name}=${proxy_value}")
  fi
done

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY no_proxy NO_PROXY

cd "$REPO_ROOT"

if ! docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
  echo "Base image is not present locally: $BASE_IMAGE" >&2
  echo "Run ./scripts/prefetch_unsloth_base_image.sh first, or override BASE_IMAGE to an existing local image." >&2
  exit 1
fi

echo "Building ${IMAGE_TAG} from source Dockerfile at ${REPO_ROOT}"
docker build \
  --file container/Dockerfile \
  --target train-nomodel \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  "${BUILD_ARGS[@]}" \
  --tag "${IMAGE_TAG}" \
  .
