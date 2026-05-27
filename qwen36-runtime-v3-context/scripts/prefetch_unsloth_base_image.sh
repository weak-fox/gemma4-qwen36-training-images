#!/usr/bin/env bash
set -euo pipefail

IMAGE_REF="${IMAGE_REF:-unsloth/unsloth:2026.4.6-pt2.10.0-vllm-0.16.0-cu12.8-studio-release-v0.1.36-beta-fixes}"
TMP_DIR="${TMP_DIR:-/root/tmp}"
OCI_ARCHIVE_PATH="${OCI_ARCHIVE_PATH:-$TMP_DIR/$(echo "$IMAGE_REF" | tr '/:' '__').oci}"

mkdir -p "$TMP_DIR"

if ! command -v skopeo >/dev/null 2>&1; then
  echo "skopeo is required but not installed" >&2
  exit 1
fi

if ! command -v ctr >/dev/null 2>&1; then
  echo "ctr is required but not installed" >&2
  exit 1
fi

saved_http_proxy="${http_proxy:-${HTTP_PROXY:-}}"
saved_https_proxy="${https_proxy:-${HTTPS_PROXY:-}}"

if [[ -z "$saved_http_proxy" || -z "$saved_https_proxy" ]]; then
  echo "HTTP_PROXY/HTTPS_PROXY (or lowercase variants) must be set for skopeo download" >&2
  exit 1
fi

export HTTP_PROXY="$saved_http_proxy"
export HTTPS_PROXY="$saved_https_proxy"
unset ALL_PROXY all_proxy
unset NO_PROXY no_proxy

echo "Prefetching $IMAGE_REF into $OCI_ARCHIVE_PATH"
skopeo copy --retry-times 10 "docker://docker.io/$IMAGE_REF" "oci-archive:$OCI_ARCHIVE_PATH:$IMAGE_REF"

echo "Importing OCI archive into containerd moby namespace"
ctr -n moby images import --all-platforms --base-name "${IMAGE_REF%:*}" "$OCI_ARCHIVE_PATH"

echo "Exporting imported image as Docker-compatible archive and loading into Docker"
unset HTTP_PROXY HTTPS_PROXY
ctr -n moby images export - "$IMAGE_REF" | docker load

echo "Verifying local Docker image"
docker image inspect "$IMAGE_REF" >/dev/null
docker images --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}' | grep "^$IMAGE_REF "
