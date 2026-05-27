#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-gemma4-qlora:runtime-with-model}"
SOURCE_IMAGE="${SOURCE_IMAGE:-gemma4-qlora:train-nomodel}"
MODEL_SOURCE_DIR="${MODEL_SOURCE_DIR:-/root/gemma4-qlora/models/gemma-4-31b-it-unsloth-bnb-4bit}"
MODEL_DIR_NAME="gemma-4-31b-it-unsloth-bnb-4bit"
MODEL_DEST_DIR="/opt/models/${MODEL_DIR_NAME}"
CONTAINER_NAME="gemma4-qlora-runtime-with-model-$$"

cleanup() {
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY no_proxy NO_PROXY

if [[ ! -d "${MODEL_SOURCE_DIR}" ]]; then
  echo "MODEL_SOURCE_DIR does not exist: ${MODEL_SOURCE_DIR}" >&2
  exit 1
fi

if [[ ! -f "${MODEL_SOURCE_DIR}/config.json" ]]; then
  echo "MODEL_SOURCE_DIR is missing config.json: ${MODEL_SOURCE_DIR}" >&2
  exit 1
fi

if ! docker image inspect "$SOURCE_IMAGE" >/dev/null 2>&1; then
  echo "SOURCE_IMAGE is not present locally: $SOURCE_IMAGE" >&2
  echo "Build the train image first with ./scripts/build_train_nomodel_image.sh, or override SOURCE_IMAGE." >&2
  exit 1
fi

DOCKER_ROOT_DIR="$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || true)"
if [[ -z "${DOCKER_ROOT_DIR}" ]]; then
  DOCKER_ROOT_DIR="/var/lib/docker"
fi

MODEL_BYTES="$(du -sb "${MODEL_SOURCE_DIR}" | awk '{print $1}')"
AVAIL_BYTES="$(df -B1 --output=avail "${DOCKER_ROOT_DIR}" | tail -n 1 | tr -d ' ')"
REQUIRED_BYTES=$(( MODEL_BYTES + 4 * 1024 * 1024 * 1024 ))

if (( AVAIL_BYTES < REQUIRED_BYTES )); then
  echo "Not enough free space under ${DOCKER_ROOT_DIR}." >&2
  echo "Need at least ${REQUIRED_BYTES} bytes, only ${AVAIL_BYTES} bytes available." >&2
  exit 1
fi

echo "Creating container ${CONTAINER_NAME} from ${SOURCE_IMAGE}"
docker create \
  --name "${CONTAINER_NAME}" \
  --user root \
  --mount "type=bind,src=${MODEL_SOURCE_DIR},dst=/mnt/model,readonly" \
  --entrypoint /bin/bash \
  "${SOURCE_IMAGE}" \
  -lc "mkdir -p '${MODEL_DEST_DIR}' && cp -a /mnt/model/. '${MODEL_DEST_DIR}/'"

echo "Copying model from ${MODEL_SOURCE_DIR} into ${SOURCE_IMAGE}"
docker start -a "${CONTAINER_NAME}"

echo "Committing ${IMAGE_TAG}"
docker commit \
  --change "ENV APP_MODE=train" \
  --change "ENV MODEL_NAME_OR_PATH=${MODEL_DEST_DIR}" \
  --change "ENV MODEL_LOCAL_FILES_ONLY=true" \
  --change "ENV TRAIN_OUTPUT_DIR=/workspace/output/run" \
  --change "ENV TRAIN_SAVE_ADAPTER=true" \
  --change "USER unsloth:runtimeusers" \
  --change 'ENTRYPOINT ["/usr/local/bin/gemma4-runtime-entrypoint"]' \
  --change 'CMD []' \
  "${CONTAINER_NAME}" \
  "${IMAGE_TAG}" >/dev/null

docker image inspect "${IMAGE_TAG}" >/dev/null
echo "Created ${IMAGE_TAG} from ${SOURCE_IMAGE}"
