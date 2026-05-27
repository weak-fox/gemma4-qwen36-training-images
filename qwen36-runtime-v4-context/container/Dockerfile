ARG BASE_IMAGE=unsloth/unsloth:2026.4.6-pt2.10.0-vllm-0.16.0-cu12.8-studio-release-v0.1.36-beta-fixes

FROM ${BASE_IMAGE} AS base

USER root

ENV PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=1800 \
    HF_HOME=/workspace/hf \
    HUGGINGFACE_HUB_CACHE=/workspace/hf/hub \
    HF_HUB_DISABLE_XET=1 \
    HF_HUB_DOWNLOAD_TIMEOUT=1800 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR /app

COPY container/requirements.overlay.txt /tmp/requirements.overlay.txt
COPY container/requirements.gemma4.txt /tmp/requirements.gemma4.txt
RUN python -m pip install --no-cache-dir --upgrade pip "setuptools<81" wheel && \
    python -m pip install --no-cache-dir -r /tmp/requirements.overlay.txt && \
    python -m pip install --no-cache-dir --no-deps -r /tmp/requirements.gemma4.txt

COPY container/entrypoint.sh /usr/local/bin/llm-runtime-entrypoint
COPY runtime /app/runtime
COPY scripts /app/scripts
COPY data /app/data
COPY examples /app/examples
COPY k8s /app/k8s

RUN chmod +x /usr/local/bin/llm-runtime-entrypoint && \
    ln -sf /usr/local/bin/llm-runtime-entrypoint /usr/local/bin/gemma4-runtime-entrypoint && \
    chmod +x /app/scripts/*.sh && \
    chmod +x /app/scripts/*.py && \
    python - <<'PY'
import importlib.metadata as metadata
import datasets
import peft
import torch
import transformers
import trl
print("python_ok")
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("trl", trl.__version__)
print("datasets", datasets.__version__)
print("peft", peft.__version__)
print("unsloth", metadata.version("unsloth"))
PY

USER unsloth:runtimeusers

ENTRYPOINT ["/usr/local/bin/llm-runtime-entrypoint"]

FROM base AS train-nomodel

ENV APP_MODE=train \
    MODEL_NAME_OR_PATH=unsloth/gemma-4-31b-it-unsloth-bnb-4bit \
    MODEL_LOCAL_FILES_ONLY=false \
    TRAIN_OUTPUT_DIR=/workspace/output/run \
    TRAIN_SAVE_ADAPTER=true

FROM base AS runtime-with-model

ARG MODEL_DIR_NAME=gemma-4-31b-it-unsloth-bnb-4bit
COPY model-bundle/${MODEL_DIR_NAME} /opt/models/${MODEL_DIR_NAME}

ENV APP_MODE=train \
    MODEL_NAME_OR_PATH=/opt/models/${MODEL_DIR_NAME} \
    MODEL_LOCAL_FILES_ONLY=true \
    TRAIN_OUTPUT_DIR=/workspace/output/run \
    TRAIN_SAVE_ADAPTER=true
