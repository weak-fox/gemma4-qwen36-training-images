# Gemma 4 / Qwen3.6 Training Images

This repository contains the Dockerfiles, runtime code, helper scripts, Kubernetes examples, smoke data, and notes recovered from `root@105.100.31.190` for Gemma 4 and Qwen3.6 QLoRA/SFT training images.

Large runtime artifacts are intentionally excluded from Git:

- model weights under `models/`
- Hugging Face caches under `hf/`
- Python virtual environments
- training outputs under `runs/`
- Docker image archives
- local serving artifacts and compiled caches

## Layout

- `gemma4-qlora/`: main training/runtime workspace. Includes the source Dockerfile, training scripts, runtime module, Kubernetes examples, tests, and small sample datasets.
- `gemma4-qlora-image/`: earlier image packaging workspace and Chinese runbooks/findings.
- `qwen36-runtime-v3-context/` through `qwen36-runtime-v6-context/`: Qwen3.6 runtime overlay snapshots.

## Common Entry Points

- `gemma4-qlora/container/Dockerfile`
- `gemma4-qlora/scripts/build_train_nomodel_image.sh`
- `gemma4-qlora/scripts/build_runtime_with_model_image.sh`
- `gemma4-qlora/scripts/train_gemma4_31b_qlora.py`
- `gemma4-qlora/runtime/train_mode.py`

The Qwen3.6 paths reuse the same runtime/training structure, with the actual model selected by image environment variables such as `MODEL_NAME_OR_PATH`.
