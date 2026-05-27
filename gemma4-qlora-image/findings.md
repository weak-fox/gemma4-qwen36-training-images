# Findings & Decisions

## Requirements
- Run the official Unsloth Gemma training flow on `root@105.100.31.190`.
- Target a 31B QLoRA run.
- First make it work in practice, then explain the complete process step by step in Chinese.
- The environment is in mainland China, so blocked web access may require:
  `export https_proxy=http://105.100.31.173:7897 http_proxy=http://105.100.31.173:7897 all_proxy=socks5://105.100.31.173:7897`

## Research Findings
- Official Unsloth Gemma 4 training page currently says its larger Gemma 4 notebooks, including `Gemma-4-31B`, need an A100 GPU.
- The same page shows the code-based section as a minimal SFT recipe for text-only fine-tuning, but the explicit 31B example shown there uses `FastVisionModel.from_pretrained(model_name = "google-gemma-4-31b-it")`.
- The page suggests reducing `per_device_train_batch_size` to 1 and lowering `max_seq_length` if OOM occurs, while keeping `use_gradient_checkpointing = "unsloth"` enabled.
- Unsloth's official requirements page currently lists the absolute minimum VRAM for `32B` QLoRA as `26 GB`.
- Unsloth's requirements page also says supported Python versions are `3.11 <= version < 3.14`.
- Unsloth's official pip install page recommends either `uv pip install unsloth --torch-backend=auto` or a venv + uv flow, and its bundled auto-install logic explicitly supports CUDA `13.0`.
- Unsloth's multi-GPU guide states:
  - single-process model splitting can place a model across GPUs with `device_map="balanced"`
  - DDP duplicates the whole model on each GPU and is therefore for speed, not for increasing per-model VRAM capacity
- Remote server `root@105.100.31.190` is reachable over SSH.
- Remote host facts gathered so far:
  - Hostname: `lyp-t4`
  - GPUs: `Tesla T4` x2, each about 15-16 GiB VRAM
  - Driver: `580.105.08`
  - CUDA reported by `nvidia-smi`: `13.0`
  - Python: `3.10.12`
  - RAM: about `62 GiB`
  - Root filesystem free space: about `59 GiB`
- Initial assessment: 2x T4 16G is materially below the A100-class setup Unsloth documents for Gemma 4 31B, so a clean 31B QLoRA run on this host is unlikely without a different strategy or different hardware.
- Refined assessment: a 4-bit 31B model may be worth one empirical attempt with cross-GPU model splitting and minimal settings, but official docs still point toward A100-class hardware and the current host remains outside the documented happy path.
- The correct public 4-bit repo name is lowercase: `unsloth/gemma-4-31b-it-unsloth-bnb-4bit`.
- The 4-bit repo currently consists of 6 files totaling about 25 GB; the largest shards are about 5 GB each.
- Direct access from the remote host to `huggingface.co` times out without a proxy.
- With the provided proxy, small ranged downloads from Hugging Face work but observed throughput was only on the order of tens of KB/s, which implies multi-day download times for the full 31B 4-bit checkpoint.
- Fresh verification at the end of the session showed proxy-based ranged file transfer can also fail intermittently with `curl: (35) error:0A000126:SSL routines::unexpected eof while reading`, so the large-file path is both slow and unstable.
- The remote host CPU flags have now changed materially and include `sse4_1`, `sse4_2`, `avx`, `avx2`, `fma`, and `avx512*`, so the previous NumPy `x86_v2` incompatibility should no longer apply.
- The official Unsloth page `Gemma 4 - How to Run Locally` currently says `Gemma-4-31B needs 20GB RAM (4-bit)` and in the hardware table shows `31B: 17–20 GB` for 4-bit inference memory.
- This `17–20 GB` / `20GB RAM` figure is from the local inference page, not the Gemma 4 training page.
- The current Unsloth requirements page still lists `32B` QLoRA fine-tuning minimum VRAM as `26 GB` absolute minimum.
- Earlier in the session the remote CPU flags did not include newer instruction sets like SSE4.1/SSE4.2/AVX, which is why the newest NumPy wheel initially failed with an `x86_v2` baseline error.
- After the CPU flag change, `numpy==2.4.4`, `torch==2.10.0+cu130`, and `unsloth==2026.4.4` all import successfully on the remote host.
- Installing `socksio==1.0.0` allows `huggingface_hub` to work with `all_proxy=socks5://105.100.31.173:7897`.
- A persistent background download job for `unsloth/gemma-4-31b-it-unsloth-bnb-4bit` is now running on the remote host.
- Current remote download state observed on 2026-04-13:
  - Background PID: `8131`
  - Log file: `/root/gemma4-qlora/logs/download_31b_4bit.log`
  - Cache/model data reached about `13.9G` combined in the latest check
  - Local model directory exists at `/root/gemma4-qlora/models/gemma-4-31b-it-unsloth-bnb-4bit`
  - Small files like `config.json`, `generation_config.json`, `chat_template.jinja`, `README.md` are already present
  - Full shards completed so far: `model-00001-of-00006.safetensors`, `model-00002-of-00006.safetensors`
  - Multiple remaining shard blobs are still downloading in `.../hf/hub/.../blobs/*.incomplete`
- A second background watcher is now running to automatically perform a model load check once `model-00006-of-00006.safetensors` appears.
- Auto-load-check watcher PID: `29084`
- Auto-load-check log: `/root/gemma4-qlora/logs/after_download_loadcheck.log`
- The Xet-backed large-file path eventually stalled. Evidence captured:
  - downloader child had a proxy socket in `CLOSE_WAIT`
  - 30-second incomplete-file growth was `0`
  - `hf_xet` logs showed repeated `tls handshake eof` against `https://cas-server.xethub.hf.co/v1/reconstructions/...`
- After restarting with `HF_HUB_DISABLE_XET=1` and only `http_proxy` / `https_proxy`, the download resumed through the regular Hugging Face path.
- Fresh verification after the restart showed:
  - downloader child PID `39933`
  - active proxy sockets in `ESTABLISHED`
  - `/root/gemma4-qlora/models/.../.cache/huggingface/download/*.incomplete` grew by `136,314,880` bytes over 30 seconds
  - the current active incomplete file for `model-00003-of-00006` reached about `470M`
- Later verification on the non-Xet path showed the download was still progressing despite another mid-file disconnect:
  - `model-00004-of-00006.safetensors` completed at `2026-04-13 13:28`
  - `model-00006-of-00006.safetensors` completed at `2026-04-13 13:34`
  - a current `.incomplete` file kept growing from about `650M` upward
  - 60-second cache growth sample was `146,800,640` bytes
- The newest error on the regular HF path is different from the Xet issue:
  - `httpx.RemoteProtocolError: peer closed connection without sending complete message body`
  - example: `received 691404281 bytes, expected 4994477839`
- Interpretation: the proxy or upstream closes some long HTTP responses mid-stream, but the current retrying downloader can continue and make net progress.
- Final download status on 2026-04-13:
  - All six weight shards are present locally.
  - Total local model directory size is about `24G`.
  - Missing small metadata files were later fetched separately.
- Final local load-check result:
  - `FastVisionModel.from_pretrained(...)` succeeded from the local model directory.
  - Loaded class: `Gemma4ForConditionalGeneration`
  - Processor: `Gemma4Processor`
  - `device_map=\"balanced\"` split layers across GPU 0 and GPU 1 successfully.
- Final smoke-train result on 2026-04-13:
  - A 1-step QLoRA smoke test completed successfully on `2x Tesla T4`.
  - Dataset: 2 tiny text-only conversational samples in `messages` format.
  - LoRA config used for the smoke test: `r=8`, `lora_alpha=8`, `finetune_vision_layers=False`, `finetune_language_layers=True`, `finetune_attention_modules=True`, `finetune_mlp_modules=True`.
  - Runtime: about `50.37s` for `max_steps=1`.
  - Peak allocated memory observed:
    - GPU 0: `11,053,898,752` bytes
    - GPU 1: `16,106,813,952` bytes
- No real training dataset was found in the current local workspace or under `/root/gemma4-qlora` on the remote host; only model metadata files were present there.
- Reusable delivery artifacts created on 2026-04-13:
  - `scripts/train_gemma4_31b_qlora.py`
  - `scripts/run_remote_gemma4_31b_qlora.sh`
  - `examples/smoke_messages.jsonl`
- Persona dataset artifacts created on 2026-04-13:
  - `scripts/build_cogiot_persona_dataset.py`
  - `data/cogiot_xiaoge_train.jsonl`
  - `data/cogiot_xiaoge_identity_seed.jsonl`
  - `data/cogiot_xiaoge_dataset_notes.md`
- The reusable training script supports:
  - `.json`, `.jsonl`, `.csv`, `.parquet`
  - `datasets.load_from_disk()` directories
  - either a `messages` column or `prompt` + `completion` columns
- The synthetic persona dataset intentionally separates:
  - identity anchoring for prompts like “你是谁”
  - lighter style bleed for normal answers
  - a shorter identity-heavy seed set for faster visible persona shifts
- Final wrapper-based smoke-train result on 2026-04-13:
  - `./run_remote_gemma4_31b_qlora.sh --max-steps 1 --save-steps 0` completed successfully on the remote host.
  - Output directory: `/root/gemma4-qlora/runs/gemma4_31b_qlora_smoke_script`
  - Runtime: about `27.33s` for `max_steps=1`.
  - Train loss: about `9.61596`.
  - Peak allocated memory observed:
    - GPU 0: `11,053,898,752` bytes
    - GPU 1: `16,106,813,952` bytes
- Persona-dataset smoke results on 2026-04-13:
  - Full persona dataset:
    - path: `/root/gemma4-qlora/data/cogiot_xiaoge_train.jsonl`
    - examples: `83`
    - `--limit-examples 8 --max-steps 1` completed successfully
    - runtime: about `25.61s`
    - train loss: about `13.57695`
  - Identity-heavy seed dataset:
    - path: `/root/gemma4-qlora/data/cogiot_xiaoge_identity_seed.jsonl`
    - examples: `44`
    - `--limit-examples 8 --max-steps 1` completed successfully
    - runtime: about `25.37s`
    - train loss: about `13.57695`
- Important trainer compatibility note:
  - In this environment, `TRL SFTTrainer` tried to compute entropy from `outputs.logits`.
  - Gemma4 training forward returned only `loss`, not `logits`, causing an `AttributeError`.
  - A minimal patched trainer that delegates to the base `transformers.Trainer.compute_loss` was sufficient to complete the smoke test.

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Audit the remote host before installing anything | Gemma-class QLoRA is highly constrained by GPU VRAM and driver/CUDA compatibility. |
| Use the official Unsloth doc as the primary authority | The user linked that page directly and the requirements are time-sensitive. |
| Prefer `uv` + isolated venv on the remote host | This matches Unsloth's current install guidance and avoids breaking the system Python. |
| Keep the environment on `/root/gemma4-qlora/.venv` | Simple, isolated, and already validated on the target host. |
| Keep trying the 31B 4-bit path on this host | The user accepts 4-bit weights, CPU compatibility is fixed, and the download is now active. |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
|       |            |
| Newest NumPy wheel incompatible with host CPU (`x86_v2`) | Reinstalled `numpy==1.26.4` in the venv. |
| `huggingface_hub` failed when `all_proxy=socks5://...` was set | Installed `socksio`, after which SOCKS proxy downloads worked. |
| 31B model lookup initially failed | Corrected the model repo name to lowercase. |
| First detached background launch hit shell quoting issues | Re-launched via `bash` heredoc over SSH. |
| `hf_xet` large-file downloads stalled with `tls handshake eof` to `cas-server.xethub.hf.co` | Restarted with `HF_HUB_DISABLE_XET=1`; regular HF download resumed. |
| Regular HF large-file download was interrupted mid-body by `RemoteProtocolError` | The retrying downloader resumed and continued; no new transport change applied yet because the current path is still making net progress. |
| Local load failed after download because the sharded-index and tokenizer/processor files were missing | Downloaded the missing small files from HF and reran the load-check successfully. |
| `TRL SFTTrainer` expected `outputs.logits` for entropy metrics, but Gemma4 returned loss-only outputs | Used a minimal patched trainer override for the smoke test. |

## Resources
- Unsloth docs page provided by user: https://unsloth.ai/docs/models/gemma-4/train
- Unsloth Gemma 4 page lines showing A100 note and 31B example captured via browser tool on 2026-04-13.
- Unsloth requirements: https://docs.unsloth.ai/get-started/beginner-start-here/unsloth-requirements
- Unsloth pip install guide: https://unsloth.ai/docs/get-started/install/pip-install
- Unsloth multi-GPU guide: https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth
- HF mirror landing page: https://hf-mirror.com
- Unsloth Gemma 4 local run page: https://unsloth.ai/docs/models/gemma-4
- Remote download log: `/root/gemma4-qlora/logs/download_31b_4bit.log`
- Remote model dir: `/root/gemma4-qlora/models/gemma-4-31b-it-unsloth-bnb-4bit`
- Smoke test output dir: `/root/gemma4-qlora/runs/gemma4_31b_qlora_smoke_patch`

## Visual/Browser Findings
- The official page section `Unsloth Core (code-based) Guide` states:
  - `Below is a minimal SFT recipe`
  - `We also made notebooks for the larger Gemma 4 models but they need A100`
  - Entries include `Gemma-4-26B-A4B - A100 GPU` and `Gemma-4-31B - A100 GPU`
- The page's visible 31B code example loads `google-gemma-4-31b-it` with `FastVisionModel`, not `FastLanguageModel`.
- The official pip install guide shows:
  - `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - `uv venv unsloth_env --python 3.13`
  - `uv pip install unsloth --torch-backend=auto`
  - A venv-based fallback using `python -m venv ...` plus `pip install uv`
- The official multi-GPU guide distinguishes:
  - `device_map="balanced"` for spreading one model across multiple GPUs
  - DDP for replicating the model on every GPU, which does not solve the capacity problem for a too-large model
- The HF mirror landing page states it can be used by setting `HF_ENDPOINT=https://hf-mirror.com`, but direct testing from this remote host did not show better throughput than the current proxy path.
- The local run page currently states:
  - `Gemma-4-31B needs 20GB RAM (4-bit) or 34GB (8-bit).`
  - In the hardware table, `31B` shows `17–20 GB` for 4-bit inference.
- This page is specifically about `How to Run Locally`, not QLoRA fine-tuning.
