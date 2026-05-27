# Progress Log

## Session: 2026-04-13

### Phase 1: Requirements & Discovery
- **Status:** complete
- **Started:** 2026-04-13
- Actions taken:
  - Loaded the mandatory process skills and created persistent planning files.
  - Captured the user goal, constraints, and proxy requirement.
  - Checked the official Unsloth Gemma 4 training page.
  - Confirmed the current doc notes that Gemma 4 31B notebooks require A100 GPUs.
  - Audited the remote host at a high level and found 2x Tesla T4 GPUs.
- Files created/modified:
  - `task_plan.md` (created)
  - `findings.md` (created)
  - `progress.md` (created)

### Phase 2: Remote Environment Audit
- **Status:** complete
- Actions taken:
  - Verified SSH access to `root@105.100.31.190`.
  - Collected GPU, driver, CUDA, Python, RAM, and disk information.
  - Began comparing official requirements with actual hardware.
  - Confirmed the remote host is Ubuntu 22.04.4, direct Hugging Face access times out, and proxy access is required.
- Files created/modified:
  - `findings.md` (updated)
  - `progress.md` (updated)

### Phase 3: Training Setup & Execution
- **Status:** complete
- Actions taken:
  - Installed `uv` and Python `3.11.15`.
  - Created `/root/gemma4-qlora/.venv`.
  - Installed `unsloth`, `torch`, and related dependencies.
  - Diagnosed and fixed a host-CPU compatibility issue by downgrading `numpy` to `1.26.4`.
  - Verified `torch.cuda.is_available() == True` and `torch.cuda.device_count() == 2`.
  - Verified `unsloth` imports successfully after the NumPy fix.
  - Diagnosed `huggingface_hub` proxy failure to missing SOCKS support under `all_proxy=socks5://...`.
  - Verified `hf_hub_download()` works after unsetting `all_proxy` and keeping only HTTP proxies.
  - Attempted to load `unsloth/gemma-4-31b-it-unsloth-bnb-4bit` with `device_map=\"balanced\"`.
  - Confirmed the full 31B 4-bit model download path is currently too slow to complete.
  - Re-checked the remote CPU and confirmed newer vector instruction support is now present.
  - Re-checked Unsloth docs and clarified that the `20GB` / `17-20GB` figure is for local 4-bit inference, not training.
  - Upgraded NumPy back to `2.4.4` and verified imports now succeed with the new CPU flags.
  - Installed `socksio` so `huggingface_hub` can use the provided SOCKS proxy directly.
  - Started a persistent background snapshot download for `unsloth/gemma-4-31b-it-unsloth-bnb-4bit`.
  - Verified the background job is active with PID `8131` and that partial shard files are growing in cache.
  - Re-checked the latest state and confirmed 2 full model shards are now present in the local model directory.
  - Started a second background watcher that will automatically run a `FastVisionModel.from_pretrained(...)` load check after the last shard is downloaded.
  - Investigated a later slowdown and found the Xet path had stalled on TLS handshake EOF errors to `cas-server.xethub.hf.co`.
  - Restarted the downloader with `HF_HUB_DISABLE_XET=1`, preserving the completed shards.
  - Verified the non-Xet downloader is actively writing to the local Hugging Face `.incomplete` cache for `model-00003`.
  - Verified later that `model-00004` completed and the downloader continued onto the next shard.
  - Captured a `RemoteProtocolError` on the regular HF path where the peer closed the body mid-stream.
  - Confirmed that despite the disconnect, the current retrying downloader still made forward progress at about `146,800,640` bytes over 60 seconds.
  - Confirmed all 6 model shard files are now present locally.
  - Identified that `model.safetensors.index.json`, `processor_config.json`, `tokenizer.json`, and `tokenizer_config.json` were missing and downloaded them separately.
  - Reran local model loading and confirmed the 31B 4-bit model loads successfully across both T4 GPUs.
  - Ran a 1-step QLoRA smoke test with a minimal patched trainer and confirmed the training step completed.
  - Searched both the local workspace and the remote training directory for a real dataset and confirmed that none is currently staged there.
  - Created `scripts/train_gemma4_31b_qlora.py` as a reusable parameterized training entrypoint.
  - Created `scripts/run_remote_gemma4_31b_qlora.sh` as a reusable remote launcher.
  - Created `examples/smoke_messages.jsonl` as a minimal reproducible smoke-test dataset.
  - Uploaded the reusable script, launcher, and sample dataset to the remote host.
  - Ran the wrapper-based smoke test remotely and confirmed the scripted path also completes successfully.
  - Designed a synthetic `cogiot / 小舸` persona with stable identity answers and a pragmatic response style.
  - Created `scripts/build_cogiot_persona_dataset.py` as a reproducible persona-data generator.
  - Materialized `data/cogiot_xiaoge_train.jsonl` with 83 examples.
  - Materialized `data/cogiot_xiaoge_identity_seed.jsonl` with 44 identity-heavy examples.
  - Added `data/cogiot_xiaoge_dataset_notes.md` to document the persona design and usage recommendations.
  - Uploaded the persona datasets and notes to the remote host.
  - Ran a wrapper-based smoke test against the full persona dataset and confirmed the training path succeeds.
  - Ran a wrapper-based smoke test against the identity-heavy seed dataset and confirmed the training path succeeds.
- Files created/modified:
  - `findings.md` (updated)
  - `progress.md` (updated)
  - `gemma4_31b_qlora_runbook_zh.md` (created)
  - `scripts/train_gemma4_31b_qlora.py` (created)
  - `scripts/run_remote_gemma4_31b_qlora.sh` (created)
  - `examples/smoke_messages.jsonl` (created)
  - `scripts/build_cogiot_persona_dataset.py` (created)
  - `data/cogiot_xiaoge_train.jsonl` (created)
  - `data/cogiot_xiaoge_identity_seed.jsonl` (created)
  - `data/cogiot_xiaoge_dataset_notes.md` (created)

## Test Results
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| Planning files created | `task_plan.md`, `findings.md`, `progress.md` | Files exist with task context | Files created successfully | pass |
| SSH access | `ssh root@105.100.31.190 'hostname; whoami; date'` | Remote shell reachable | Reachable; host `lyp-t4` | pass |
| Python upgrade | `uv python install 3.11` | Python 3.11 environment available | `Python 3.11.15` in venv | pass |
| Torch import | `import torch` | Torch imports and sees GPUs | Passed after pinning `numpy==1.26.4`; 2 GPUs visible | pass |
| Unsloth import | `import unsloth` | Unsloth imports cleanly | Passed after NumPy fix | pass |
| HF hub config download | `hf_hub_download(..., 'config.json')` | Config downloaded through proxy | Passed when `all_proxy` was unset | pass |
| 31B repo load start | `FastVisionModel.from_pretrained(...)` | Model download and load begins | Began download after repo/proxy fixes | partial |
| 31B full fetch feasibility | Timed ranged download over proxy | Reasonable throughput | Too slow; full 25 GB download infeasible on current path | fail |
| 31B ranged file verification | `curl --range ... model-00006-of-00006.safetensors` via proxy | Stable binary transfer | Failed with SSL EOF on fresh verification | fail |
| CPU compatibility retest | Upgrade to `numpy==2.4.4` and import stack | Modern NumPy works after CPU change | Passed | pass |
| SOCKS proxy retest | `hf_hub_download(...config.json...)` with `all_proxy=socks5://...` | HF hub works through SOCKS | Passed after installing `socksio` | pass |
| Background 31B download launch | `snapshot_download(...)` in `nohup` loop | Detached download process starts | Passed, PID `8131` | pass |
| Background 31B data growth | Inspect cache and local model dir | Partial files should accumulate | Passed, cache at about `4.6G` with multiple shard `.incomplete` files | pass |
| 31B completed shard check | Inspect local model dir | Full shard files should appear over time | Passed, `model-00001` and `model-00002` are complete | pass |
| Post-download watcher launch | Background wait + load-check process | Detached watcher should start | Passed, PID `29084` | pass |
| Xet stall diagnosis | Socket state + xet log + growth sample | Need root cause evidence | Found `CLOSE_WAIT`, zero growth, and `tls handshake eof` in Xet log | pass |
| Non-Xet restart verification | Growth in `.cache/huggingface/download/*.incomplete` | Download should resume | Passed, +136,314,880 bytes in 30s | pass |
| Non-Xet later progress check | Full files + 60-second cache growth | Need to know if it truly stopped | `model-00004` completed; cache still grew +146,800,640 bytes in 60s | pass |
| Final shard presence check | `model-00001` ... `model-00006` all present | Full local checkpoint should exist | Passed, 6 shard files found | pass |
| Local load-check after metadata fix | `FastVisionModel.from_pretrained(local_dir, device_map=\"balanced\")` | Model should load across 2 GPUs | Passed | pass |
| 1-step smoke training | Patched `SFTTrainer` + `max_steps=1` | One QLoRA step should complete | Passed in ~50.37s | pass |
| Script syntax check | `python3 -m py_compile scripts/train_gemma4_31b_qlora.py` | Script should parse cleanly | Passed | pass |
| Remote script help check | `python train_gemma4_31b_qlora.py --help` | Script should import cleanly and show CLI | Passed | pass |
| Wrapper-based remote smoke training | `./run_remote_gemma4_31b_qlora.sh --max-steps 1 --save-steps 0` | Scriptized training path should complete 1 step | Passed in ~27.33s | pass |
| Persona dataset generation | `python3 scripts/build_cogiot_persona_dataset.py` | Build reproducible persona datasets | Passed; 83 full examples and 44 seed examples generated | pass |
| Full persona dataset smoke | `DATASET_PATH=/root/gemma4-qlora/data/cogiot_xiaoge_train.jsonl ... --limit-examples 8 --max-steps 1` | Full persona dataset should parse and train for 1 step | Passed in ~25.61s | pass |
| Identity-seed dataset smoke | `DATASET_PATH=/root/gemma4-qlora/data/cogiot_xiaoge_identity_seed.jsonl ... --limit-examples 8 --max-steps 1` | Identity-heavy seed dataset should parse and train for 1 step | Passed in ~25.37s | pass |

## Error Log
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
|           |       | 1       |            |
| 2026-04-13 | NumPy `x86_v2` baseline / duplicate-load import failure | 1 | Replaced NumPy 2.4.4 with 1.26.4 in the venv |
| 2026-04-13 | HF hub SOCKS proxy import error (`socksio` missing) | 1 | Removed `all_proxy` for Python downloads |
| 2026-04-13 | Wrong 31B repo name case caused config lookup failure | 1 | Switched to lowercase model repo ID |
| 2026-04-13 | 31B checkpoint transfer too slow via available proxy | 1 | Stopped the long-running load attempt and documented the blocker |
| 2026-04-13 | Initial background download command hit shell quoting error | 1 | Rewrote remote launch using `bash` heredoc over SSH |
| 2026-04-13 | `hf_xet` large-file transfer stalled with TLS handshake EOF | 1 | Disabled Xet and restarted the downloader using regular HF downloads |
| 2026-04-13 | Regular HF large-file response closed mid-body (`RemoteProtocolError`) | 1 | Left the retrying downloader in place because it continued to make net forward progress |
| 2026-04-13 | Local load failed because sharded index and tokenizer/processor files were absent | 1 | Downloaded the missing small files and reran the load-check |
| 2026-04-13 | `TRL SFTTrainer` crashed because `outputs.logits` was missing | 1 | Patched the smoke-test trainer to use the base Trainer loss path instead of TRL entropy metrics |

## 5-Question Reboot Check
| Question | Answer |
|----------|--------|
| Where am I? | Delivery phase with download, local load, and 1-step smoke training all confirmed |
| Where am I going? | Final handoff with the exact reproducible Chinese runbook |
| What's the goal? | Run Unsloth Gemma 4 31B QLoRA on the remote host and document the exact process |
| What have I learned? | Xet is unusable here, but the regular HF path plus a small-file cleanup yields a complete checkpoint, and a patched TRL trainer can run a 1-step QLoRA smoke test on 2x T4 |
| What have I done? | Audited the host, fixed compatibility issues, completed the 31B 4-bit checkpoint download, fixed missing metadata files, verified local multi-GPU loading, and completed a 1-step smoke train |
