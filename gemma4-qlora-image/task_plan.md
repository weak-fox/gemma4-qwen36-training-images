# Task Plan: Run Unsloth Gemma 4 31B QLoRA on remote server

## Goal
Run the official Unsloth Gemma 4 training flow for a 31B QLoRA model on `root@105.100.31.190`, verify it works end to end, and produce a complete step-by-step Chinese guide tailored for a China-based network environment with the provided proxy.

## Current Phase
Phase 5

## Phases
### Phase 1: Requirements & Discovery
- [x] Understand user intent
- [x] Identify constraints and requirements
- [x] Document findings in findings.md
- **Status:** complete

### Phase 2: Remote Environment Audit
- [x] Verify SSH access
- [x] Inspect GPU, memory, disk, CUDA, Python, and network/proxy status
- [x] Determine whether hardware can realistically run Gemma 4 31B QLoRA
- **Status:** complete

### Phase 3: Training Setup & Execution
- [x] Reproduce the official Unsloth setup on the remote server
- [x] Resolve dependency, access, or proxy issues
- [x] Launch a minimal successful model load run
- [x] Launch a minimal successful training smoke test
- **Status:** complete

### Phase 4: Verification
- [x] Confirm 31B 4-bit model weights download and local load work
- [x] Capture exact commands, logs, and any deviations from docs
- [x] Record operational caveats
- [x] Confirm one-step training can run on the target hardware
- **Status:** complete

### Phase 5: Delivery
- [x] Write the full step-by-step workflow in Chinese
- [x] Include proxy handling, troubleshooting, and next actions
- [x] Package the verified flow into reusable scripts
- [x] Produce a synthetic persona dataset that can demonstrate fine-tuning effect
- [x] Deliver concise results with evidence
- **Status:** in_progress

## Key Questions
1. What does the official Unsloth Gemma 4 training page currently require for 31B QLoRA?
2. Does `root@105.100.31.190` have enough GPU memory and software prerequisites to run it?
3. What exact commands and environment changes are needed in a China network environment?

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Use file-based planning in the project root | This task is multi-step, remote, and likely to require many tool calls and retries. |
| Verify official docs before remote execution | Model names, supported versions, and install steps can change. |
| Use the provided proxies with `socksio` installed in the venv | `huggingface_hub` can now use `all_proxy=socks5://...` directly. |
| Keep `numpy==2.4.4` on this host after the CPU flag upgrade | The previous incompatibility is gone and the modern wheel now imports cleanly. |
| Run the 31B 4-bit checkpoint download as a persistent background job | The download is large and needs retry-safe execution independent of this session. |
| Disable Xet for this download path | The proxy intermittently breaks TLS handshakes to `cas-server.xethub.hf.co`, while regular HF download is progressing. |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
|       | 1       |            |
| `RuntimeError: No config file found` for 31B repo | 1 | Fixed by switching repo name to lowercase `unsloth/gemma-4-31b-it-unsloth-bnb-4bit` and unblocking `huggingface_hub` proxy handling. |
| `ImportError/RuntimeError` from NumPy x86_v2 baseline | 1 | Downgraded to `numpy==1.26.4`, which imports successfully on this CPU. |
| `ImportError: Using SOCKS proxy, but the 'socksio' package is not installed` | 1 | Installed `socksio` in the venv, allowing `all_proxy` usage. |
| First detached download launch failed due to shell quoting | 1 | Re-launched the background downloader using `bash` heredoc over SSH. |
| Xet download path stalled on large shard | 1 | Restarted the downloader with `HF_HUB_DISABLE_XET=1` and HTTP(S) proxy only. |
| Auto load-check initially failed because small repo metadata files were missing | 1 | Downloaded `model.safetensors.index.json`, `processor_config.json`, `tokenizer.json`, and `tokenizer_config.json`, then reran load successfully. |
| `TRL SFTTrainer` crashed because `outputs.logits` was `None` for Gemma4 | 1 | Used a minimal patched `SFTTrainer` override that skips TRL's entropy-on-logits path and relies on the model loss directly. |

## Notes
- Use the user-provided proxy where remote web access is blocked.
- Avoid repeating failed remote setup steps; log the failure and adjust the approach.
