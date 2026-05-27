from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="unsloth/Qwen3.6-27B",
    local_dir="/root/gemma4-qlora/models/qwen3.6-27b-bnb-4bit",
    resume_download=True,
    max_workers=1,
)
