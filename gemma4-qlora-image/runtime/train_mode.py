from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from runtime.config import MinioSettings, TrainingSettings
from runtime.storage import MinioStorage, parse_s3_uri


def _resolve_dataset_path(settings: TrainingSettings, storage: MinioStorage) -> Path:
    if settings.dataset_uri:
        s3_uri = parse_s3_uri(settings.dataset_uri)
        file_name = Path(s3_uri.key).name or "dataset.jsonl"
        local_path = settings.workspace_root / "input" / file_name
        print(f"DATASET_DOWNLOAD {settings.dataset_uri} -> {local_path}", flush=True)
        return storage.download_object(settings.dataset_uri, local_path)

    assert settings.dataset_path is not None
    return settings.dataset_path


def _build_train_command(settings: TrainingSettings, dataset_path: Path) -> list[str]:
    command = [
        sys.executable,
        str(settings.repo_root / "scripts" / "train_gemma4_31b_qlora.py"),
        "--dataset-path",
        str(dataset_path),
        "--dataset-split",
        settings.dataset_split,
        "--messages-column",
        settings.messages_column,
        "--prompt-column",
        settings.prompt_column,
        "--completion-column",
        settings.completion_column,
        "--model-dir",
        settings.model_name_or_path,
        "--output-dir",
        str(settings.output_dir),
        "--max-seq-length",
        str(settings.max_seq_length),
        "--per-device-train-batch-size",
        str(settings.batch_size),
        "--gradient-accumulation-steps",
        str(settings.gradient_accumulation_steps),
        "--num-train-epochs",
        str(settings.num_train_epochs),
        "--max-steps",
        str(settings.max_steps),
        "--learning-rate",
        str(settings.learning_rate),
        "--warmup-steps",
        str(settings.warmup_steps),
        "--logging-steps",
        str(settings.logging_steps),
        "--save-steps",
        str(settings.save_steps),
        "--save-total-limit",
        str(settings.save_total_limit),
    ]

    if settings.system_column:
        command.extend(["--system-column", settings.system_column])
    if settings.limit_examples > 0:
        command.extend(["--limit-examples", str(settings.limit_examples)])
    if settings.model_local_files_only:
        command.append("--local-files-only")
    if settings.save_adapter:
        command.append("--save-adapter")
    if settings.base_result_uri:
        command.extend(["--base-result-dir", str(settings.workspace_root / "artifacts" / "base-result")])

    return command


def run_training(settings: TrainingSettings) -> None:
    storage = MinioStorage(MinioSettings.from_env())
    settings.workspace_root.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = _resolve_dataset_path(settings, storage)
    if settings.base_result_uri:
        base_result_root = settings.workspace_root / "artifacts" / "base-result"
        print(f"BASE_RESULT_DOWNLOAD {settings.base_result_uri} -> {base_result_root}", flush=True)
        storage.download_prefix(settings.base_result_uri, base_result_root)
    command = _build_train_command(settings, dataset_path)

    print("TRAIN_COMMAND", " ".join(command), flush=True)
    subprocess.run(command, cwd=settings.workspace_root, check=True)

    print(f"TRAIN_UPLOAD {settings.output_dir} -> {settings.output_uri}", flush=True)
    storage.upload_directory(settings.output_dir, settings.output_uri)
    print("TRAIN_UPLOAD_DONE", flush=True)
