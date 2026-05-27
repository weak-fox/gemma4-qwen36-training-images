from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse


DEFAULT_MODEL_REPO = "unsloth/gemma-4-31b-it-unsloth-bnb-4bit"
DEFAULT_BAKED_MODEL_PATH = "/opt/models/gemma-4-31b-it-unsloth-bnb-4bit"
DEFAULT_WORKSPACE = "/workspace"
DEFAULT_MODEL_LOADER_TYPE = "fast_vision"
DEFAULT_MODEL_LOAD_IN_4BIT = True
DEFAULT_MODEL_4BIT_QUANT_TYPE = "nf4"
DEFAULT_MODEL_4BIT_COMPUTE_DTYPE = "auto"
DEFAULT_TRAIN_FINETUNE_SCOPE = "language_only"
DEFAULT_TRAIN_PRECISION = "auto"

SUPPORTED_MODEL_LOADER_TYPES = {"fast_language", "fast_vision", "hf_language", "hf_vision"}
SUPPORTED_MODEL_4BIT_QUANT_TYPES = {"nf4", "fp4"}
SUPPORTED_MODEL_4BIT_COMPUTE_DTYPES = {"auto", "float16", "bfloat16", "float32"}
SUPPORTED_TRAIN_FINETUNE_SCOPES = {"language_only", "all"}
SUPPORTED_TRAIN_PRECISIONS = {"auto", "float16", "bfloat16", "float32"}


class ConfigError(ValueError):
    """Raised when required runtime configuration is missing or invalid."""


def parse_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ConfigError(f"Invalid boolean value: {value!r}")


def get_optional_env(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def get_required_env(name: str) -> str:
    value = get_optional_env(name)
    if value is None:
        raise ConfigError(f"Missing required environment variable: {name}")
    return value


def get_int_env(name: str, default: int) -> int:
    value = get_optional_env(name)
    return default if value is None else int(value)


def get_float_env(name: str, default: float) -> float:
    value = get_optional_env(name)
    return default if value is None else float(value)


def get_choice_env(name: str, default: str, choices: set[str]) -> str:
    value = (get_optional_env(name) or default).lower()
    if value not in choices:
        allowed = ", ".join(sorted(choices))
        raise ConfigError(f"Unsupported {name}: {value!r}; expected one of: {allowed}")
    return value


def get_json_object_env(name: str) -> dict[str, Any]:
    value = get_optional_env(name)
    if value is None:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON object in {name}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ConfigError(f"{name} must be a JSON object")
    return parsed


def infer_default_model_name_or_path() -> str:
    baked_model_path = Path(DEFAULT_BAKED_MODEL_PATH)
    return str(baked_model_path) if baked_model_path.exists() else DEFAULT_MODEL_REPO


def infer_default_local_files_only() -> bool:
    return Path(DEFAULT_BAKED_MODEL_PATH).exists()


def normalize_minio_endpoint(endpoint: str, secure_default: bool) -> tuple[str, bool]:
    if endpoint.startswith("http://") or endpoint.startswith("https://"):
        parsed = urlparse(endpoint)
        if not parsed.netloc:
            raise ConfigError(f"Invalid MINIO_ENDPOINT: {endpoint!r}")
        return parsed.netloc, parsed.scheme == "https"
    return endpoint, secure_default


@dataclass(frozen=True)
class MinioSettings:
    endpoint: str
    access_key: str
    secret_key: str
    secure: bool
    region: Optional[str]

    @classmethod
    def from_env(cls) -> "MinioSettings":
        secure_default = parse_bool(get_optional_env("MINIO_SECURE"), default=True)
        endpoint, secure = normalize_minio_endpoint(get_required_env("MINIO_ENDPOINT"), secure_default)
        return cls(
            endpoint=endpoint,
            access_key=get_required_env("MINIO_ACCESS_KEY"),
            secret_key=get_required_env("MINIO_SECRET_KEY"),
            secure=secure,
            region=get_optional_env("MINIO_REGION"),
        )


@dataclass(frozen=True)
class TrainingSettings:
    repo_root: Path
    workspace_root: Path
    model_name_or_path: str
    model_local_files_only: bool
    model_loader_type: str
    model_load_in_4bit: bool
    model_4bit_quant_type: str
    model_4bit_compute_dtype: str
    model_chat_template_kwargs: dict[str, Any]
    finetune_scope: str
    train_precision: str
    base_result_uri: Optional[str]
    dataset_uri: Optional[str]
    dataset_path: Optional[Path]
    output_dir: Path
    output_uri: str
    dataset_split: str
    messages_column: str
    prompt_column: str
    completion_column: str
    system_column: str
    limit_examples: int
    max_seq_length: int
    batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: float
    max_steps: int
    learning_rate: float
    warmup_steps: int
    logging_steps: int
    save_steps: int
    save_total_limit: int
    save_adapter: bool
    prepare_kbit_training: bool

    @classmethod
    def from_env(cls) -> "TrainingSettings":
        repo_root = Path(__file__).resolve().parent.parent
        workspace_root = Path(get_optional_env("WORKSPACE_DIR") or DEFAULT_WORKSPACE)
        dataset_uri = get_optional_env("TRAIN_DATASET_URI")
        dataset_path_value = get_optional_env("TRAIN_DATASET_PATH")
        dataset_path = Path(dataset_path_value) if dataset_path_value else None

        if dataset_uri is None and dataset_path is None:
            raise ConfigError("One of TRAIN_DATASET_URI or TRAIN_DATASET_PATH must be set")

        return cls(
            repo_root=repo_root,
            workspace_root=workspace_root,
            model_name_or_path=get_optional_env("MODEL_NAME_OR_PATH") or infer_default_model_name_or_path(),
            model_local_files_only=parse_bool(
                get_optional_env("MODEL_LOCAL_FILES_ONLY"),
                default=infer_default_local_files_only(),
            ),
            model_loader_type=get_choice_env(
                "MODEL_LOADER_TYPE",
                DEFAULT_MODEL_LOADER_TYPE,
                SUPPORTED_MODEL_LOADER_TYPES,
            ),
            model_load_in_4bit=parse_bool(
                get_optional_env("MODEL_LOAD_IN_4BIT"),
                default=DEFAULT_MODEL_LOAD_IN_4BIT,
            ),
            model_4bit_quant_type=get_choice_env(
                "MODEL_4BIT_QUANT_TYPE",
                DEFAULT_MODEL_4BIT_QUANT_TYPE,
                SUPPORTED_MODEL_4BIT_QUANT_TYPES,
            ),
            model_4bit_compute_dtype=get_choice_env(
                "MODEL_4BIT_COMPUTE_DTYPE",
                DEFAULT_MODEL_4BIT_COMPUTE_DTYPE,
                SUPPORTED_MODEL_4BIT_COMPUTE_DTYPES,
            ),
            model_chat_template_kwargs=get_json_object_env("MODEL_CHAT_TEMPLATE_KWARGS_JSON"),
            finetune_scope=get_choice_env(
                "TRAIN_FINETUNE_SCOPE",
                DEFAULT_TRAIN_FINETUNE_SCOPE,
                SUPPORTED_TRAIN_FINETUNE_SCOPES,
            ),
            train_precision=get_choice_env(
                "TRAIN_PRECISION",
                DEFAULT_TRAIN_PRECISION,
                SUPPORTED_TRAIN_PRECISIONS,
            ),
            base_result_uri=get_optional_env("TRAIN_BASE_RESULT_URI"),
            dataset_uri=dataset_uri,
            dataset_path=dataset_path,
            output_dir=Path(get_optional_env("TRAIN_OUTPUT_DIR") or f"{workspace_root}/output/run"),
            output_uri=get_required_env("TRAIN_OUTPUT_URI"),
            dataset_split=get_optional_env("TRAIN_DATASET_SPLIT") or "train",
            messages_column=get_optional_env("TRAIN_MESSAGES_COLUMN") or "messages",
            prompt_column=get_optional_env("TRAIN_PROMPT_COLUMN") or "prompt",
            completion_column=get_optional_env("TRAIN_COMPLETION_COLUMN") or "completion",
            system_column=get_optional_env("TRAIN_SYSTEM_COLUMN") or "",
            limit_examples=get_int_env("TRAIN_LIMIT_EXAMPLES", 0),
            max_seq_length=get_int_env("TRAIN_MAX_SEQ_LENGTH", 512),
            batch_size=get_int_env("TRAIN_BATCH_SIZE", 1),
            gradient_accumulation_steps=get_int_env("TRAIN_GRAD_ACCUM_STEPS", 1),
            num_train_epochs=get_float_env("TRAIN_NUM_EPOCHS", 1.0),
            max_steps=get_int_env("TRAIN_MAX_STEPS", -1),
            learning_rate=get_float_env("TRAIN_LEARNING_RATE", 2e-4),
            warmup_steps=get_int_env("TRAIN_WARMUP_STEPS", 0),
            logging_steps=get_int_env("TRAIN_LOGGING_STEPS", 1),
            save_steps=get_int_env("TRAIN_SAVE_STEPS", 50),
            save_total_limit=get_int_env("TRAIN_SAVE_TOTAL_LIMIT", 2),
            save_adapter=parse_bool(get_optional_env("TRAIN_SAVE_ADAPTER"), default=True),
            prepare_kbit_training=parse_bool(get_optional_env("TRAIN_PREPARE_KBIT"), default=True),
        )


@dataclass(frozen=True)
class ServingSettings:
    workspace_root: Path
    artifact_root: Path
    model_name_or_path: str
    model_local_files_only: bool
    model_loader_type: str
    model_load_in_4bit: bool
    model_4bit_quant_type: str
    model_4bit_compute_dtype: str
    model_chat_template_kwargs: dict[str, Any]
    artifact_uri: Optional[str]
    host: str
    port: int
    max_new_tokens_default: int
    temperature_default: float
    top_p_default: float
    response_model_name: Optional[str]

    @classmethod
    def from_env(cls) -> "ServingSettings":
        workspace_root = Path(get_optional_env("WORKSPACE_DIR") or DEFAULT_WORKSPACE)
        return cls(
            workspace_root=workspace_root,
            artifact_root=Path(get_optional_env("SERVE_ARTIFACT_DIR") or f"{workspace_root}/artifacts/run"),
            model_name_or_path=get_optional_env("MODEL_NAME_OR_PATH") or infer_default_model_name_or_path(),
            model_local_files_only=parse_bool(
                get_optional_env("MODEL_LOCAL_FILES_ONLY"),
                default=infer_default_local_files_only(),
            ),
            model_loader_type=get_choice_env(
                "MODEL_LOADER_TYPE",
                DEFAULT_MODEL_LOADER_TYPE,
                SUPPORTED_MODEL_LOADER_TYPES,
            ),
            model_load_in_4bit=parse_bool(
                get_optional_env("MODEL_LOAD_IN_4BIT"),
                default=DEFAULT_MODEL_LOAD_IN_4BIT,
            ),
            model_4bit_quant_type=get_choice_env(
                "MODEL_4BIT_QUANT_TYPE",
                DEFAULT_MODEL_4BIT_QUANT_TYPE,
                SUPPORTED_MODEL_4BIT_QUANT_TYPES,
            ),
            model_4bit_compute_dtype=get_choice_env(
                "MODEL_4BIT_COMPUTE_DTYPE",
                DEFAULT_MODEL_4BIT_COMPUTE_DTYPE,
                SUPPORTED_MODEL_4BIT_COMPUTE_DTYPES,
            ),
            model_chat_template_kwargs=get_json_object_env("MODEL_CHAT_TEMPLATE_KWARGS_JSON"),
            artifact_uri=get_optional_env("SERVE_ARTIFACT_URI"),
            host=get_optional_env("SERVE_HOST") or "0.0.0.0",
            port=get_int_env("SERVE_PORT", 8000),
            max_new_tokens_default=get_int_env("SERVE_MAX_NEW_TOKENS_DEFAULT", 256),
            temperature_default=get_float_env("SERVE_TEMPERATURE_DEFAULT", 0.2),
            top_p_default=get_float_env("SERVE_TOP_P_DEFAULT", 0.95),
            response_model_name=get_optional_env("SERVE_RESPONSE_MODEL_NAME"),
        )


def get_app_mode() -> str:
    mode = (get_optional_env("APP_MODE") or "train").lower()
    if mode not in {"train", "serve"}:
        raise ConfigError(f"Unsupported APP_MODE: {mode!r}")
    return mode
