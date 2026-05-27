from __future__ import annotations

import inspect
from typing import Any, Callable

from runtime.config import ConfigError

try:
    import unsloth  # noqa: F401
except ModuleNotFoundError:
    unsloth = None


def torch_dtype_from_name(dtype_name: str) -> Any:
    import torch

    if dtype_name == "auto":
        if torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    try:
        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype_name]
    except KeyError as exc:
        raise ConfigError(f"Unsupported MODEL_4BIT_COMPUTE_DTYPE: {dtype_name!r}") from exc


def build_bnb_4bit_config(quant_type: str, compute_dtype_name: str) -> Any:
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype=torch_dtype_from_name(compute_dtype_name),
        bnb_4bit_use_double_quant=True,
    )


def filter_supported_kwargs(callable_obj: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs

    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return kwargs

    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def build_unsloth_from_pretrained_kwargs(
    *,
    model_name_or_path: str,
    max_seq_length: int,
    model_load_in_4bit: bool,
    model_4bit_quant_type: str,
    model_4bit_compute_dtype: str,
    local_files_only: bool,
    device_map: str = "balanced",
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model_name": model_name_or_path,
        "max_seq_length": max_seq_length,
        "dtype": None,
        "load_in_4bit": model_load_in_4bit,
        "device_map": device_map,
        "local_files_only": local_files_only,
    }
    if model_load_in_4bit:
        kwargs.update(
            {
                "quantization_config": build_bnb_4bit_config(
                    model_4bit_quant_type,
                    model_4bit_compute_dtype,
                ),
            }
        )
    return kwargs


def get_unsloth_model_loader(model_loader_type: str) -> Any:
    if model_loader_type == "fast_language":
        from unsloth import FastLanguageModel

        return FastLanguageModel
    if model_loader_type == "fast_vision":
        from unsloth import FastVisionModel

        return FastVisionModel
    raise ConfigError(f"Unsupported MODEL_LOADER_TYPE: {model_loader_type!r}")


def call_with_supported_kwargs(callable_obj: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    return callable_obj(*args, **filter_supported_kwargs(callable_obj, kwargs))


def load_unsloth_model(
    *,
    model_loader_type: str,
    model_name_or_path: str,
    max_seq_length: int,
    model_load_in_4bit: bool,
    model_4bit_quant_type: str,
    model_4bit_compute_dtype: str,
    local_files_only: bool,
    device_map: str = "balanced",
) -> tuple[Any, Any, Any]:
    loader = get_unsloth_model_loader(model_loader_type)
    kwargs = build_unsloth_from_pretrained_kwargs(
        model_name_or_path=model_name_or_path,
        max_seq_length=max_seq_length,
        model_load_in_4bit=model_load_in_4bit,
        model_4bit_quant_type=model_4bit_quant_type,
        model_4bit_compute_dtype=model_4bit_compute_dtype,
        local_files_only=local_files_only,
        device_map=device_map,
    )
    model, processor = call_with_supported_kwargs(loader.from_pretrained, **kwargs)
    return loader, model, processor


def prepare_for_training(loader: Any, model: Any) -> None:
    for_training = getattr(loader, "for_training", None)
    if callable(for_training):
        for_training(model)


def prepare_for_inference(loader: Any, model: Any) -> None:
    for_inference = getattr(loader, "for_inference", None)
    if callable(for_inference):
        for_inference(model)
