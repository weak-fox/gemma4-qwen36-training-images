from __future__ import annotations

import json
from pathlib import Path
from typing import Any


QUANT_STATE_SUFFIXES = (
    "absmax",
    "quant_map",
    "nested_absmax",
    "nested_quant_map",
)


def _append_unique(items: list[str], item: str) -> None:
    if item not in items:
        items.append(item)


def bnb_weight_key_candidates(module_name: str) -> list[str]:
    candidates: list[str] = []
    _append_unique(candidates, module_name + ".weight")

    normalized_names = [module_name]
    if module_name.endswith(".base_layer"):
        _append_unique(normalized_names, module_name[: -len(".base_layer")])
    for prefix in ("base_model.model.", "base_model.model.model."):
        for normalized_name in list(normalized_names):
            if normalized_name.startswith(prefix):
                _append_unique(normalized_names, normalized_name[len(prefix) :])

    for normalized_name in normalized_names:
        if normalized_name.startswith("model."):
            suffix = normalized_name[len("model.") :]
            _append_unique(candidates, "model." + suffix + ".weight")
            if not suffix.startswith("language_model."):
                _append_unique(candidates, "model.language_model." + suffix + ".weight")
        else:
            _append_unique(candidates, "model." + normalized_name + ".weight")
            if not normalized_name.startswith("language_model."):
                _append_unique(candidates, "model.language_model." + normalized_name + ".weight")

    return candidates


def _load_index(model_name_or_path: str) -> dict[str, Any] | None:
    model_dir = Path(model_name_or_path)
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        return None
    with index_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _find_quant_state_suffix(weight_map: dict[str, str], weight_key: str) -> str | None:
    for quant_type in ("nf4", "fp4"):
        suffix = f"quant_state.bitsandbytes__{quant_type}"
        if weight_key + "." + suffix in weight_map:
            return suffix
    return None


def _read_tensor(model_dir: Path, weight_map: dict[str, str], tensor_key: str) -> Any:
    from safetensors import safe_open

    shard = weight_map[tensor_key]
    with safe_open(str(model_dir / shard), framework="pt", device="cpu") as handle:
        return handle.get_tensor(tensor_key)


def _read_quantized_stats(model_dir: Path, weight_map: dict[str, str], weight_key: str) -> dict[str, Any] | None:
    quant_state_suffix = _find_quant_state_suffix(weight_map, weight_key)
    if quant_state_suffix is None:
        return None

    suffixes = QUANT_STATE_SUFFIXES + (quant_state_suffix,)
    stats: dict[str, Any] = {}
    for suffix in suffixes:
        tensor_key = weight_key + "." + suffix
        if tensor_key not in weight_map:
            return None
        stats[tensor_key] = _read_tensor(model_dir, weight_map, tensor_key)
    return stats


def restore_bnb_4bit_quant_states(model: Any, model_name_or_path: str, *, force: bool = True) -> int:
    """Restore serialized BnB 4-bit quant_state for packed Linear4bit modules.

    Some HF/BnB loading paths attach a Params4bit object but leave it in a state where
    forward still sees the packed uint8 shape. Rebuilding from the safetensors sidecar
    metadata keeps the weight and quant_state paired exactly as saved_pretrained wrote
    them, and also handles PEFT's base_model.* module-name prefixes.
    """

    index = _load_index(model_name_or_path)
    if not index:
        return 0

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        return 0

    try:
        import bitsandbytes as bnb
    except ImportError:
        return 0

    model_dir = Path(model_name_or_path)
    restored = 0
    for module_name, module in model.named_modules():
        if module.__class__.__name__ != "Linear4bit":
            continue
        if hasattr(module, "base_layer"):
            continue
        weight = getattr(module, "weight", None)
        if weight is None:
            continue
        if len(getattr(weight, "shape", ())) != 2 or weight.shape[1] != 1:
            continue
        if not force and getattr(weight, "quant_state", None) is not None:
            continue

        stats = None
        for weight_key in bnb_weight_key_candidates(module_name):
            if weight_key in weight_map:
                stats = _read_quantized_stats(model_dir, weight_map, weight_key)
                if stats is not None:
                    break
        if stats is None:
            continue

        module.weight = bnb.nn.Params4bit.from_prequantized(
            data=weight.data,
            quantized_stats=stats,
            requires_grad=False,
            device=weight.device,
            module=module,
        )
        restored += 1

    return restored


def restore_missing_bnb_4bit_quant_states(model: Any, model_name_or_path: str) -> int:
    """Backward-compatible wrapper for callers that only want missing quant_state repair."""

    return restore_bnb_4bit_quant_states(model, model_name_or_path, force=False)
