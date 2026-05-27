#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import unsloth  # noqa: F401
import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from peft import PeftModel
from unsloth.trainer import UnslothVisionDataCollator
from transformers import Trainer as HFTrainer
from trl import SFTConfig, SFTTrainer

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.model_loading import (  # noqa: E402
    call_with_supported_kwargs,
    load_unsloth_model,
    prepare_for_training,
)


DEFAULT_MODEL_DIR = "/root/gemma4-qlora/models/gemma-4-31b-it-unsloth-bnb-4bit"
DEFAULT_OUTPUT_DIR = "/root/gemma4-qlora/runs/gemma4_31b_qlora_run"


class PatchedSFTTrainer(SFTTrainer):
    """Skip TRL's entropy-on-logits path for model outputs that only expose loss."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        train_inputs = dict(inputs)
        train_inputs["use_cache"] = False
        loss, outputs = HFTrainer.compute_loss(
            self,
            model,
            train_inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )
        return (loss, outputs) if return_outputs else loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an Unsloth FastLanguageModel or FastVisionModel with QLoRA on a local dataset."
    )
    parser.add_argument("--dataset-path", required=True, help="Path to json/jsonl/csv/parquet file or load_from_disk directory.")
    parser.add_argument("--dataset-split", default="train", help="Dataset split when loading from a DatasetDict or hub-style layout.")
    parser.add_argument("--messages-column", default="messages", help="Column containing OpenAI-style chat messages.")
    parser.add_argument("--prompt-column", default="prompt", help="Prompt column used when messages are not present.")
    parser.add_argument("--completion-column", default="completion", help="Completion column used when messages are not present.")
    parser.add_argument("--system-column", default="", help="Optional system prompt column.")
    parser.add_argument("--limit-examples", type=int, default=0, help="Use only the first N examples for smoke tests.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Local model directory or model id.")
    parser.add_argument(
        "--model-loader-type",
        choices=["fast_language", "fast_vision", "hf_language", "hf_vision"],
        default="fast_vision",
    )
    parser.add_argument("--model-load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model-4bit-quant-type", choices=["nf4", "fp4"], default="nf4")
    parser.add_argument(
        "--model-4bit-compute-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument(
        "--model-chat-template-kwargs-json",
        default="",
        help='JSON object passed to processor.apply_chat_template, for example {"enable_thinking": false}.',
    )
    parser.add_argument("--finetune-scope", choices=["language_only", "all"], default="language_only")
    parser.add_argument("--precision", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for checkpoints and logs.")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1, help="Override epochs when > 0.")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--local-files-only", action="store_true", help="Do not hit Hugging Face during model load.")
    parser.add_argument("--fp16", action="store_true", help="Force fp16 training. Enabled by default when bf16 is not selected.")
    parser.add_argument("--bf16", action="store_true", help="Enable bf16 training if hardware supports it.")
    parser.add_argument("--resume-from-checkpoint", default="", help="Resume training from a previous checkpoint directory.")
    parser.add_argument("--base-result-dir", default="", help="Training result directory that contains adapter/ for continued fine-tuning.")
    parser.add_argument("--save-adapter", action="store_true", help="Save the trained adapter and processor after training.")
    return parser.parse_args()


def load_training_dataset(dataset_path: str, dataset_split: str) -> Dataset:
    path = Path(dataset_path)
    if path.is_dir():
        dataset_or_dict = load_from_disk(str(path))
        if isinstance(dataset_or_dict, DatasetDict):
            return dataset_or_dict[dataset_split]
        return dataset_or_dict

    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        return load_dataset("json", data_files=str(path), split="train")
    if suffix == ".csv":
        return load_dataset("csv", data_files=str(path), split="train")
    if suffix == ".parquet":
        return load_dataset("parquet", data_files=str(path), split="train")

    raise ValueError(
        f"Unsupported dataset path: {dataset_path}. Expected a load_from_disk directory or one of .json, .jsonl, .csv, .parquet."
    )


def coerce_json_if_needed(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    if stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def normalize_content(content: Any) -> list[dict[str, Any]]:
    content = coerce_json_if_needed(content)
    if isinstance(content, str):
        if not content.strip():
            raise ValueError("message content is empty")
        return [{"type": "text", "text": content}]
    if isinstance(content, dict):
        if "text" in content:
            return [{"type": "text", "text": content["text"]}]
        if content.get("type") == "text" and "text" in content:
            return [content]
        raise ValueError(f"unsupported content dict: {content}")
    if isinstance(content, list):
        normalized_items: list[dict[str, Any]] = []
        for item in content:
            item = coerce_json_if_needed(item)
            if isinstance(item, str):
                normalized_items.append({"type": "text", "text": item})
                continue
            if not isinstance(item, dict):
                raise ValueError(f"unsupported content item: {item!r}")
            if item.get("type") == "text" and "text" in item:
                normalized_items.append(item)
                continue
            if "text" in item:
                normalized_items.append({"type": "text", "text": item["text"]})
                continue
            raise ValueError(f"unsupported content item dict: {item}")
        if not normalized_items:
            raise ValueError("message content list is empty")
        return normalized_items
    raise ValueError(f"unsupported content type: {type(content).__name__}")


def normalize_messages(value: Any) -> list[dict[str, Any]]:
    value = coerce_json_if_needed(value)
    if not isinstance(value, list) or not value:
        raise ValueError("messages must be a non-empty list")

    normalized_messages: list[dict[str, Any]] = []
    for index, message in enumerate(value):
        message = coerce_json_if_needed(message)
        if not isinstance(message, dict):
            raise ValueError(f"message {index} is not a dict")
        role = message.get("role")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"message {index} has unsupported role: {role!r}")
        normalized_messages.append(
            {
                "role": role,
                "content": normalize_content(message.get("content")),
            }
        )
    return normalized_messages


def build_messages(example: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    raw_messages = example.get(args.messages_column)
    raw_messages = coerce_json_if_needed(raw_messages)
    if raw_messages not in (None, ""):
        return {"messages": normalize_messages(raw_messages)}

    prompt = example.get(args.prompt_column)
    completion = example.get(args.completion_column)
    system_text = example.get(args.system_column) if args.system_column else None

    if prompt in (None, "") or completion in (None, ""):
        raise ValueError(
            f"example must contain either `{args.messages_column}` or both `{args.prompt_column}` and `{args.completion_column}`"
        )

    prompt_content = normalize_content(prompt)
    completion_content = normalize_content(completion)
    messages: list[dict[str, Any]] = []
    if system_text not in (None, ""):
        messages.append({"role": "system", "content": normalize_content(system_text)})
    messages.append({"role": "user", "content": prompt_content})
    messages.append({"role": "assistant", "content": completion_content})
    return {"messages": messages}


def prepare_dataset(dataset: Dataset, args: argparse.Namespace) -> Dataset:
    if args.limit_examples > 0:
        dataset = dataset.select(range(min(args.limit_examples, len(dataset))))

    prepared_dataset = dataset.map(
        lambda row: build_messages(row, args),
        remove_columns=dataset.column_names,
        desc="Normalizing dataset into messages format",
    )

    if len(prepared_dataset) == 0:
        raise ValueError("dataset is empty after preparation")

    preview = prepared_dataset[0]["messages"]
    print("DATASET_ROWS", len(prepared_dataset), flush=True)
    print("DATASET_PREVIEW", json.dumps(preview, ensure_ascii=False), flush=True)
    return prepared_dataset


def parse_chat_template_kwargs(value: str) -> dict[str, Any]:
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("--model-chat-template-kwargs-json must be a JSON object")
    return parsed


def prepare_language_dataset(dataset: Dataset, processor: Any, chat_template_kwargs: dict[str, Any]) -> Dataset:
    def render_chat(row: dict[str, Any]) -> dict[str, str]:
        text = call_with_supported_kwargs(
            processor.apply_chat_template,
            row["messages"],
            tokenize=False,
            add_generation_prompt=False,
            **chat_template_kwargs,
        )
        return {"text": text}

    return dataset.map(
        render_chat,
        remove_columns=dataset.column_names,
        desc="Rendering messages with chat template",
    )


def build_training_args(args: argparse.Namespace) -> SFTConfig:
    save_strategy = "steps" if args.save_steps > 0 else "no"
    precision = resolve_training_precision(args)
    is_vision = args.model_loader_type == "fast_vision" or (
        args.model_loader_type == "hf_vision" and args.finetune_scope != "language_only"
    )

    config_kwargs: dict[str, Any] = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "warmup_steps": args.warmup_steps,
        "learning_rate": args.learning_rate,
        "optim": "adamw_8bit",
        "fp16": precision == "float16",
        "bf16": precision == "bfloat16",
        "logging_steps": args.logging_steps,
        "save_strategy": save_strategy,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "report_to": "none",
        "remove_unused_columns": not is_vision,
        "dataset_text_field": "" if is_vision else "text",
        "max_length": args.max_seq_length,
        "seed": args.seed,
    }
    if is_vision:
        config_kwargs["dataset_kwargs"] = {"skip_prepare_dataset": True}

    return SFTConfig(**config_kwargs)


def resolve_training_precision(args: argparse.Namespace) -> str:
    if args.bf16:
        return "bfloat16"
    if args.fp16:
        return "float16"
    if args.precision == "auto":
        return "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
    return args.precision


def resolve_model_4bit_compute_dtype(args: argparse.Namespace) -> str:
    if args.model_4bit_compute_dtype != "auto":
        return args.model_4bit_compute_dtype
    if not torch.cuda.is_available():
        return "float32"
    return "bfloat16" if torch.cuda.is_bf16_supported() else "float16"


def apply_peft_model(loader: Any, model: Any, args: argparse.Namespace) -> Any:
    if args.model_loader_type.startswith("hf_"):
        from peft import LoraConfig, TaskType, get_peft_model

        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "in_proj_qkv",
            "in_proj_z",
            "in_proj_b",
            "in_proj_a",
            "out_proj",
        ]
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
        )
        peft_model = get_peft_model(model, config)
        if args.model_loader_type == "hf_vision" and args.finetune_scope == "language_only":
            for name, parameter in peft_model.named_parameters():
                if any(part in name.lower() for part in ("vision", "visual", "image", "video")):
                    parameter.requires_grad = False
        return peft_model

    peft_kwargs: dict[str, Any] = {
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "bias": "none",
        "use_gradient_checkpointing": "unsloth",
        "random_state": args.seed,
        "max_seq_length": args.max_seq_length,
    }
    if args.model_loader_type == "fast_vision":
        peft_kwargs.update(
            {
                "finetune_vision_layers": args.finetune_scope == "all",
                "finetune_language_layers": True,
                "finetune_attention_modules": True,
                "finetune_mlp_modules": True,
            }
        )
    else:
        peft_kwargs["target_modules"] = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    return call_with_supported_kwargs(loader.get_peft_model, model, **peft_kwargs)


def main() -> None:
    args = parse_args()

    dataset = load_training_dataset(args.dataset_path, args.dataset_split)
    dataset = prepare_dataset(dataset, args)

    print("MODEL_DIR", args.model_dir, flush=True)
    print("MODEL_LOADER_TYPE", args.model_loader_type, flush=True)
    chat_template_kwargs = parse_chat_template_kwargs(args.model_chat_template_kwargs_json)
    if chat_template_kwargs:
        print("MODEL_CHAT_TEMPLATE_KWARGS", json.dumps(chat_template_kwargs, ensure_ascii=False), flush=True)
    print("TRAIN_PRECISION", resolve_training_precision(args), flush=True)
    model_4bit_compute_dtype = resolve_model_4bit_compute_dtype(args)
    print(
        "MODEL_QUANTIZATION "
        f"load_in_4bit={args.model_load_in_4bit} "
        f"quant_type={args.model_4bit_quant_type} "
        f"compute_dtype={model_4bit_compute_dtype}",
        flush=True,
    )
    print("TRAIN_FINETUNE_SCOPE", args.finetune_scope, flush=True)
    print("OUTPUT_DIR", args.output_dir, flush=True)
    print("CUDA_DEVICES", torch.cuda.device_count(), flush=True)

    loader, model, processor = load_unsloth_model(
        model_loader_type=args.model_loader_type,
        model_name_or_path=args.model_dir,
        max_seq_length=args.max_seq_length,
        model_load_in_4bit=args.model_load_in_4bit,
        model_4bit_quant_type=args.model_4bit_quant_type,
        model_4bit_compute_dtype=model_4bit_compute_dtype,
        local_files_only=args.local_files_only,
    )
    print("BASE_MODEL_OK", type(model).__name__, flush=True)

    if args.base_result_dir:
        adapter_dir = Path(args.base_result_dir) / "adapter"
        if not (adapter_dir / "adapter_model.safetensors").exists():
            raise ValueError(f"Missing adapter_model.safetensors under {adapter_dir}")
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=True)
    else:
        model = apply_peft_model(loader, model, args)
    prepare_for_training(loader, model)
    print("PEFT_READY", flush=True)

    trainer_dataset = dataset
    data_collator = None
    if args.model_loader_type == "fast_vision" or (
        args.model_loader_type == "hf_vision" and args.finetune_scope != "language_only"
    ):
        data_collator = UnslothVisionDataCollator(model, processor, max_seq_length=args.max_seq_length)
    else:
        trainer_dataset = prepare_language_dataset(dataset, processor, chat_template_kwargs)

    trainer = PatchedSFTTrainer(
        model=model,
        train_dataset=trainer_dataset,
        data_collator=data_collator,
        processing_class=processor,
        args=build_training_args(args),
    )
    print("TRAINER_READY", flush=True)

    resume_from_checkpoint = args.resume_from_checkpoint or None
    result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    print("TRAIN_RESULT", result.metrics, flush=True)

    if args.save_adapter:
        adapter_dir = str(Path(args.output_dir) / "adapter")
        processor_dir = str(Path(args.output_dir) / "processor")
        model.save_pretrained(adapter_dir)
        processor.save_pretrained(processor_dir)
        print("ADAPTER_SAVED", adapter_dir, flush=True)
        print("PROCESSOR_SAVED", processor_dir, flush=True)

    if torch.cuda.is_available():
        peak_allocations = [torch.cuda.max_memory_allocated(index) for index in range(torch.cuda.device_count())]
        print("CUDA_MAX_MEM", peak_allocations, flush=True)


if __name__ == "__main__":
    main()
