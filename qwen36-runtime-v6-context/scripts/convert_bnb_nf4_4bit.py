#!/usr/bin/env python3

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, BitsAndBytesConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a Transformers checkpoint to a saved BitsAndBytes 4-bit checkpoint.")
    parser.add_argument("--input-dir", required=True, help="Source Transformers model directory.")
    parser.add_argument("--output-dir", required=True, help="Destination directory for the 4-bit checkpoint.")
    parser.add_argument("--model-class", default="image-text-to-text", choices=["image-text-to-text", "vision2seq", "causal-lm"])
    parser.add_argument("--quant-type", default="nf4", choices=["nf4", "fp4"])
    parser.add_argument("--compute-dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--double-quant", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-memory", default="", help='JSON map such as {"0":"14GiB","1":"14GiB","cpu":"80GiB"}.')
    parser.add_argument("--max-shard-size", default="4GB")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def get_model_class(name: str) -> Any:
    if name == "image-text-to-text":
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText
    if name == "vision2seq":
        from transformers import AutoModelForVision2Seq

        return AutoModelForVision2Seq
    if name == "causal-lm":
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM
    raise ValueError(f"Unsupported model class: {name}")


def parse_max_memory(value: str) -> dict[Any, str] | None:
    if not value:
        return None
    parsed = json.loads(value)
    return {int(key) if str(key).isdigit() else key: memory for key, memory in parsed.items()}


def copy_sidecar_files(input_dir: Path, output_dir: Path) -> None:
    for pattern in ("*.jinja", "*.md", "LICENSE", "configuration.json", "*preprocessor_config.json"):
        for source in input_dir.glob(pattern):
            target = output_dir / source.name
            if source.is_file() and not target.exists():
                shutil.copy2(source, target)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} already exists; pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    max_memory = parse_max_memory(args.max_memory)
    compute_dtype = dtype_from_name(args.compute_dtype)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=args.quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=args.double_quant,
    )

    model_cls = get_model_class(args.model_class)
    print(
        json.dumps(
            {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "model_class": model_cls.__name__,
                "quant_type": args.quant_type,
                "compute_dtype": args.compute_dtype,
                "double_quant": args.double_quant,
                "device_map": args.device_map,
                "max_memory": max_memory,
            },
            ensure_ascii=True,
        ),
        flush=True,
    )

    model = model_cls.from_pretrained(
        str(input_dir),
        quantization_config=quantization_config,
        device_map=args.device_map,
        max_memory=max_memory,
        torch_dtype=compute_dtype,
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    processor = AutoProcessor.from_pretrained(
        str(input_dir),
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )

    print("loaded; saving model", flush=True)
    model.save_pretrained(str(output_dir), safe_serialization=True, max_shard_size=args.max_shard_size)
    processor.save_pretrained(str(output_dir))
    copy_sidecar_files(input_dir, output_dir)

    config_path = output_dir / "config.json"
    with config_path.open() as handle:
        config = json.load(handle)
    print(
        json.dumps(
            {
                "saved": str(output_dir),
                "quantization_config": config.get("quantization_config"),
                "files": len(list(output_dir.iterdir())),
            },
            ensure_ascii=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
