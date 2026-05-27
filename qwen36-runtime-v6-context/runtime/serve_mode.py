from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path
from typing import Any

from runtime.config import ConfigError, MinioSettings, ServingSettings
from runtime.messages import normalize_openai_messages
from runtime.model_loading import call_with_supported_kwargs, load_unsloth_model, prepare_for_inference
from runtime.storage import MinioStorage

from pydantic import BaseModel, Field


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[dict[str, Any]]
    stream: bool = False
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)

    class Config:
        extra = "allow"


class ChatService:
    def __init__(self, settings: ServingSettings):
        self._settings = settings
        self._lock = threading.Lock()
        self._ready = False
        self._model = None
        self._processor = None
        self._input_device = "cpu"

    @property
    def ready(self) -> bool:
        return self._ready

    def load(self) -> None:
        import torch
        from peft import PeftModel

        loader, model, base_processor = load_unsloth_model(
            model_loader_type=self._settings.model_loader_type,
            model_name_or_path=self._settings.model_name_or_path,
            max_seq_length=2048,
            model_load_in_4bit=self._settings.model_load_in_4bit,
            model_4bit_quant_type=self._settings.model_4bit_quant_type,
            model_4bit_compute_dtype=self._settings.model_4bit_compute_dtype,
            local_files_only=self._settings.model_local_files_only,
        )
        processor = base_processor
        artifact_root = None
        if self._settings.artifact_uri:
            storage = MinioStorage(MinioSettings.from_env())
            artifact_root = storage.download_prefix(self._settings.artifact_uri, self._settings.artifact_root)
        elif (self._settings.artifact_root / "adapter" / "adapter_model.safetensors").exists():
            artifact_root = self._settings.artifact_root

        if artifact_root is not None:
            adapter_dir = artifact_root / "adapter"
            if not (adapter_dir / "adapter_model.safetensors").exists():
                raise ConfigError(
                    f"Missing adapter weights under {adapter_dir}. "
                    "SERVE_ARTIFACT_URI must point to a full training run prefix that contains adapter/."
                )
            print(f"SERVE_ARTIFACT_READY {artifact_root}", flush=True)
            model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
            processor_dir = artifact_root / "processor"
            if processor_dir.exists():
                processor = _load_saved_processor(
                    processor_dir,
                    local_files_only=self._settings.model_local_files_only,
                )
        prepare_for_inference(loader, model)

        self._model = model
        self._processor = processor
        self._input_device = _infer_input_device(model, torch)
        self._ready = True
        print(
            "SERVE_MODEL_READY "
            f"loader={self._settings.model_loader_type} "
            f"load_in_4bit={self._settings.model_load_in_4bit} "
            f"input_device={self._input_device}",
            flush=True,
        )

    def generate(
        self,
        messages: list[dict[str, Any]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> dict[str, Any]:
        if not self._ready or self._model is None or self._processor is None:
            raise ConfigError("Model service is not ready")

        import torch

        normalized_messages = normalize_openai_messages(messages)
        prompt = call_with_supported_kwargs(
            self._processor.apply_chat_template,
            normalized_messages,
            add_generation_prompt=True,
            tokenize=False,
            **self._settings.model_chat_template_kwargs,
        )
        model_inputs = self._processor(text=prompt, return_tensors="pt")
        model_inputs = {
            key: value.to(self._input_device) if hasattr(value, "to") else value
            for key, value in model_inputs.items()
        }

        sampling = temperature > 0
        generate_kwargs: dict[str, Any] = {
            **model_inputs,
            "max_new_tokens": max_new_tokens,
            "top_p": top_p,
            "do_sample": sampling,
            "use_cache": True,
        }
        if sampling:
            generate_kwargs["temperature"] = temperature

        with self._lock, torch.inference_mode():
            output_ids = self._model.generate(**generate_kwargs)

        input_length = int(model_inputs["input_ids"].shape[-1])
        completion_ids = output_ids[0][input_length:]
        completion_text = self._processor.decode(completion_ids, skip_special_tokens=True).strip()

        return {
            "text": completion_text,
            "prompt_tokens": int(model_inputs["input_ids"].shape[-1]),
            "completion_tokens": int(completion_ids.shape[-1]),
        }


def _infer_input_device(model: Any, torch_module: Any) -> str:
    device_map = getattr(model, "hf_device_map", None) or {}
    if isinstance(device_map, dict):
        for target in device_map.values():
            if isinstance(target, str) and target.startswith("cuda"):
                return target
    if torch_module.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _load_saved_processor(processor_dir: Path, local_files_only: bool) -> Any:
    from transformers import AutoProcessor, AutoTokenizer

    try:
        return AutoProcessor.from_pretrained(str(processor_dir), local_files_only=local_files_only)
    except Exception:
        return AutoTokenizer.from_pretrained(str(processor_dir), local_files_only=local_files_only)


def run_server(settings: ServingSettings) -> None:
    from fastapi import Body, FastAPI, HTTPException
    import uvicorn

    service = ChatService(settings)
    service.load()

    app = FastAPI(title="LLM-VLM QLoRA Runtime", version="1.0.0")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> dict[str, str]:
        if not service.ready:
            raise HTTPException(status_code=503, detail="model-not-ready")
        return {"status": "ready"}

    @app.post("/v1/chat/completions")
    def chat_completions(payload: ChatCompletionRequest = Body(...)) -> dict[str, Any]:
        if payload.stream:
            raise HTTPException(status_code=400, detail="stream=true is not supported")

        try:
            result = service.generate(
                messages=payload.messages,
                max_new_tokens=payload.max_tokens or settings.max_new_tokens_default,
                temperature=payload.temperature if payload.temperature is not None else settings.temperature_default,
                top_p=payload.top_p if payload.top_p is not None else settings.top_p_default,
            )
        except ConfigError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - runtime safeguard
            raise HTTPException(status_code=500, detail=f"inference failed: {exc}") from exc

        response_model = (
            payload.model
            or settings.response_model_name
            or Path(settings.model_name_or_path).name
        )
        created = int(time.time())
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": response_model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result["text"]},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
            },
        }

    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")
