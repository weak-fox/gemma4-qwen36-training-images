from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from runtime.config import ConfigError, ServingSettings, TrainingSettings, parse_bool
from runtime.bnb_quant_state import bnb_weight_key_candidates
from runtime.messages import normalize_openai_messages
from runtime.model_loading import build_unsloth_from_pretrained_kwargs
from runtime.storage import parse_s3_uri, prefix_with_relative_path
from runtime.train_mode import _build_train_command


class ParseBoolTests(unittest.TestCase):
    def test_parse_bool_accepts_true_variants(self) -> None:
        self.assertTrue(parse_bool("true"))
        self.assertTrue(parse_bool("Yes"))
        self.assertTrue(parse_bool("1"))

    def test_parse_bool_accepts_false_variants(self) -> None:
        self.assertFalse(parse_bool("false"))
        self.assertFalse(parse_bool("No"))
        self.assertFalse(parse_bool("0"))

    def test_parse_bool_rejects_invalid_value(self) -> None:
        with self.assertRaises(ConfigError):
            parse_bool("maybe")


class S3UriTests(unittest.TestCase):
    def test_parse_s3_uri_supports_s3_and_minio(self) -> None:
        parsed = parse_s3_uri("s3://bucket/path/to/file.jsonl")
        self.assertEqual(parsed.bucket, "bucket")
        self.assertEqual(parsed.key, "path/to/file.jsonl")

        parsed = parse_s3_uri("minio://bucket/path/to/dir")
        self.assertEqual(parsed.bucket, "bucket")
        self.assertEqual(parsed.key, "path/to/dir")

    def test_parse_s3_uri_rejects_missing_key(self) -> None:
        with self.assertRaises(ConfigError):
            parse_s3_uri("s3://bucket")

    def test_prefix_with_relative_path(self) -> None:
        self.assertEqual(prefix_with_relative_path("root/prefix", "adapter/file.bin"), "root/prefix/adapter/file.bin")
        self.assertEqual(prefix_with_relative_path("", "adapter/file.bin"), "adapter/file.bin")


class MessageNormalizationTests(unittest.TestCase):
    def test_normalize_openai_messages_accepts_string_and_list_content(self) -> None:
        normalized = normalize_openai_messages(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            ]
        )
        self.assertEqual(normalized[0]["content"][0]["text"], "You are helpful.")
        self.assertEqual(normalized[1]["content"][0]["text"], "Hello")

    def test_normalize_openai_messages_rejects_non_text_parts(self) -> None:
        with self.assertRaises(ConfigError):
            normalize_openai_messages(
                [{"role": "user", "content": [{"type": "image_url", "image_url": "ignored"}]}]
            )


class SettingsTests(unittest.TestCase):
    def test_training_settings_require_dataset_input(self) -> None:
        with patch.dict(os.environ, {"TRAIN_OUTPUT_URI": "s3://bucket/runs/run-1"}, clear=True):
            with self.assertRaises(ConfigError):
                TrainingSettings.from_env()

    def test_training_settings_prefers_default_workspace(self) -> None:
        env = {
            "TRAIN_DATASET_PATH": "/tmp/train.jsonl",
            "TRAIN_OUTPUT_URI": "s3://bucket/runs/run-1",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = TrainingSettings.from_env()
        self.assertEqual(str(settings.output_dir), "/workspace/output/run")
        self.assertEqual(settings.dataset_path.as_posix(), "/tmp/train.jsonl")

    def test_training_settings_parses_model_loader_and_quantization_contract(self) -> None:
        env = {
            "TRAIN_DATASET_PATH": "/tmp/train.jsonl",
            "TRAIN_OUTPUT_URI": "s3://bucket/runs/run-1",
            "MODEL_LOADER_TYPE": "fast_language",
            "MODEL_LOAD_IN_4BIT": "false",
            "MODEL_4BIT_QUANT_TYPE": "fp4",
            "MODEL_4BIT_COMPUTE_DTYPE": "auto",
            "MODEL_CHAT_TEMPLATE_KWARGS_JSON": '{"enable_thinking": false}',
            "TRAIN_PREPARE_KBIT": "false",
            "TRAIN_FINETUNE_SCOPE": "all",
            "TRAIN_PRECISION": "auto",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = TrainingSettings.from_env()

        self.assertEqual(settings.model_loader_type, "fast_language")
        self.assertFalse(settings.model_load_in_4bit)
        self.assertEqual(settings.model_4bit_quant_type, "fp4")
        self.assertEqual(settings.model_4bit_compute_dtype, "auto")
        self.assertEqual(settings.model_chat_template_kwargs, {"enable_thinking": False})
        self.assertFalse(settings.prepare_kbit_training)
        self.assertEqual(settings.finetune_scope, "all")
        self.assertEqual(settings.train_precision, "auto")

    def test_training_settings_accepts_hf_loader_types(self) -> None:
        env = {
            "TRAIN_DATASET_PATH": "/tmp/train.jsonl",
            "TRAIN_OUTPUT_URI": "s3://bucket/runs/run-1",
            "MODEL_LOADER_TYPE": "hf_vision",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = TrainingSettings.from_env()

        self.assertEqual(settings.model_loader_type, "hf_vision")

    def test_settings_reject_non_object_chat_template_kwargs(self) -> None:
        env = {
            "TRAIN_DATASET_PATH": "/tmp/train.jsonl",
            "TRAIN_OUTPUT_URI": "s3://bucket/runs/run-1",
            "MODEL_CHAT_TEMPLATE_KWARGS_JSON": "[]",
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ConfigError):
                TrainingSettings.from_env()

    def test_training_command_propagates_model_loader_and_quantization_flags(self) -> None:
        env = {
            "TRAIN_DATASET_PATH": "/tmp/train.jsonl",
            "TRAIN_OUTPUT_URI": "s3://bucket/runs/run-1",
            "MODEL_LOADER_TYPE": "fast_vision",
            "MODEL_LOAD_IN_4BIT": "true",
            "MODEL_4BIT_QUANT_TYPE": "nf4",
            "MODEL_4BIT_COMPUTE_DTYPE": "auto",
            "TRAIN_FINETUNE_SCOPE": "language_only",
            "MODEL_CHAT_TEMPLATE_KWARGS_JSON": '{"enable_thinking": false}',
        }
        with patch.dict(os.environ, env, clear=True):
            settings = TrainingSettings.from_env()

        command = _build_train_command(settings, Path("/tmp/train.jsonl"))
        self.assertIn("--model-loader-type", command)
        self.assertIn("fast_vision", command)
        self.assertIn("--model-load-in-4bit", command)
        self.assertIn("--model-4bit-quant-type", command)
        self.assertIn("nf4", command)
        self.assertIn("--model-4bit-compute-dtype", command)
        self.assertIn("auto", command)
        self.assertIn("--model-chat-template-kwargs-json", command)
        self.assertIn('{"enable_thinking": false}', command)
        self.assertIn("--finetune-scope", command)
        self.assertIn("language_only", command)
        self.assertIn("--precision", command)
        self.assertIn("auto", command)
        self.assertIn("--prepare-kbit-training", command)

    def test_serving_settings_support_optional_artifact_and_model_loading_flags(self) -> None:
        env = {
            "MODEL_LOADER_TYPE": "fast_language",
            "MODEL_LOCAL_FILES_ONLY": "false",
            "MODEL_LOAD_IN_4BIT": "true",
            "MODEL_4BIT_QUANT_TYPE": "nf4",
            "MODEL_4BIT_COMPUTE_DTYPE": "bfloat16",
            "MODEL_CHAT_TEMPLATE_KWARGS_JSON": '{"enable_thinking": false}',
        }
        with patch.dict(os.environ, {}, clear=True):
            default_settings = ServingSettings.from_env()
        self.assertIsNone(default_settings.artifact_uri)

        with patch.dict(os.environ, env, clear=True):
            settings = ServingSettings.from_env()
        self.assertEqual(settings.model_loader_type, "fast_language")
        self.assertFalse(settings.model_local_files_only)
        self.assertTrue(settings.model_load_in_4bit)
        self.assertEqual(settings.model_4bit_quant_type, "nf4")
        self.assertEqual(settings.model_4bit_compute_dtype, "bfloat16")
        self.assertEqual(settings.model_chat_template_kwargs, {"enable_thinking": False})


class ModelLoadingTests(unittest.TestCase):
    def test_bnb_weight_key_candidates_support_vlm_checkpoint_loaded_as_language_model(self) -> None:
        candidates = bnb_weight_key_candidates("model.layers.0.linear_attn.in_proj_qkv")

        self.assertEqual(candidates[0], "model.layers.0.linear_attn.in_proj_qkv.weight")
        self.assertIn("model.language_model.layers.0.linear_attn.in_proj_qkv.weight", candidates)

    def test_unsloth_kwargs_use_quantization_config_without_legacy_bnb_kwargs(self) -> None:
        quantization_config = object()
        with patch("runtime.model_loading.build_bnb_4bit_config", return_value=quantization_config):
            kwargs = build_unsloth_from_pretrained_kwargs(
                model_name_or_path="/models/qwen",
                max_seq_length=128,
                model_load_in_4bit=True,
                model_4bit_quant_type="nf4",
                model_4bit_compute_dtype="float16",
                local_files_only=True,
            )

        self.assertTrue(kwargs["load_in_4bit"])
        self.assertIs(kwargs["quantization_config"], quantization_config)
        self.assertNotIn("bnb_4bit_quant_type", kwargs)
        self.assertNotIn("bnb_4bit_compute_dtype", kwargs)
        self.assertNotIn("bnb_4bit_use_double_quant", kwargs)


class TrainingScriptTests(unittest.TestCase):
    def test_training_script_accepts_auto_model_compute_dtype(self) -> None:
        script = Path(__file__).resolve().parents[1] / "scripts" / "train_gemma4_31b_qlora.py"
        source = script.read_text(encoding="utf-8")

        self.assertIn('choices=["auto", "float16", "bfloat16", "float32"]', source)
        self.assertIn('"hf_vision"', source)
        self.assertIn("def resolve_model_4bit_compute_dtype", source)


if __name__ == "__main__":
    unittest.main()
