from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from runtime.config import ConfigError, ServingSettings, TrainingSettings, parse_bool
from runtime.messages import normalize_openai_messages
from runtime.storage import parse_s3_uri, prefix_with_relative_path


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

    def test_serving_settings_require_artifact_uri(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ConfigError):
                ServingSettings.from_env()


if __name__ == "__main__":
    unittest.main()
