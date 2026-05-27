from __future__ import annotations

from typing import Any

from runtime.config import ConfigError


ALLOWED_ROLES = {"system", "user", "assistant"}


def normalize_text_parts(content: Any) -> list[dict[str, str]]:
    if isinstance(content, str):
        text = content.strip()
        if not text:
            raise ConfigError("Message content must not be empty")
        return [{"type": "text", "text": text}]

    if isinstance(content, list):
        parts: list[dict[str, str]] = []
        for item in content:
            if not isinstance(item, dict):
                raise ConfigError("Message content list items must be objects")
            item_type = item.get("type", "text")
            if item_type != "text":
                raise ConfigError(f"Unsupported content item type: {item_type!r}")
            text = str(item.get("text", "")).strip()
            if not text:
                raise ConfigError("Text content items must not be empty")
            parts.append({"type": "text", "text": text})
        if not parts:
            raise ConfigError("Message content list must not be empty")
        return parts

    raise ConfigError("Message content must be a string or a list of text parts")


def normalize_openai_messages(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list) or not messages:
        raise ConfigError("messages must be a non-empty list")

    normalized_messages: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ConfigError("Each message must be an object")
        role = message.get("role")
        if role not in ALLOWED_ROLES:
            raise ConfigError(f"Unsupported message role: {role!r}")
        normalized_messages.append(
            {
                "role": role,
                "content": normalize_text_parts(message.get("content")),
            }
        )
    return normalized_messages
