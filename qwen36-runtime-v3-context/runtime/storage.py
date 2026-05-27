from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

from runtime.config import ConfigError, MinioSettings


@dataclass(frozen=True)
class S3Uri:
    bucket: str
    key: str


def parse_s3_uri(uri: str) -> S3Uri:
    parsed = urlparse(uri)
    if parsed.scheme not in {"s3", "minio"}:
        raise ConfigError(f"Unsupported URI scheme in {uri!r}; expected s3://")
    bucket = parsed.netloc.strip()
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        raise ConfigError(f"URI must include both bucket and key: {uri!r}")
    return S3Uri(bucket=bucket, key=key)


def normalize_prefix(key: str) -> str:
    return key.strip("/")


def prefix_with_relative_path(prefix: str, relative_path: str) -> str:
    normalized_prefix = normalize_prefix(prefix)
    normalized_relative_path = relative_path.strip("/")
    if not normalized_prefix:
        return normalized_relative_path
    if not normalized_relative_path:
        return normalized_prefix
    return f"{normalized_prefix}/{normalized_relative_path}"


class MinioStorage:
    def __init__(self, settings: MinioSettings):
        from minio import Minio

        self._client = Minio(
            settings.endpoint,
            access_key=settings.access_key,
            secret_key=settings.secret_key,
            secure=settings.secure,
            region=settings.region,
        )

    def download_object(self, uri: str, destination: Path) -> Path:
        s3_uri = parse_s3_uri(uri)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._client.fget_object(s3_uri.bucket, s3_uri.key, str(destination))
        return destination

    def download_prefix(self, uri: str, destination_dir: Path) -> Path:
        s3_uri = parse_s3_uri(uri)
        prefix = normalize_prefix(s3_uri.key)
        list_prefix = f"{prefix}/" if prefix else ""
        objects = list(self._client.list_objects(s3_uri.bucket, prefix=list_prefix, recursive=True))
        file_objects = [obj for obj in objects if not obj.object_name.endswith("/")]
        if not file_objects:
            raise ConfigError(f"No objects found under artifact prefix: {uri}")

        destination_dir.mkdir(parents=True, exist_ok=True)
        for obj in file_objects:
            relative_name = obj.object_name[len(list_prefix):].lstrip("/")
            if not relative_name:
                continue
            local_path = destination_dir / relative_name
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self._client.fget_object(s3_uri.bucket, obj.object_name, str(local_path))
        return destination_dir

    def upload_directory(self, source_dir: Path, destination_uri: str) -> None:
        if not source_dir.exists():
            raise ConfigError(f"Training output directory does not exist: {source_dir}")

        s3_uri = parse_s3_uri(destination_uri)
        files = list(iter_files(source_dir))
        if not files:
            raise ConfigError(f"Training output directory is empty: {source_dir}")

        for file_path in files:
            relative_path = file_path.relative_to(source_dir).as_posix()
            object_name = prefix_with_relative_path(s3_uri.key, relative_path)
            self._client.fput_object(s3_uri.bucket, object_name, str(file_path))


def iter_files(root: Path) -> Iterable[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())

