"""Artifact storage backends.

Keys are POSIX-style relative paths (``<stage>/<artifact_id>/<file>``). Backends:

* ``LocalArtifactStore`` — filesystem root (Mac development, tests, worker cache).
  Supports hardlink import so legacy data is adopted without extra disk usage.
* ``S3ArtifactStore`` — MinIO or any S3-compatible endpoint (VPS / Vast.ai workers).
"""

from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..core.errors import ConfigError
from ..core.network_policy import DEFAULT_FORBIDDEN_HOST_PATTERNS, assert_allowed


def normalize_key(key: str) -> str:
    """Validate a storage key: relative, no traversal, no empty segments."""
    if not key or key.startswith("/") or "\\" in key:
        raise ValueError(f"invalid storage key '{key}'")
    parts = key.split("/")
    if any(part in ("..", ".", "") for part in parts):
        raise ValueError(f"invalid storage key '{key}'")
    return key


class ArtifactStore(ABC):
    @abstractmethod
    def uri(self, key: str) -> str: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def put_file(self, source: Path, key: str, *, link: bool = False) -> str:
        """Store a local file under ``key`` and return its URI."""

    @abstractmethod
    def put_bytes(self, data: bytes, key: str) -> str: ...

    @abstractmethod
    def read_bytes(self, key: str) -> bytes: ...

    @abstractmethod
    def fetch(self, key: str, destination: Path) -> Path:
        """Materialise ``key`` at ``destination`` (download or link)."""

    @abstractmethod
    def key_from_uri(self, uri: str) -> str: ...

    def materialize(self, key: str, cache_dir: Path) -> Path:
        """A readable local path for ``key`` (downloads into ``cache_dir`` for remote stores)."""
        destination = Path(cache_dir) / normalize_key(key)
        if not destination.exists():
            self.fetch(key, destination)
        return destination


class LocalArtifactStore(ArtifactStore):
    scheme = "local"

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def path_for(self, key: str) -> Path:
        return self.root / normalize_key(key)

    def uri(self, key: str) -> str:
        return f"{self.scheme}://{normalize_key(key)}"

    def exists(self, key: str) -> bool:
        return self.path_for(key).exists()

    def key_from_uri(self, uri: str) -> str:
        prefix = f"{self.scheme}://"
        if not uri.startswith(prefix):
            raise ValueError(f"not a {self.scheme} URI: {uri}")
        return normalize_key(uri[len(prefix):])

    def materialize(self, key: str, cache_dir: Path) -> Path:
        return self.path_for(key)

    def put_file(self, source: Path, key: str, *, link: bool = False) -> str:
        destination = self.path_for(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if link and _same_inode(source, destination):
                return self.uri(key)
            destination.unlink()
        if link:
            try:
                os.link(source, destination)
                return self.uri(key)
            except OSError:
                pass  # cross-device or unsupported: fall back to copy
        shutil.copy2(source, destination)
        return self.uri(key)

    def put_bytes(self, data: bytes, key: str) -> str:
        destination = self.path_for(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_name(destination.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, destination)
        return self.uri(key)

    def read_bytes(self, key: str) -> bytes:
        return self.path_for(key).read_bytes()

    def fetch(self, key: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = self.path_for(key)
        if destination.exists():
            destination.unlink()
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        return destination


def _same_inode(a: Path, b: Path) -> bool:
    try:
        sa, sb = a.stat(), b.stat()
    except OSError:
        return False
    return (sa.st_dev, sa.st_ino) == (sb.st_dev, sb.st_ino)


class S3ArtifactStore(ArtifactStore):
    """S3-compatible store (MinIO by default). Requires the optional ``s3`` extra (boto3)."""

    def __init__(
        self,
        *,
        endpoint: str,
        bucket: str,
        prefix: str = "",
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        forbidden_hosts: tuple[str, ...] | list[str] = DEFAULT_FORBIDDEN_HOST_PATTERNS,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        assert_allowed(endpoint, list(forbidden_hosts), what="storage endpoint")
        self.endpoint = endpoint
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        if client_factory is None:
            try:
                import boto3  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover - depends on optional extra
                raise ConfigError("S3/MinIO storage requires boto3: pip install -e '.[s3]'") from exc
            client_factory = boto3.client
        self._client = client_factory(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
        )

    def _object_key(self, key: str) -> str:
        key = normalize_key(key)
        return f"{self.prefix}/{key}" if self.prefix else key

    def uri(self, key: str) -> str:
        return f"s3://{self.bucket}/{self._object_key(key)}"

    def key_from_uri(self, uri: str) -> str:
        prefix = f"s3://{self.bucket}/" + (f"{self.prefix}/" if self.prefix else "")
        if not uri.startswith(prefix):
            raise ValueError(f"URI {uri} is outside this store ({prefix})")
        return normalize_key(uri[len(prefix):])

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=self._object_key(key))
            return True
        except Exception as exc:  # botocore ClientError; avoid importing botocore here
            code = getattr(exc, "response", {}).get("Error", {}).get("Code")
            if code in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def put_file(self, source: Path, key: str, *, link: bool = False) -> str:
        self._client.upload_file(str(source), self.bucket, self._object_key(key))
        return self.uri(key)

    def put_bytes(self, data: bytes, key: str) -> str:
        self._client.put_object(Bucket=self.bucket, Key=self._object_key(key), Body=data)
        return self.uri(key)

    def read_bytes(self, key: str) -> bytes:
        response = self._client.get_object(Bucket=self.bucket, Key=self._object_key(key))
        return response["Body"].read()

    def fetch(self, key: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._client.download_file(self.bucket, self._object_key(key), str(destination))
        return destination


def store_from_config(loaded: Any) -> ArtifactStore:
    """Build the configured store for a ``LoadedConfig``."""
    spec = loaded.config.storage
    if spec.provider == "local":
        return LocalArtifactStore(loaded.artifact_root)
    secret = spec.secret_access_key.get_secret_value() if spec.secret_access_key else None
    return S3ArtifactStore(
        endpoint=spec.endpoint,
        bucket=spec.bucket,
        prefix=f"{spec.prefix.strip('/')}/{loaded.project_id}",
        access_key_id=spec.access_key_id,
        secret_access_key=secret,
        forbidden_hosts=loaded.config.packages.forbid_hosts_matching,
    )
