"""Content hashing and deterministic artifact identifiers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path, PurePath
from typing import Any

_CHUNK = 8 * 1024 * 1024
ARTIFACT_ID_PREFIX = "art_"
ARTIFACT_ID_HEX_LEN = 40


def sha256_file(path: str | Path) -> str:
    """Stream a file through SHA-256 without loading it into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, PurePath):
        return value.as_posix()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    raise TypeError(f"Value of type {type(value).__name__} is not JSON-serialisable for hashing")


def canonical_json(value: Any) -> bytes:
    """Stable JSON encoding: sorted keys, no whitespace, UTF-8."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    ).encode("utf-8")


def params_hash(params: Any) -> str:
    return sha256_bytes(canonical_json(params if params is not None else {}))


def compute_artifact_id(
    *,
    stage: str,
    stage_version: str,
    input_ids: Iterable[str],
    params_digest: str,
    model_fingerprint: str = "",
    work_item: str = "project",
) -> str:
    """Deterministic artifact id.

    Identical (stage, stage_version, inputs, params, model, work_item) always yield the
    same id, so an existing valid artifact with that id is a cache hit. Changing any
    component yields a new id, which is how invalidation propagates downstream.
    """
    payload = {
        "stage": stage,
        "stage_version": stage_version,
        "inputs": sorted(input_ids),
        "params": params_digest,
        "model": model_fingerprint,
        "work_item": work_item,
    }
    return ARTIFACT_ID_PREFIX + sha256_bytes(canonical_json(payload))[:ARTIFACT_ID_HEX_LEN]
