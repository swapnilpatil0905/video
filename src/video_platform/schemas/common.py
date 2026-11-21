"""Shared data-contract primitives (schema_version 1).

Conventions:
* Timestamps are float seconds on the **source** timeline.
* IDs are stable strings: shots ``SH_00001``, segments ``SEG_000001``, speakers
  ``SPK_00``, source characters ``SRC_CHAR_0001``, replacement characters
  ``GEN_CHAR_0001``, voices ``VOICE_0001``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ArtifactRef(Contract):
    artifact_id: str
    uri: str | None = None
    sha256: str | None = None
    media_type: str | None = None
    role: str | None = Field(default=None, description="e.g. 'normalized', 'legacy_raw', 'input'")


class Metric(Contract):
    value: float | int | str | bool | None
    unit: str | None = None
    kind: Literal["objective", "subjective"] = "objective"
    threshold: float | None = None
    passed: bool | None = None
    note: str | None = None


class Labeled(Contract, Generic[T]):
    """A model- or rule-produced label with its confidence and provenance."""

    value: T
    confidence: float | None = Field(default=None, ge=0, le=1)
    source: str | None = Field(default=None, description="model id, rule name, or 'human'")


ArtifactStatus = Literal["valid", "quarantined", "superseded", "failed"]


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


class ArtifactRecord(Contract):
    """Provenance record written for every generated (or imported) artifact."""

    schema_version: Literal[1] = 1
    project_id: str
    stage: str
    stage_version: str
    artifact_id: str
    work_item: str = "project"
    model_id: str | None = None
    model_fingerprint: str = ""
    inputs: list[ArtifactRef] = Field(default_factory=list)
    outputs: list[ArtifactRef] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    params_hash: str
    created_at: datetime = Field(default_factory=utc_now)
    processing_seconds: float | None = None
    gpu_type: str | None = None
    metrics: dict[str, Metric] = Field(default_factory=dict)
    status: ArtifactStatus = "valid"
    rights_basis: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict, description="e.g. legacy import details")
