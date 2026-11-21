"""Character, face and speaker↔character data contracts."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import Field

from .common import Contract


class BoundingBox(Contract):
    x: float
    y: float
    width: float
    height: float


class FaceDetection(Contract):
    face_id: str
    shot_id: str
    time_s: float | None = None
    bbox: BoundingBox
    confidence: float
    crop_uri: str | None = None


class RepresentativeFace(Contract):
    face_id: str
    shot_id: str
    uri: str | None = None
    centroid_similarity: float | None = None


Importance = Literal["main", "supporting", "recurring_secondary", "minor", "background", "unclassified"]


class SourceCharacter(Contract):
    """A character identity observed in the source video (face cluster level)."""

    character_id: str
    kind: Literal["source"] = "source"
    legacy_ids: list[str] = Field(default_factory=list)
    status: Literal["provisional", "human_confirmed", "merged", "rejected"] = "provisional"
    face_count: int = 0
    shot_ids: list[str] = Field(default_factory=list)
    first_shot_id: str | None = None
    last_shot_id: str | None = None
    mean_similarity: float | None = None
    p10_similarity: float | None = None
    representative_faces: list[RepresentativeFace] = Field(default_factory=list)
    speaker_ids: list[str] = Field(default_factory=list)
    importance: Importance = "unclassified"
    importance_score: float | None = None


class Evidence(Contract):
    kind: Literal["asd", "cooccurrence", "voice_cluster", "human"]
    weight: float | None = None
    detail: dict[str, float | int | str | None] = Field(default_factory=dict)


class SpeakerCharacterMapping(Contract):
    speaker_id: str
    character_id: str | None = Field(description="None = could not be reliably matched")
    status: Literal["accepted", "uncertain", "rejected", "human_confirmed"]
    confidence: float | None = None
    rank: int | None = None
    supporting_shots: int = 0
    supporting_segments: int = 0
    supporting_speech_s: float = 0.0
    evidence: list[Evidence] = Field(default_factory=list)
    contradictions: list[Evidence] = Field(default_factory=list)
    uncertainty_reason: str | None = None


_CHAR_RE = re.compile(r"(\d+)$")


def source_character_id(legacy_label: str) -> str:
    """'CHAR_0003' → 'SRC_CHAR_0003'."""
    match = _CHAR_RE.search(legacy_label)
    if not match:
        raise ValueError(f"cannot derive a character id from '{legacy_label}'")
    return f"SRC_CHAR_{int(match.group(1)):04d}"
