"""Audio-side data contracts."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import Field

from .common import Contract, Labeled

MatchStrength = Literal["strong", "moderate", "weak", "none"]


class TranscriptSegment(Contract):
    segment_id: str
    start_s: float
    end_s: float
    text: str
    language: str | None = None
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None
    flags: list[str] = Field(default_factory=list, description="e.g. invalid_timestamp, filtered_watermark")


class SpeakerTurn(Contract):
    start_s: float
    end_s: float
    speaker_id: str


class AudioSegment(Contract):
    """Aligned speech segment (transcript × diarization × shots)."""

    segment_id: str
    start_s: float
    end_s: float
    text: str
    language: str | None = None
    speaker_id: str | None = None
    speaker_overlap_ratio: float | None = None
    speaker_match: MatchStrength = "none"
    overlapping_speakers: list[str] = Field(default_factory=list)
    shot_id: str | None = None
    shot_match: MatchStrength = "none"
    overlapping_shot_ids: list[str] = Field(default_factory=list)
    emotion: Labeled[str] | None = None
    delivery: list[Labeled[str]] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


def segment_id(number: int) -> str:
    return f"SEG_{number:06d}"


_SPEAKER_RE = re.compile(r"(\d+)$")


def speaker_id(label: str) -> str:
    """Normalise diarization labels: 'SPEAKER_02' → 'SPK_02'."""
    match = _SPEAKER_RE.search(label)
    if not match:
        return f"SPK_{label}"
    return f"SPK_{int(match.group(1)):02d}"
