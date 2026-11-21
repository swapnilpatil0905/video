"""Video-side data contracts."""

from __future__ import annotations

from pydantic import Field

from .common import Contract, Labeled


class AudioStreamInfo(Contract):
    index: int
    codec: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    duration_s: float | None = None
    language: str | None = None


class VideoMetadata(Contract):
    schema_version: int = 1
    source_sha256: str
    size_bytes: int
    container: str | None = None
    duration_s: float
    width: int
    height: int
    display_aspect: str
    fps_num: int
    fps_den: int
    video_codec: str | None = None
    pix_fmt: str | None = None
    rotation: int = 0
    bitrate: int | None = None
    video_stream_duration_s: float | None = None
    audio_streams: list[AudioStreamInfo] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def fps(self) -> float:
        return self.fps_num / self.fps_den if self.fps_den else 0.0


class Keyframe(Contract):
    shot_id: str
    time_s: float
    uri: str


class Shot(Contract):
    """A camera shot (contiguous frames between cuts). Scenes group shots (later milestone)."""

    shot_id: str
    start_s: float
    end_s: float
    scene_id: str | None = None
    legacy_scene_number: int | None = Field(default=None, description="Numbering used by the pre-platform scripts")
    shot_type: Labeled[str] | None = None
    camera_angle: Labeled[str] | None = None
    camera_motion: Labeled[str] | None = None
    indoor_outdoor: Labeled[str] | None = None
    environment_id: str | None = None
    lighting: Labeled[str] | None = None
    objects: list[Labeled[str]] = Field(default_factory=list)
    keyframes: list[Keyframe] = Field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def shot_id(number: int) -> str:
    return f"SH_{number:05d}"
