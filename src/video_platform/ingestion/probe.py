"""Media inspection with ffprobe (S0 ingestion helper)."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable
from fractions import Fraction
from math import gcd
from pathlib import Path
from typing import Any

from ..core.errors import ConfigError, StageFailure
from ..schemas.media import AudioStreamInfo, VideoMetadata

ProbeFn = Callable[[Path], dict[str, Any]]


def ffprobe_json(path: Path) -> dict[str, Any]:
    binary = shutil.which("ffprobe")
    if binary is None:
        raise ConfigError("ffprobe not found on PATH (install FFmpeg from a public source)")
    result = subprocess.run(
        [binary, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise StageFailure(f"ffprobe failed for {path}: {result.stderr.strip()[:500]}")
    return json.loads(result.stdout)


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rate(value: Any) -> Fraction | None:
    """Parse an ffprobe rate like '30000/1001'; '0/0' and garbage → None."""
    try:
        rate = Fraction(str(value))
    except (ValueError, ZeroDivisionError, TypeError):
        return None
    return rate if rate > 0 else None


def _rotation(stream: dict[str, Any]) -> int:
    for item in stream.get("side_data_list", []) or []:
        if "rotation" in item:
            return int(float(item["rotation"])) % 360
    rotate = (stream.get("tags") or {}).get("rotate")
    return int(rotate) % 360 if rotate else 0


def metadata_from_probe(probe: dict[str, Any], *, sha256: str, size_bytes: int) -> VideoMetadata:
    streams = probe.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise StageFailure("no video stream found")
    fmt = probe.get("format", {})
    warnings: list[str] = []

    rate = _rate(video.get("avg_frame_rate")) or _rate(video.get("r_frame_rate"))
    if not rate:
        raise StageFailure("could not determine frame rate")
    width, height = int(video["width"]), int(video["height"])
    rotation = _rotation(video)
    display_w, display_h = (height, width) if rotation in (90, 270) else (width, height)
    divisor = gcd(display_w, display_h) or 1
    aspect = f"{display_w // divisor}:{display_h // divisor}"

    duration = _float(fmt.get("duration")) or _float(video.get("duration"))
    if duration is None:
        raise StageFailure("could not determine duration (file may be incomplete)")
    video_duration = _float(video.get("duration"))
    if video_duration is not None and abs(video_duration - duration) > 0.5:
        warnings.append(f"container duration {duration:.3f}s differs from video stream {video_duration:.3f}s")

    audio = [
        AudioStreamInfo(
            index=int(s.get("index", i)),
            codec=s.get("codec_name"),
            sample_rate=int(s["sample_rate"]) if s.get("sample_rate") else None,
            channels=s.get("channels"),
            duration_s=_float(s.get("duration")),
            language=(s.get("tags") or {}).get("language"),
        )
        for i, s in enumerate(streams)
        if s.get("codec_type") == "audio"
    ]
    if not audio:
        warnings.append("no audio stream")

    return VideoMetadata(
        source_sha256=sha256,
        size_bytes=size_bytes,
        container=fmt.get("format_name"),
        duration_s=round(duration, 6),
        width=width,
        height=height,
        display_aspect=aspect,
        fps_num=rate.numerator,
        fps_den=rate.denominator,
        video_codec=video.get("codec_name"),
        pix_fmt=video.get("pix_fmt"),
        rotation=rotation,
        bitrate=int(fmt["bit_rate"]) if fmt.get("bit_rate") else None,
        video_stream_duration_s=video_duration,
        audio_streams=audio,
        warnings=warnings,
    )


def probe_video(path: Path, *, sha256: str, probe: ProbeFn = ffprobe_json) -> VideoMetadata:
    return metadata_from_probe(probe(path), sha256=sha256, size_bytes=path.stat().st_size)
