"""FFmpeg / audio helpers shared by stage executors (core runtime)."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")
R = TypeVar("R")

X264_INTERMEDIATE = ["-c:v", "libx264", "-preset", "medium", "-crf", "17", "-threads", "4"]
X264_FINAL = ["-c:v", "libx264", "-preset", "slow", "-crf", "17"]
NVENC = ["-c:v", "h264_nvenc", "-preset", "p6", "-tune", "hq", "-rc", "vbr", "-cq", "18", "-b:v", "0"]

from ..core.errors import ConfigError, StageFailure


def _bin(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise ConfigError(f"{name} not found on PATH (apt-get install ffmpeg)")
    return path


def ffmpeg(args: list[str], *, capture: bool = False) -> str:
    cmd = [_bin("ffmpeg"), "-hide_banner", "-nostdin", "-y", "-loglevel", "info" if capture else "error", *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise StageFailure(f"ffmpeg failed: {' '.join(args[:6])} …\n{result.stderr[-1500:]}")
    return result.stderr


@lru_cache(maxsize=1)
def nvenc_available() -> bool:
    """True only if a real test encode with h264_nvenc succeeds (driver + container capabilities)."""
    if os.environ.get("VP_USE_NVENC", "1") == "0" or shutil.which("ffmpeg") is None:
        return False
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.2",
         *NVENC, "-f", "null", "-"], capture_output=True, text=True)
    return result.returncode == 0


def intermediate_codec() -> list[str]:
    return NVENC if nvenc_available() else X264_INTERMEDIATE


def pmap(fn: Callable[[T], R], items: list[T], workers: int) -> list[R]:
    """Run ``fn`` over items with a thread pool (for subprocess-bound work: ffmpeg, model CLIs)."""
    if workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items))


def duration(path: Path) -> float:
    result = subprocess.run(
        [_bin("ffprobe"), "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise StageFailure(f"ffprobe failed for {path}: {result.stderr[-500:]}")
    return float(json.loads(result.stdout)["format"]["duration"])


def extract_wav(src: Path, out: Path, *, stream: int = 0, rate: int = 16000, channels: int = 1, codec: str = "pcm_s16le") -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg(["-i", str(src), "-map", f"0:a:{stream}", "-vn", "-ac", str(channels), "-ar", str(rate), "-c:a", codec, str(out)])
    return out


def cut_video(src: Path, start: float, end: float, out: Path, *, fps: float | None = None) -> Path:
    """Frame-accurate re-encoded cut without audio."""
    out.parent.mkdir(parents=True, exist_ok=True)
    vf = [f"fps={fps}"] if fps else []
    ffmpeg([
        "-ss", f"{max(0.0, start):.3f}", "-i", str(src), "-t", f"{max(0.04, end - start):.3f}", "-an",
        *(["-vf", ",".join(vf)] if vf else []),
        *intermediate_codec(), "-pix_fmt", "yuv420p", str(out),
    ])
    return out


def conform_video(src: Path, out: Path, *, seconds: float, fps: float, width: int, height: int) -> Path:
    """Scale/pad to WxH, constant fps, exact duration (pads by cloning the last frame). No audio."""
    out.parent.mkdir(parents=True, exist_ok=True)
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps},"
        f"tpad=stop_mode=clone:stop_duration={seconds:.3f}"
    )
    ffmpeg([
        "-i", str(src), "-an", "-vf", vf, "-t", f"{seconds:.3f}", "-r", f"{fps}",
        *intermediate_codec(), "-pix_fmt", "yuv420p", "-video_track_timescale", "90000", str(out),
    ])
    return out


def concat_videos(videos: Iterable[Path], out: Path, *, fps: float, final_quality: bool) -> Path:
    """Join clips and re-encode once, so clips made at different times/encoders always join cleanly."""
    out.parent.mkdir(parents=True, exist_ok=True)
    listing = out.with_suffix(".txt")
    listing.write_text("".join(f"file '{Path(v).resolve()}'\n" for v in videos), encoding="utf-8")
    codec = X264_FINAL if final_quality or not nvenc_available() else NVENC
    ffmpeg(["-f", "concat", "-safe", "0", "-i", str(listing), "-an", "-r", f"{fps}", *codec, "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(out)])
    return out


def retime_audio(src: Path, out: Path, *, tempo: float = 1.0, rate: int = 48000) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    filters = [f"atempo={tempo:.4f}"] if abs(tempo - 1.0) > 1e-3 else []
    ffmpeg(["-i", str(src), *(["-filter:a", ",".join(filters)] if filters else []), "-ar", str(rate), "-ac", "1",
            "-c:a", "pcm_s16le", str(out)])
    return out


def mix_dialogue_and_bed(dialogue: Path, bed: Path | None, out: Path, *, dialogue_db: float, bed_db: float,
                         lufs: float, true_peak: float) -> Path:
    norm = f"loudnorm=I={lufs}:TP={true_peak}:LRA=11"
    if bed is None:
        ffmpeg(["-i", str(dialogue), "-af", f"volume={dialogue_db}dB,{norm}", "-ar", "48000", "-ac", "2",
                "-c:a", "aac", "-b:a", "192k", str(out)])
        return out
    graph = (
        f"[0:a]volume={dialogue_db}dB,aformat=channel_layouts=stereo[d];"
        f"[1:a]volume={bed_db}dB,aformat=channel_layouts=stereo[b];"
        f"[d][b]amix=inputs=2:duration=longest:normalize=0,{norm}[m]"
    )
    ffmpeg(["-i", str(dialogue), "-i", str(bed), "-filter_complex", graph, "-map", "[m]", "-ar", "48000",
            "-c:a", "aac", "-b:a", "192k", str(out)])
    return out


def mux(video: Path, audio: Path, out: Path) -> Path:
    ffmpeg(["-i", str(video), "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "copy",
            "-shortest", "-movflags", "+faststart", str(out)])
    return out


def burn_subtitles(video: Path, srt: Path, out: Path) -> Path:
    escaped = str(srt).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    ffmpeg(["-i", str(video), "-vf", f"subtitles='{escaped}'", *X264_FINAL,
            "-c:a", "copy", "-movflags", "+faststart", str(out)])
    return out


def still(video: Path, at: float, out: Path, *, width: int | None = None) -> Path:
    vf = ["-vf", f"scale={width}:-2"] if width else []
    ffmpeg(["-ss", f"{at:.3f}", "-i", str(video), "-frames:v", "1", *vf, "-q:v", "3", str(out)])
    return out


def video_checks(video: Path) -> dict[str, float | int]:
    """Objective checks: black frames, frozen frames, integrated loudness."""
    log = ffmpeg(["-i", str(video), "-vf", "blackdetect=d=0.5:pic_th=0.98,freezedetect=n=0.003:d=2",
                  "-af", "ebur128=peak=true", "-f", "null", "-"], capture=True)
    black = re.findall(r"black_duration:([\d.]+)", log)
    frozen = re.findall(r"freeze_duration: ([\d.]+)", log)
    loud = re.findall(r"I:\s+(-?[\d.]+) LUFS", log)
    peak = re.findall(r"Peak:\s+(-?[\d.]+) dBFS", log)
    return {
        "black_segments": len(black),
        "black_seconds": round(sum(map(float, black)), 3),
        "frozen_segments": len(frozen),
        "frozen_seconds": round(sum(map(float, frozen)), 3),
        "integrated_lufs": float(loud[-1]) if loud else float("nan"),
        "true_peak_dbtp": float(peak[-1]) if peak else float("nan"),
    }


def srt_timestamp(seconds: float) -> str:
    ms = int(round(max(0.0, seconds) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(entries: Iterable[tuple[float, float, str]], out: Path) -> Path:
    blocks = []
    for index, (start, end, text) in enumerate(entries, 1):
        if text.strip():
            blocks.append(f"{index}\n{srt_timestamp(start)} --> {srt_timestamp(end)}\n{text.strip()}\n")
    out.write_text("\n".join(blocks), encoding="utf-8")
    return out
