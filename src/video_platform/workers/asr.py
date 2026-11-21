"""ASR worker (faster-whisper). Runtime: core.

Modes:
  transcribe  — one long file → segments (source-language transcription)
  check       — many short files → text (QA back-transcription of generated speech)
"""

from __future__ import annotations

from .common import device


def run(request: dict) -> dict:
    from faster_whisper import WhisperModel

    dev = device()
    model = WhisperModel(request["model"], device=dev, compute_type="float16" if dev == "cuda" else "int8")
    if request.get("mode", "transcribe") == "check":
        texts = {}
        for item in request["files"]:
            segments, _ = model.transcribe(item["path"], language=request.get("language"), beam_size=5)
            texts[item["id"]] = " ".join(s.text.strip() for s in segments)
        return {"texts": texts}

    segments, info = model.transcribe(
        request["audio"],
        language=request.get("language"),
        vad_filter=True,
        condition_on_previous_text=bool(request.get("condition_on_previous_text", False)),
        beam_size=5,
    )
    out = []
    for index, seg in enumerate(segments):
        out.append({
            "id": index,
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text.strip(),
            "avg_logprob": float(seg.avg_logprob),
            "no_speech_prob": float(seg.no_speech_prob),
            "compression_ratio": float(seg.compression_ratio),
        })
    return {"language": info.language, "segments": out}
