"""Speech emotion / audio event worker (SenseVoiceSmall via FunASR). Runtime: audio."""

from __future__ import annotations

import re

from .common import device

_TAG = re.compile(r"<\|([^|]+)\|>")
_EMOTIONS = {
    "HAPPY": "happy", "SAD": "sad", "ANGRY": "angry", "NEUTRAL": "neutral", "FEARFUL": "fearful",
    "DISGUSTED": "disgusted", "SURPRISED": "surprised", "EMO_UNKNOWN": "unknown",
}
_EVENTS = {"BGM", "Speech", "Applause", "Laughter", "Cry", "Sneeze", "Breath", "Cough"}


def parse_tags(text: str) -> dict:
    tags = _TAG.findall(text)
    emotion = next((_EMOTIONS[t] for t in tags if t in _EMOTIONS), "unknown")
    events = [t.lower() for t in tags if t in _EVENTS]
    return {"emotion": emotion, "events": events}


def run(request: dict) -> dict:
    from funasr import AutoModel

    dev = "cuda:0" if device() == "cuda" else "cpu"
    model = AutoModel(model=request["hf_repo"], hub="hf", device=dev, disable_update=True)
    files = request["files"]
    results: dict[str, dict] = {}
    errors: dict[str, str] = {}
    batch = int(request.get("batch_size", 32))
    for start in range(0, len(files), batch):
        chunk = files[start : start + batch]
        try:
            outputs = model.generate(
                input=[item["path"] for item in chunk], language=request.get("language", "auto"), use_itn=False
            )
            for item, out in zip(chunk, outputs):
                results[item["id"]] = parse_tags(out.get("text", ""))
        except Exception as exc:  # record and continue: emotion is advisory
            for item in chunk:
                errors[item["id"]] = f"{type(exc).__name__}: {exc}"
    return {"results": results, "errors": errors}
