"""Hindi / Indic TTS worker (IndicF5, AI4Bharat). Runtime: indicf5.

Speaks in the *designed* reference voice (from voice_design) using its known transcript,
never a source actor's voice.
"""

from __future__ import annotations

from .common import Progress


def run(request: dict) -> dict:
    import numpy as np
    import soundfile as sf
    from transformers import AutoModel

    model = AutoModel.from_pretrained(request["hf_repo"], trust_remote_code=True)
    try:
        import torch

        if torch.cuda.is_available():
            model = model.to("cuda")
    except Exception:  # noqa: BLE001 - CPU fallback
        pass
    rate = int(request.get("sample_rate", 24000))
    out, errors = {}, {}
    progress = Progress("tts-indicf5", len(request["items"]))
    for item in request["items"]:
        progress.step()
        try:
            audio = model(item["text"], ref_audio_path=item["ref"], ref_text=item["ref_text"])
            audio = np.asarray(audio)
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            audio = audio.astype(np.float32).squeeze()
            sf.write(item["out"], audio, samplerate=rate)
            out[item["id"]] = {"path": item["out"], "duration_s": float(len(audio) / rate)}
        except Exception as exc:  # noqa: BLE001
            errors[item["id"]] = f"{type(exc).__name__}: {exc}"[:500]
    return {"lines": out, "errors": errors}
