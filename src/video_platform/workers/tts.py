"""Dialogue TTS worker (Chatterbox Multilingual). Runtime: tts.

Uses a *generated* reference voice (from voice_design) as the speaker prompt, never a
source actor's voice.
"""

from __future__ import annotations

from .common import device


def run(request: dict) -> dict:
    import soundfile as sf
    import torch
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    dev = "cuda" if device() == "cuda" and torch.cuda.is_available() else "cpu"
    model = ChatterboxMultilingualTTS.from_pretrained(device=dev)
    out, errors = {}, {}
    for item in request["items"]:
        try:
            torch.manual_seed(int(item.get("seed", 0)))
            wav = model.generate(
                item["text"],
                language_id=request["language"],
                audio_prompt_path=item["ref"],
                exaggeration=float(item["exaggeration"]),
                cfg_weight=float(item["cfg_weight"]),
            )
            audio = wav.squeeze().detach().cpu().numpy()
            sf.write(item["out"], audio, model.sr)
            out[item["id"]] = {"path": item["out"], "duration_s": float(len(audio) / model.sr)}
        except Exception as exc:
            errors[item["id"]] = f"{type(exc).__name__}: {exc}"
    return {"lines": out, "errors": errors}
