"""Mandarin ASR with FunASR Paraformer (+ VAD + punctuation, sentence timestamps). Runtime: audio."""

from __future__ import annotations

from .common import device


def run(request: dict) -> dict:
    from funasr import AutoModel

    dev = "cuda:0" if device() == "cuda" else "cpu"
    p = request.get("params", {})
    model = AutoModel(model=p.get("model", "paraformer-zh"), vad_model=p.get("vad_model", "fsmn-vad"),
                      punc_model=p.get("punc_model", "ct-punc"), hub=p.get("hub", "ms"), device=dev, disable_update=True)
    result = model.generate(input=request["audio"], batch_size_s=300, sentence_timestamp=True)
    sentences = (result[0] if result else {}).get("sentence_info") or []
    segments = []
    for index, sent in enumerate(sentences):
        text = str(sent.get("text", "")).strip()
        if not text:
            continue
        segments.append({"id": index, "start": float(sent["start"]) / 1000.0, "end": float(sent["end"]) / 1000.0,
                         "text": text})
    if not segments:
        raise RuntimeError("FunASR returned no sentence timestamps")
    return {"language": request.get("language") or "zh", "segments": segments}
