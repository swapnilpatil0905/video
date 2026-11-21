"""Speaker diarization worker (pyannote.audio). Runtime: diar. Needs HF_TOKEN (gated model)."""

from __future__ import annotations

import os


def run(request: dict) -> dict:
    import torch
    from pyannote.audio import Pipeline

    token = os.environ.get("HF_TOKEN")
    try:
        pipeline = Pipeline.from_pretrained(request["hf_repo"], token=token)
    except TypeError:  # pyannote < 4
        pipeline = Pipeline.from_pretrained(request["hf_repo"], use_auth_token=token)
    if pipeline is None:
        raise RuntimeError(
            f"could not load {request['hf_repo']}: accept its terms on huggingface.co and set HF_TOKEN in .env"
        )
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))

    kwargs = {k: request[k] for k in ("min_speakers", "max_speakers") if request.get(k)}
    output = pipeline(request["audio"], **kwargs)
    annotation = getattr(output, "speaker_diarization", output)

    turns = []
    if hasattr(annotation, "itertracks"):
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            turns.append((float(turn.start), float(turn.end), str(speaker)))
    else:
        for turn, speaker in annotation:
            turns.append((float(turn.start), float(turn.end), str(speaker)))
    turns.sort()
    return {"turns": [{"start": s, "end": e, "speaker_id": spk} for s, e, spk in turns if e > s]}
