"""Speaker diarization with DiariZen (WavLM + pyannote clustering). Runtime: diarizen."""

from __future__ import annotations


def run(request: dict) -> dict:
    from diarizen.pipelines.inference import DiariZenPipeline

    pipeline = DiariZenPipeline.from_pretrained(request["hf_repo"])
    annotation = pipeline(request["audio"])
    annotation = getattr(annotation, "speaker_diarization", annotation)
    turns = [(float(t.start), float(t.end), str(spk)) for t, _, spk in annotation.itertracks(yield_label=True)]
    turns.sort()
    return {"turns": [{"start": a, "end": b, "speaker_id": spk} for a, b, spk in turns if b > a]}
