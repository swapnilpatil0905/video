"""Analysis stage executors (S0–S4): ingestion, audio, shots, faces, alignment, speakers."""

from __future__ import annotations

import csv
import json
import platform
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..core.errors import ConfigError, StageFailure
from ..ingestion.probe import probe_video
from ..migration.legacy import convert_speaker_turns, convert_transcript
from ..orchestration.runner import StageContext, executor
from ..schemas.audio import AudioSegment, segment_id, speaker_id
from ..schemas.characters import (
    BoundingBox, Evidence, FaceDetection, RepresentativeFace, SourceCharacter, SpeakerCharacterMapping,
    source_character_id,
)
from ..schemas.media import Keyframe, Shot, shot_id
from . import media


def _dump(models: list[Any]) -> list[dict[str, Any]]:
    return [m.model_dump(mode="json", exclude_none=True) for m in models]


def _hf_repo(s: StageContext, model_id: str) -> str:
    record = s.ctx.registry.get(model_id)
    if not record.weights.hf_repo:
        raise ConfigError(f"model {model_id} has no weights.hf_repo in the registry")
    return record.weights.hf_repo


def load_shots(s: StageContext) -> list[dict[str, Any]]:
    return s.dep_json("shot_detection", "shots.json")


def load_segments(s: StageContext) -> list[dict[str, Any]]:
    return s.dep_json("alignment", "audio_segments.json")


def load_embeddings(s: StageContext):
    """(face_ids, embeddings) from the platform format or the legacy npz (int ids)."""
    import numpy as np

    data = np.load(s.dep_path("face_embedding", "embeddings.npz", "face_embeddings.npz"))
    ids = data["face_ids"]
    face_ids = [f"FACE_{int(i):06d}" for i in ids] if ids.dtype.kind in "iu" else [str(i) for i in ids]
    return face_ids, data["embeddings"].astype("float32")


def load_clusters(s: StageContext) -> dict[str, str]:
    """face_id → cluster label (CHAR_xxxx | UNASSIGNED)."""
    ref = s.dep_ref("face_clustering", "clusters.json", required=False)
    if ref is not None:
        return s.dep_json("face_clustering", "clusters.json")
    path = s.dep_path("face_clustering", "clustered_faces.csv")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {f"FACE_{int(r['face_id']):06d}": r["cluster_id"] for r in csv.DictReader(handle)}


def character_centroids(s: StageContext, min_faces: int = 3) -> dict[str, list[float]]:
    import numpy as np

    face_ids, vectors = load_embeddings(s)
    clusters = load_clusters(s)
    members: dict[str, list[int]] = defaultdict(list)
    for index, face in enumerate(face_ids):
        label = clusters.get(face)
        if label and label != "UNASSIGNED":
            members[source_character_id(label)].append(index)
    centroids = {}
    for character, indices in members.items():
        if len(indices) >= min_faces:
            centroid = vectors[indices].mean(axis=0)
            centroids[character] = (centroid / (np.linalg.norm(centroid) + 1e-9)).tolist()
    return centroids


# ---------------------------------------------------------------------------
# S0 / audio
# ---------------------------------------------------------------------------


@executor("ingest")
def ingest(s: StageContext) -> None:
    source = s.loaded.source_video_path
    meta = probe_video(source, sha256=s.planner.source_hash())
    s.put_json("video_metadata.json", meta.model_dump(mode="json"))
    free = shutil.disk_usage(s.loaded.state_dir).free
    if free < meta.size_bytes * 15:
        s.warn(f"low disk: {free / 1e9:.1f} GB free, production may need ~{meta.size_bytes * 15 / 1e9:.0f} GB")
    for warning in meta.warnings:
        s.warn(warning)
    s.metric("duration_s", meta.duration_s, unit="s")
    s.metric("resolution", f"{meta.width}x{meta.height}")
    s.metric("fps", round(meta.fps, 3), unit="fps")


@executor("extract_audio")
def extract_audio(s: StageContext) -> None:
    p = s.params
    out = media.extract_wav(s.loaded.source_video_path, s.workdir / "speech.wav", stream=p["stream_index"],
                            rate=p["sample_rate"], channels=p["channels"], codec=p["codec"])
    s.put_file(out)


@executor("extract_audio_full")
def extract_audio_full(s: StageContext) -> None:
    out = media.extract_wav(s.loaded.source_video_path, s.workdir / "full_48k.wav", rate=48000, channels=2)
    s.put_file(out)


@executor("asr")
def asr(s: StageContext) -> None:
    p = s.params
    if p["backend"] == "mlx_whisper" and platform.machine() not in ("arm64",):
        raise ConfigError("analysis.asr.backend 'mlx_whisper' runs only on Apple Silicon; "
                          "use backend: faster_whisper, model: faster-whisper-large-v3")
    audio = s.dep_path("extract_audio", "speech.wav", ".wav")
    result = None
    if p["backend"] == "funasr":
        try:
            record = s.ctx.registry.get(p["model"])
            result = s.runtime.call_worker("audio", "asr_funasr", {
                "audio": str(audio), "language": p["language"], "params": record.inference_parameters,
            }, workdir=s.workdir)
            s.provenance["engine"] = f"funasr ({p['model']})"
        except Exception as exc:  # noqa: BLE001
            s.warn(f"FunASR failed, falling back to {p['fallback_model']}: {str(exc)[:200]}")
    if result is None:
        model = p["model"] if p["backend"] == "faster_whisper" else p["fallback_model"]
        result = s.runtime.call_worker("core", "asr", {
            "mode": "transcribe", "model": _hf_repo(s, model), "audio": str(audio), "language": p["language"],
            "condition_on_previous_text": p["condition_on_previous_text"],
        }, workdir=s.workdir)
        s.provenance["engine"] = f"faster-whisper ({model})"
    segments = convert_transcript(result, s.config.languages.source)
    s.put_json("transcript.json", _dump(segments))
    s.metric("segment_count", len(segments))


@executor("transcript_filter")
def transcript_filter(s: StageContext) -> None:
    patterns = [p.lower() for p in s.params["drop_patterns"]]
    kept, removed = [], []
    for seg in s.dep_json("asr", "transcript.json"):
        if any(p in seg["text"].lower() for p in patterns):
            removed.append(seg["segment_id"])
            continue
        if seg["end_s"] <= seg["start_s"] and "invalid_timestamp" not in seg.get("flags", []):
            seg.setdefault("flags", []).append("invalid_timestamp")
        kept.append(seg)
    s.put_json("transcript_clean.json", {"segments": kept, "removed_segment_ids": removed})
    s.metric("segment_count", len(kept))
    s.metric("removed_by_filter", len(removed))


@executor("diarization")
def diarization(s: StageContext) -> None:
    p = s.params
    audio = s.dep_path("extract_audio", "speech.wav", ".wav")
    result = None
    if p["engine"] == "diarizen":
        try:
            result = s.runtime.call_worker("diarizen", "diarize_diarizen", {
                "hf_repo": _hf_repo(s, p["model"]), "audio": str(audio)}, workdir=s.workdir)
            s.provenance["engine"] = f"diarizen ({p['model']})"
        except Exception as exc:  # noqa: BLE001
            s.warn(f"DiariZen failed, falling back to pyannote: {str(exc)[:200]}")
    if result is None:
        model = p["model"] if p["engine"] == "pyannote" else p["fallback_model"]
        result = s.runtime.call_worker("diar", "diarize", {
            "hf_repo": _hf_repo(s, model), "audio": str(audio),
            "min_speakers": p["min_speakers"], "max_speakers": p["max_speakers"],
        }, workdir=s.workdir)
        s.provenance["engine"] = f"pyannote ({model})"
    turns = convert_speaker_turns(result)
    s.put_json("speaker_turns.json", _dump(turns))
    s.metric("turn_count", len(turns))
    s.metric("speaker_count", len({t.speaker_id for t in turns}))


def find_separation_outputs(folder: Path) -> tuple[Path | None, Path | None]:
    """audio-separator names outputs '<stem>_(Vocals)_<model>.wav' / '<stem>_(Instrumental)_<model>.wav'."""
    vocals = next(iter(sorted(folder.glob("*(Vocals)*.wav"))), None)
    rest = next(iter(sorted(folder.glob("*(Instrumental)*.wav"))), None)
    return vocals, rest


@executor("source_separation")
def source_separation(s: StageContext) -> None:
    p = s.params
    audio = s.dep_path("extract_audio_full", "full_48k.wav")
    if p["engine"] == "bs_roformer":
        out_dir = s.workdir / "roformer"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            s.runtime.run_command("sep", [
                str(Path(s.runtime.python("sep")).parent / "audio-separator"), str(audio),
                "--model_filename", p["roformer_model_file"], "--output_dir", str(out_dir), "--output_format", "WAV",
                "--model_file_dir", str(s.runtime.weights_dir / "audio-separator"),
            ], log_name="bs_roformer")
            vocals, rest = find_separation_outputs(out_dir)
            if vocals and rest:
                s.put_file(vocals, "vocals.wav")
                s.put_file(rest, "music_effects.wav")
                s.provenance["engine"] = f"bs_roformer ({p['roformer_model_file']})"
                return
            s.warn(f"BS-RoFormer finished but outputs were not found in {out_dir}; falling back to Demucs")
        except Exception as exc:  # noqa: BLE001 - fallback keeps the run going
            s.warn(f"BS-RoFormer unavailable ({str(exc)[:200]}); falling back to Demucs")
    out_root = s.workdir / "demucs"
    s.runtime.run_command("audio", [
        s.runtime.python("audio"), "-m", "demucs", "-n", p["demucs_model"], "--two-stems=vocals",
        "--segment", str(int(p["segment_seconds"])), "-o", str(out_root), str(audio),  # demucs wants an int
    ], log_name="demucs")
    stem_dir = out_root / p["demucs_model"] / audio.stem
    vocals, rest = stem_dir / "vocals.wav", stem_dir / "no_vocals.wav"
    if not (vocals.exists() and rest.exists()):
        raise StageFailure(f"demucs outputs not found in {stem_dir}")
    s.put_file(vocals, "vocals.wav")
    s.put_file(rest, "music_effects.wav")
    s.provenance["engine"] = f"demucs ({p['demucs_model']})"


# ---------------------------------------------------------------------------
# Shots and faces
# ---------------------------------------------------------------------------


@executor("shot_detection")
def shot_detection(s: StageContext) -> None:
    p = s.params
    if p["detector"] == "transnetv2":
        raise ConfigError("transnetv2 executor is not available yet; use pyscenedetect_content or _adaptive")
    from scenedetect import SceneManager, open_video
    from scenedetect.detectors import AdaptiveDetector, ContentDetector

    video = open_video(str(s.loaded.source_video_path))
    manager = SceneManager()
    if p["detector"] == "pyscenedetect_adaptive":
        manager.add_detector(AdaptiveDetector(min_scene_len=p["min_scene_len_frames"]))
    else:
        manager.add_detector(ContentDetector(threshold=p["threshold"], min_scene_len=p["min_scene_len_frames"]))
    manager.detect_scenes(video, show_progress=False)
    scenes = manager.get_scene_list()
    meta = s.dep_json("ingest", "video_metadata.json")
    bounds = [(a.get_seconds(), b.get_seconds()) for a, b in scenes] or [(0.0, meta["duration_s"])]
    shots = [Shot(shot_id=shot_id(i), start_s=round(a, 3), end_s=round(b, 3), legacy_scene_number=i)
             for i, (a, b) in enumerate(bounds, 1)]
    s.put_json("shots.json", _dump(shots))
    s.metric("shot_count", len(shots))


@executor("frame_sampling")
def frame_sampling(s: StageContext) -> None:
    import cv2

    p = s.params
    shots = load_shots(s)
    capture = cv2.VideoCapture(str(s.loaded.source_video_path))
    if not capture.isOpened():
        raise StageFailure("cannot open source video")
    keyframes = []
    frame_dir = s.workdir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for shot in shots:
        start, end = shot["start_s"], shot["end_s"]
        if p["strategy"] == "midpoint" or not p["fps"]:
            times = [(start + end) / 2]
        else:
            count = max(1, min(p["max_per_shot"], int((end - start) * p["fps"])))
            times = [start + (end - start) * (i + 0.5) / count for i in range(count)]
        for index, t in enumerate(times):
            capture.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = capture.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (p["width"], int(h * p["width"] / w)))
            name = f"{shot['shot_id']}_{index:02d}.jpg"
            path = frame_dir / name
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, p["jpeg_quality"]])
            ref = s.put_file(path, f"frames/{name}")
            keyframes.append(Keyframe(shot_id=shot["shot_id"], time_s=round(t, 3), uri=ref.uri))
    capture.release()
    s.put_json("keyframes.json", _dump(keyframes))
    s.metric("keyframe_count", len(keyframes))


@executor("face_detection")
def face_detection(s: StageContext) -> None:
    p = s.params
    record = s.ctx.registry.get(p["model"])
    images = []
    for kf in s.dep_json("frame_sampling", "keyframes.json"):
        path = s.store.materialize(s.store.key_from_uri(kf["uri"]), s.cache_dir)
        images.append({"path": str(path), "shot_id": kf["shot_id"], "time_s": kf["time_s"]})
    if p["model"].startswith("insightface"):
        npz = s.workdir / "embeddings.npz"
        result = s.runtime.call_worker("core", "faces", {
            "mode": "detect_insightface", "images": images, "crop_dir": str(s.workdir / "crops"), "out_npz": str(npz),
            "det_size": 640, **{k: p[k] for k in ("confidence_threshold", "padding", "crop_size", "jpeg_quality")},
        }, workdir=s.workdir)
        s.put_file(npz, "embeddings.npz")
        s.put_json("face_attributes.json", result["attributes"])
    elif p["model"].startswith("opencv-yunet"):
        result = None
    else:
        raise ConfigError("face_detection supports opencv-yunet-* and insightface-* models")
    if result is None:
        result = s.runtime.call_worker("core", "faces", {
        "mode": "detect", "hf_repo": record.weights.hf_repo, "model_file": (record.weights.files or
                                                                            ["face_detection_yunet_2023mar.onnx"])[0],
        "images": images, "crop_dir": str(s.workdir / "crops"), **{k: p[k] for k in (
            "confidence_threshold", "nms_threshold", "top_k", "padding", "crop_size", "jpeg_quality")},
    }, workdir=s.workdir)
    detections = []
    for d in result["detections"]:
        face = f"FACE_{d['face_number']:06d}"
        ref = s.put_file(Path(d["crop"]), f"crops/{face}.jpg")
        x, y, w, h = d["bbox"]
        detections.append(FaceDetection(face_id=face, shot_id=d["shot_id"], time_s=d["time_s"],
                                        bbox=BoundingBox(x=x, y=y, width=w, height=h),
                                        confidence=d["confidence"], crop_uri=ref.uri))
    s.put_json("face_detections.json", _dump(detections))
    s.metric("face_count", len(detections))


@executor("face_embedding")
def face_embedding(s: StageContext) -> None:
    p = s.params
    reused = s.dep_ref("face_detection", "embeddings.npz", required=False)
    if reused is not None and p["model"].startswith("insightface") and \
            s.ctx.registry.get(s.config.analysis.face_detection.model).model_id == p["model"]:
        # same model already produced identity embeddings during detection: reuse them (no second pass)
        s.put_file(s.dep_path("face_detection", "embeddings.npz"), "embeddings.npz")
        attrs = s.dep_ref("face_detection", "face_attributes.json", required=False)
        if attrs is not None:
            s.put_json("face_attributes.json", s.dep_json("face_detection", "face_attributes.json"))
        s.metric("embedding_count", "reused from face_detection")
        return
    crops = []
    for d in s.dep_json("face_detection", "face_detections.json"):
        path = s.store.materialize(s.store.key_from_uri(d["crop_uri"]), s.cache_dir)
        crops.append({"face_id": d["face_id"], "path": str(path)})
    out = s.workdir / "embeddings.npz"
    result = s.runtime.call_worker("core", "faces", {"mode": "embed", "crops": crops, "det_size": p["det_size"],
                                                      "out_npz": str(out)}, workdir=s.workdir)
    s.put_file(out)
    s.put_json("face_attributes.json", result["attributes"])
    s.metric("embedding_count", result["count"])


@executor("face_clustering")
def face_clustering(s: StageContext) -> None:
    from sklearn.cluster import DBSCAN

    p = s.params
    face_ids, vectors = load_embeddings(s)
    labels = DBSCAN(eps=p["eps"], min_samples=p["min_samples"], metric=p["metric"]).fit_predict(vectors) if len(face_ids) else []
    clusters = {face: ("UNASSIGNED" if int(l) == -1 else f"CHAR_{int(l) + 1:04d}") for face, l in zip(face_ids, labels)}
    s.put_json("clusters.json", clusters)
    s.metric("cluster_count", len({c for c in clusters.values() if c != "UNASSIGNED"}))
    s.metric("unassigned_faces", sum(c == "UNASSIGNED" for c in clusters.values()))


@executor("character_database")
def character_database(s: StageContext) -> None:
    import numpy as np

    k = s.params["representatives_per_cluster"]
    detections = {d["face_id"]: d for d in s.dep_json("face_detection", "face_detections.json")}
    face_ids, vectors = load_embeddings(s)
    index = {f: i for i, f in enumerate(face_ids)}
    members: dict[str, list[str]] = defaultdict(list)
    for face, label in load_clusters(s).items():
        if label != "UNASSIGNED":
            members[label].append(face)
    characters = []
    for label, faces in members.items():
        idx = [index[f] for f in faces if f in index]
        sims = np.array([])
        if idx:
            centroid = vectors[idx].mean(axis=0)
            centroid /= np.linalg.norm(centroid) + 1e-9
            sims = vectors[idx] @ centroid
        order = np.argsort(-sims) if len(sims) else []
        shots = sorted({detections[f]["shot_id"] for f in faces if f in detections})
        reps = []
        for i in list(order)[:k]:
            face = faces[int(i)]
            det = detections.get(face, {})
            reps.append(RepresentativeFace(face_id=face, shot_id=det.get("shot_id", ""), uri=det.get("crop_uri"),
                                           centroid_similarity=round(float(sims[int(i)]), 4)))
        characters.append(SourceCharacter(
            character_id=source_character_id(label), legacy_ids=[label], face_count=len(faces), shot_ids=shots,
            first_shot_id=shots[0] if shots else None, last_shot_id=shots[-1] if shots else None,
            mean_similarity=round(float(sims.mean()), 4) if len(sims) else None,
            p10_similarity=round(float(np.percentile(sims, 10)), 4) if len(sims) else None,
            representative_faces=reps,
        ))
    characters.sort(key=lambda c: -c.face_count)
    s.put_json("source_characters.json", _dump(characters))
    s.metric("character_count", len(characters))


# ---------------------------------------------------------------------------
# Alignment and speakers
# ---------------------------------------------------------------------------


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _strength(ratio: float, strong: float, moderate: float) -> str:
    return "strong" if ratio >= strong else "moderate" if ratio >= moderate else "weak" if ratio > 0 else "none"


def align(transcript: list[dict], turns: list[dict], shots: list[dict], strong: float, moderate: float) -> list[AudioSegment]:
    """Transcript × diarization × shots. Invalid timestamps are skipped (and stay flagged upstream)."""
    out = []
    for seg in transcript:
        start, end = seg["start_s"], seg["end_s"]
        if end <= start:
            continue
        dur = end - start
        speakers: dict[str, float] = defaultdict(float)
        for t in turns:
            if t["start_s"] < end and t["end_s"] > start:
                speakers[t["speaker_id"]] += _overlap(start, end, t["start_s"], t["end_s"])
        shot_overlaps = [(sh["shot_id"], _overlap(start, end, sh["start_s"], sh["end_s"]))
                         for sh in shots if sh["start_s"] < end and sh["end_s"] > start]
        best_speaker = max(speakers, key=speakers.get) if speakers else None
        best_shot = max(shot_overlaps, key=lambda x: x[1]) if shot_overlaps else None
        sp_ratio = min(1.0, speakers[best_speaker] / dur) if best_speaker else 0.0
        sh_ratio = min(1.0, best_shot[1] / dur) if best_shot else 0.0
        out.append(AudioSegment(
            segment_id=seg["segment_id"], start_s=start, end_s=end, text=seg["text"], language=seg.get("language"),
            speaker_id=best_speaker, speaker_overlap_ratio=round(sp_ratio, 4),
            speaker_match=_strength(sp_ratio, strong, moderate),  # type: ignore[arg-type]
            overlapping_speakers=sorted(k for k in speakers if k != best_speaker),
            shot_id=best_shot[0] if best_shot else None, shot_match=_strength(sh_ratio, strong, moderate),  # type: ignore[arg-type]
            overlapping_shot_ids=[sid for sid, _ in shot_overlaps],
            flags=["overlapping_speech"] if len(speakers) > 1 else [],
        ))
    return out


@executor("alignment")
def alignment(s: StageContext) -> None:
    p = s.params
    transcript = s.dep_json("transcript_filter", "transcript_clean.json")["segments"]
    segments = align(transcript, s.dep_json("diarization", "speaker_turns.json"), load_shots(s),
                     p["strong_ratio"], p["moderate_ratio"])
    s.put_json("audio_segments.json", _dump(segments))
    s.metric("segment_count", len(segments))
    for level in ("strong", "moderate", "weak", "none"):
        s.metric(f"speaker_match_{level}", sum(x.speaker_match == level for x in segments))


@executor("speaker_mapping_prior")
def speaker_mapping_prior(s: StageContext) -> None:
    characters = s.dep_json("character_database", "source_characters.json")
    by_shot: dict[str, set[str]] = defaultdict(set)
    for c in characters:
        for sh in c["shot_ids"]:
            by_shot[sh].add(c["character_id"])
    evidence: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for seg in load_segments(s):
        if not seg.get("speaker_id") or not seg.get("shot_id"):
            continue
        visible = by_shot.get(seg["shot_id"], set())
        dur = seg["end_s"] - seg["start_s"]
        for c in visible:
            e = evidence[(seg["speaker_id"], c)]
            e["single" if len(visible) == 1 else "multi"] += dur
            e["segments"] += 1
    mappings = []
    for (spk, char), e in sorted(evidence.items()):
        mappings.append(SpeakerCharacterMapping(
            speaker_id=spk, character_id=char, status="uncertain", supporting_segments=int(e["segments"]),
            supporting_speech_s=round(e["single"], 3),
            evidence=[Evidence(kind="cooccurrence", detail={"single_character_seconds": round(e["single"], 3),
                                                            "multi_character_seconds": round(e["multi"], 3)})],
            uncertainty_reason="co-occurrence only: visible during speech is not proof of speaking",
        ))
    s.put_json("speaker_character_prior.json", _dump(mappings))
    s.metric("candidate_count", len(mappings))


@executor("emotion")
def emotion(s: StageContext) -> None:
    import soundfile as sf

    p = s.params
    segments = load_segments(s)
    emotions = {seg["segment_id"]: {"emotion": "unknown", "events": []} for seg in segments}
    if not p["enabled"]:
        s.warn("emotion analysis disabled; all segments 'unknown'")
        s.put_json("emotions.json", emotions)
        return
    audio_path = s.dep_path("extract_audio", "speech.wav", ".wav")
    data, rate = sf.read(str(audio_path), dtype="float32")
    clip_dir = s.workdir / "clips"
    clip_dir.mkdir(exist_ok=True)
    files = []
    for seg in segments:
        a, b = int(seg["start_s"] * rate), int(seg["end_s"] * rate)
        if b - a < rate * 0.3:
            continue
        path = clip_dir / f"{seg['segment_id']}.wav"
        sf.write(str(path), data[a:b], rate)
        files.append({"id": seg["segment_id"], "path": str(path)})
    try:
        result = s.runtime.call_worker("audio", "emotion", {
            "engine": p["engine"], "emotion2vec_repo": p["emotion2vec_repo"],
            "hf_repo": p["hf_repo"], "files": files, "language": s.config.languages.source,
        }, workdir=s.workdir)
        if p["engine"] == "emotion2vec" and not result["results"]:
            s.warn("emotion2vec produced no results; falling back to SenseVoice")
            result = s.runtime.call_worker("audio", "emotion", {
                "engine": "sensevoice", "hf_repo": p["hf_repo"], "files": files, "language": s.config.languages.source,
            }, workdir=s.workdir, tag="sensevoice")
    except Exception as exc:
        if p["required"]:
            raise
        s.warn(f"emotion model failed, all segments 'unknown': {exc}")
        result = {"results": {}, "errors": {}}
    for seg_id, value in result["results"].items():
        emotions[seg_id] = value
    if result.get("errors"):
        s.warn(f"{len(result['errors'])} segments failed emotion analysis (marked unknown)")
    s.put_json("emotions.json", emotions)
    counts: dict[str, int] = defaultdict(int)
    for value in emotions.values():
        counts[value["emotion"]] += 1
    for name, count in counts.items():
        s.metric(f"emotion_{name}", count)


@executor("active_speaker")
def active_speaker(s: StageContext) -> None:
    p = s.params
    shots = {sh["shot_id"]: sh for sh in load_shots(s)}
    centroids = character_centroids(s)
    segments = [
        {"segment_id": seg["segment_id"], "start": seg["start_s"], "end": seg["end_s"],
         "shot_start": shots[seg["shot_id"]]["start_s"], "shot_end": shots[seg["shot_id"]]["end_s"]}
        for seg in load_segments(s) if seg.get("shot_id") in shots
    ]
    result = None
    method = p["method"]
    if method == "light_asd":
        try:
            result = s.runtime.call_worker("asd", "asd_light", {
                "repo": str(s.runtime.repo("lightasd")), "weights": p["light_asd_weights"],
                "video": str(s.loaded.source_video_path), "audio": str(_speech_audio(s)),
                "segments": segments, "centroids": centroids, "params": p,
            }, workdir=s.workdir, timeout_minutes=600)
            if result.get("fatal"):
                raise RuntimeError(result["fatal"][-300:])
        except Exception as exc:  # noqa: BLE001
            s.warn(f"Light-ASD unavailable, using the lower-face-motion heuristic: {str(exc)[:300]}")
            result, method = None, "lower_face_motion"
    if result is None:
        result = s.runtime.call_worker("core", "asd", {
            "video": str(s.loaded.source_video_path), "segments": segments, "centroids": centroids, "params": p,
        }, workdir=s.workdir)
    decisions = {}
    threshold = p["speaking_probability"] if method == "light_asd" else None
    counts: dict[str, int] = defaultdict(int)
    for seg_id, r in result["results"].items():
        visible = [(c, v) for c, v in r["characters"].items()
                   if v["visible_ratio"] >= p["min_visible_ratio"] and v["motion"] is not None]
        visible.sort(key=lambda x: -x[1]["motion"])
        if threshold is not None:  # trained model: probabilities
            visible = [(c, v) for c, v in visible if v["motion"] >= threshold] or []
        if not visible:
            decision, speaker = "no_visible_face" if threshold is None else "no_confident_speaker", None
        elif len(visible) == 1:
            decision, speaker = ("asd_margin" if threshold is not None else "single_visible"), visible[0][0]
        elif visible[0][1]["motion"] >= p["margin_ratio"] * max(visible[1][1]["motion"], 1e-6):
            decision, speaker = "asd_margin", visible[0][0]
        else:
            decision, speaker = "ambiguous", None
        counts[decision] += 1
        decisions[seg_id] = {"decision": decision, "speaking_character": speaker, "characters": r["characters"]}
    s.put_json("active_speaker.json", decisions)
    for name, count in counts.items():
        s.metric(f"asd_{name}", count)
    s.provenance["method"] = ("Light-ASD (trained, AVA-ActiveSpeaker) + identity" if method == "light_asd"
                              else "heuristic lower-face motion + identity (fallback)")


def _speech_audio(s: StageContext) -> Path:
    """16 kHz speech track: legacy audio/source.wav or this run's speech.wav (via the alignment's inputs)."""
    candidates = [s.loaded.platform_root / "audio" / "source.wav"]
    for record in s.index.list_records(s.loaded.project_id, "extract_audio"):
        for ref in record.outputs:
            if ref.uri and ref.uri.endswith(".wav"):
                candidates.append(s.store.materialize(s.store.key_from_uri(ref.uri), s.cache_dir))
    for path in candidates:
        if Path(path).exists():
            return Path(path)
    raise StageFailure("no 16 kHz speech audio found for active speaker detection")


@executor("speaker_mapping")
def speaker_mapping(s: StageContext) -> None:
    p = s.params
    asd = s.dep_json("active_speaker", "active_speaker.json")
    prior = s.dep_json("speaker_mapping_prior", "speaker_character_prior.json")
    weights = {"asd_margin": 1.0, "single_visible": 0.4}
    votes: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    strong: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    speech: dict[str, float] = defaultdict(float)
    for seg in load_segments(s):
        spk = seg.get("speaker_id")
        if not spk:
            continue
        speech[spk] += seg["end_s"] - seg["start_s"]
        d = asd.get(seg["segment_id"])
        if d and d["speaking_character"] and d["decision"] in weights:
            votes[spk][d["speaking_character"]] += weights[d["decision"]]
            if d["decision"] == "asd_margin":
                strong[spk][d["speaking_character"]] += 1
    prior_by_spk: dict[str, list[dict]] = defaultdict(list)
    for m in prior:
        prior_by_spk[m["speaker_id"]].append(m)
    mappings = []
    for spk in sorted(speech):
        v = votes.get(spk, {})
        total = sum(v.values())
        prior_evidence = [Evidence(kind="cooccurrence", detail=m["evidence"][0]["detail"] | {"character_id": m["character_id"]})
                          for m in sorted(prior_by_spk[spk], key=lambda m: -m["supporting_speech_s"])[:3]]
        if not total:
            mappings.append(SpeakerCharacterMapping(
                speaker_id=spk, character_id=None, status="uncertain", supporting_speech_s=round(speech[spk], 2),
                evidence=prior_evidence, uncertainty_reason="no active-speaker evidence (off-screen / voice-over / faces not visible)",
            ))
            continue
        best = max(v, key=v.get)
        confidence = v[best] / total
        n_strong = strong[spk][best]
        accepted = confidence >= p["accept_confidence"] and n_strong >= p["min_supporting_segments"]
        reason = None if accepted else (
            f"confidence {confidence:.2f} < {p['accept_confidence']}" if confidence < p["accept_confidence"]
            else f"only {n_strong} clear active-speaker segments (< {p['min_supporting_segments']})")
        mappings.append(SpeakerCharacterMapping(
            speaker_id=spk, character_id=best, status="accepted" if accepted else "uncertain",
            confidence=round(confidence, 3), supporting_segments=n_strong, supporting_speech_s=round(speech[spk], 2),
            evidence=[Evidence(kind="asd", weight=round(v[best], 2), detail={"votes": round(v[best], 2), "total": round(total, 2)}),
                      *prior_evidence],
            contradictions=[Evidence(kind="asd", weight=round(w, 2), detail={"character_id": c})
                            for c, w in v.items() if c != best],
            uncertainty_reason=reason,
        ))
    chars = defaultdict(list)
    for m in mappings:
        if m.character_id:
            chars[m.character_id].append(m.speaker_id)
    for char, spks in chars.items():
        if len(spks) > 1:
            s.warn(f"{char} is the best match for several speakers {spks} (diarization over-split or ambiguity)")
    s.put_json("speaker_mapping.json", _dump(mappings))
    s.metric("accepted", sum(m.status == "accepted" for m in mappings))
    s.metric("uncertain", sum(m.status == "uncertain" for m in mappings))


def rank_percentiles(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values, key=values.get)
    n = len(ordered)
    return {k: (i / (n - 1) if n > 1 else 1.0) for i, k in enumerate(ordered)}


def classify_characters(chars: list[dict], speech: dict[str, float], total_shots: int, p: dict) -> list[dict]:
    """Importance score and tier per character (pure function; unit-tested)."""
    eligible = [c for c in chars if c["face_count"] >= p["min_face_count"] or speech.get(c["character_id"], 0) > 0]
    if not eligible:
        return []
    def thirds(c: dict) -> int:
        nums = [int(x.split("_")[1]) for x in c["shot_ids"]] if c["shot_ids"] else []
        return len({min(2, int(3 * (n - 1) / max(1, total_shots))) for n in nums})
    feats = {
        "speech": rank_percentiles({c["character_id"]: speech.get(c["character_id"], 0.0) for c in eligible}),
        "screen": rank_percentiles({c["character_id"]: float(c["face_count"]) for c in eligible}),
        "scenes": rank_percentiles({c["character_id"]: float(len(c["shot_ids"])) for c in eligible}),
        "recurrence": {c["character_id"]: thirds(c) / 3 for c in eligible},
    }
    w = p["weights"]
    out = []
    for c in eligible:
        cid = c["character_id"]
        score = sum(w[k] * feats[k][cid] for k in w)
        if speech.get(cid, 0) == 0 and score < p["supporting_threshold"]:
            tier = "background"
        elif score >= p["main_threshold"]:
            tier = "main"
        elif score >= p["supporting_threshold"]:
            tier = "supporting"
        elif thirds(c) >= p["recurring_min_thirds"]:
            tier = "recurring_secondary"
        else:
            tier = "minor"
        out.append({"character_id": cid, "score": round(score, 4), "tier": tier, "thirds": thirds(c),
                    "speech_s": round(speech.get(cid, 0.0), 2), "face_count": c["face_count"],
                    "shot_count": len(c["shot_ids"])})
    return sorted(out, key=lambda x: -x["score"])


@executor("character_profile")
def character_profile(s: StageContext) -> None:
    from collections import Counter

    chars = s.dep_json("character_database", "source_characters.json")
    mappings = s.dep_json("speaker_mapping", "speaker_mapping.json")
    detections = {d["face_id"]: d for d in s.dep_json("face_detection", "face_detections.json")}
    speech: dict[str, float] = defaultdict(float)
    speakers: dict[str, list[str]] = defaultdict(list)
    for m in mappings:
        if m.get("character_id"):
            speech[m["character_id"]] += m["supporting_speech_s"]
            speakers[m["character_id"]].append(m["speaker_id"])
    total_shots = max((int(sh.split("_")[1]) for c in chars for sh in c["shot_ids"]), default=1)
    profiles = classify_characters(chars, speech, total_shots, s.params)

    crops = []
    keep = {p["character_id"] for p in profiles if p["tier"] != "background"}
    for c in chars:
        if c["character_id"] not in keep:
            continue
        for rep in c["representative_faces"]:
            det = detections.get(rep["face_id"])
            if det and det.get("crop_uri"):
                path = s.store.materialize(s.store.key_from_uri(det["crop_uri"]), s.cache_dir)
                crops.append({"face_id": f"{c['character_id']}|{rep['face_id']}", "path": str(path)})
    attrs = {}
    if crops:
        result = s.runtime.call_worker("core", "faces", {"mode": "embed", "crops": crops, "det_size": 320,
                                                          "out_npz": str(s.workdir / "rep.npz")}, workdir=s.workdir)
        attrs = result["attributes"]
    for prof in profiles:
        genders = Counter(v["gender"] for k, v in attrs.items() if k.startswith(prof["character_id"] + "|") and v["gender"])
        ages = sorted(v["age"] for k, v in attrs.items() if k.startswith(prof["character_id"] + "|") and v["age"] is not None)
        prof["gender"] = {"M": "male", "F": "female"}.get(genders.most_common(1)[0][0], None) if genders else None
        prof["age_estimate"] = ages[len(ages) // 2] if ages else None
        prof["speaker_ids"] = speakers.get(prof["character_id"], [])
    s.put_json("character_profiles.json", profiles)
    for tier in ("main", "supporting", "recurring_secondary", "minor", "background"):
        s.metric(f"tier_{tier}", sum(p["tier"] == tier for p in profiles))
    s.provenance["note"] = "gender/age are model estimates from face crops, used only to design fictional replacements"


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
