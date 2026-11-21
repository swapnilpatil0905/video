"""Shared test fixtures: a synthetic project and a miniature legacy workspace."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_001_CONFIG = REPO_ROOT / "projects" / "project_001" / "config.yaml"


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


def write_yaml(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def minimal_config(project_id: str = "test_project", **sections) -> dict:
    config = {
        "schema_version": 1,
        "project": {
            "id": project_id,
            "name": "Test project",
            "project_type": "full_remake",
            "source_video": "input/source.mp4",
        },
        "languages": {"source": "zh", "target": "hi"},
        "audio": {"target_language": "hi"},
    }
    for key, value in sections.items():
        config[key] = value
    return config


@pytest.fixture
def make_project(tmp_path, monkeypatch):
    """Create a project directory with config and a dummy source video."""
    monkeypatch.setenv("VP_PLATFORM_ROOT", str(REPO_ROOT))

    def _make(config: dict | None = None, *, with_source: bool = True, name: str = "test_project") -> Path:
        project_dir = tmp_path / "projects" / name
        cfg_path = write_yaml(project_dir / "config.yaml", config or minimal_config(name))
        if with_source:
            (project_dir / "input").mkdir(parents=True, exist_ok=True)
            (project_dir / "input" / "source.mp4").write_bytes(b"fake video bytes")
        return cfg_path

    return _make


FAKE_PROBE = {
    "streams": [
        {"index": 0, "codec_type": "video", "codec_name": "h264", "width": 720, "height": 1280,
         "avg_frame_rate": "30/1", "r_frame_rate": "30/1", "pix_fmt": "yuv420p", "duration": "12.0"},
        {"index": 1, "codec_type": "audio", "codec_name": "aac", "sample_rate": "44100", "channels": 2,
         "duration": "12.0"},
    ],
    "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "12.000", "bit_rate": "2000000"},
}


@pytest.fixture
def fake_probe():
    return lambda path: FAKE_PROBE


def _csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(header)] + [",".join(str(v) for v in row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def legacy_workspace(tmp_path) -> Path:
    """A miniature legacy workspace with the same file formats as the real one."""
    root = tmp_path / "legacy"
    (root / "input").mkdir(parents=True)
    (root / "input" / "source.mp4").write_bytes(b"legacy source video")
    (root / "audio").mkdir()
    (root / "audio" / "source.wav").write_bytes(b"RIFF....WAVE")
    (root / "pipeline").mkdir()
    (root / "pipeline" / "transcribe.py").write_text("# legacy script\n")

    raw = {
        "text": "...",
        "language": "zh",
        "segments": [
            {"id": 0, "start": 0.0, "end": 2.0, "text": "你好", "avg_logprob": -0.2, "no_speech_prob": 0.01,
             "compression_ratio": 1.1},
            {"id": 1, "start": 2.0, "end": 3.0, "text": "优优独播剧场"},
            {"id": 2, "start": 3.0, "end": 3.0, "text": "坏的时间戳"},
            {"id": 3, "start": 4.0, "end": 6.0, "text": "再见"},
        ],
    }
    clean = {**raw, "segments": [s for s in raw["segments"] if s["id"] != 1]}
    (root / "transcripts").mkdir()
    (root / "transcripts" / "chinese_transcript.json").write_text(json.dumps(raw, ensure_ascii=False))
    (root / "transcripts" / "chinese_transcript_clean.json").write_text(json.dumps(clean, ensure_ascii=False))

    (root / "diarization").mkdir()
    turns = {"turns": [{"start": 0.0, "end": 2.1, "duration": 2.1, "speaker_id": "SPEAKER_00"},
                       {"start": 4.0, "end": 6.0, "duration": 2.0, "speaker_id": "SPEAKER_01"}]}
    (root / "diarization" / "speaker_turns.json").write_text(json.dumps(turns))
    _csv(root / "diarization" / "speaker_turns.csv", ["start", "end", "duration", "speaker_id"], [[0.0, 2.1, 2.1, "SPEAKER_00"]])
    (root / "diarization" / "speaker_summary.json").write_text(json.dumps({"processing_seconds": 12.5}))

    _csv(root / "scenes" / "scene_timeline.csv", ["scene_id", "start_seconds", "end_seconds", "duration_seconds"],
         [[1, 0.0, 3.0, 3.0], [2, 3.0, 12.0, 9.0]])
    _csv(root / "scenes" / "frame_manifest.csv", ["scene_id", "start_seconds", "end_seconds", "midpoint_seconds", "frame_path"],
         [[1, 0.0, 3.0, 1.5, "scenes/frames/scene_00001.jpg"], [2, 3.0, 12.0, 7.5, "scenes/frames/scene_00002.jpg"]])
    (root / "scenes" / "frames").mkdir(parents=True)
    (root / "scenes" / "frames" / "scene_00001.jpg").write_bytes(b"jpg1")
    (root / "scenes" / "frames" / "scene_00002.jpg").write_bytes(b"jpg2")

    _csv(root / "faces" / "face_manifest.csv", ["face_id", "scene", "face_crop", "x", "y", "width", "height", "confidence"],
         [[1, "scene_00001.jpg", "faces/crops/face_000001.jpg", 10, 20, 30, 40, 0.95],
          [2, "scene_00002.jpg", "faces/crops/face_000002.jpg", 11, 21, 31, 41, 0.91]])
    (root / "faces" / "crops").mkdir(parents=True)
    (root / "faces" / "crops" / "face_000001.jpg").write_bytes(b"c1")
    (root / "faces" / "crops" / "face_000002.jpg").write_bytes(b"c2")
    (root / "faces" / "face_embeddings.npz").write_bytes(b"npz")
    _csv(root / "faces" / "embedding_manifest.csv", ["face_id", "scene", "face_crop", "embedding_index"],
         [[1, "scene_00001.jpg", "faces/crops/face_000001.jpg", 0], [2, "scene_00002.jpg", "faces/crops/face_000002.jpg", 1]])
    _csv(root / "faces" / "clustered_faces.csv", ["face_id", "scene", "face_crop", "cluster_id"],
         [[1, "scene_00001.jpg", "faces/crops/face_000001.jpg", "CHAR_0001"], [2, "scene_00002.jpg", "faces/crops/face_000002.jpg", "CHAR_0001"]])
    _csv(root / "faces" / "cluster_summary.csv", ["cluster_id", "face_count", "scene_count", "first_scene", "last_scene"],
         [["CHAR_0001", 2, 2, 1, 2], ["UNASSIGNED", 0, 0, 0, 0]])
    for name in ("cluster_audit.csv", "cluster_representatives.csv", "clustering_evaluation.csv"):
        (root / "faces" / name).write_text("h\n")

    (root / "characters").mkdir()
    database = {"characters": [{"character_id": "CHAR_0001", "status": "provisional", "face_count": 2, "scene_count": 2,
                                "first_scene": 1, "last_scene": 2, "mean_similarity": 0.8, "p10_similarity": 0.7,
                                "scene_ids": [1, 2],
                                "representative_faces": [{"face_id": 1, "scene": "scene_00001.jpg",
                                                          "image": "faces/crops/face_000001.jpg", "similarity": 0.9}]}]}
    (root / "characters" / "character_database.json").write_text(json.dumps(database))
    (root / "characters" / "character_scene_mapping.csv").write_text("scene_id,character_id,face_id\n1,CHAR_0001,1\n")
    refined = {"speakers": {"SPEAKER_00": {"total_speech_seconds": 2.1, "candidates": [
        {"character_id": "CHAR_0001", "rank": 1, "evidence_level": "weak_candidate", "overlap_seconds": 2.0,
         "single_character_seconds": 2.0, "multi_character_seconds": 0.0, "unique_scenes": 1, "unique_segments": 1,
         "ambiguous_percentage": 0.0}]}}}
    (root / "characters" / "speaker_character_refined.json").write_text(json.dumps(refined))
    for name in ("speaker_character_refined.csv", "speaker_character_candidates.json", "speaker_character_candidates.csv"):
        (root / "characters" / name).write_text("{}" if name.endswith(".json") else "h\n")

    # Timeline segment_id is the original Whisper id (id 2 = invalid timestamp, dropped from the timeline).
    timeline = {"segments": [
        {"segment_id": 0, "start": 0.0, "end": 2.0, "chinese_text": "你好", "speaker_id": "SPEAKER_00",
         "speaker_overlap_ratio": 1.0, "speaker_confidence": "strong", "scene_id": "1", "scene_confidence": "strong",
         "overlapping_scene_ids": ["1"], "outside_audio": False},
        {"segment_id": 3, "start": 4.0, "end": 6.0, "chinese_text": "再见", "speaker_id": "SPEAKER_01",
         "speaker_overlap_ratio": 1.0, "speaker_confidence": "strong", "scene_id": "2", "scene_confidence": "strong",
         "overlapping_scene_ids": ["2"], "outside_audio": False},
    ]}
    (root / "timeline" / "audit").mkdir(parents=True)
    (root / "timeline" / "master_timeline.json").write_text(json.dumps(timeline, ensure_ascii=False))
    (root / "timeline" / "master_timeline.csv").write_text("h\n")
    (root / "timeline" / "timeline_quality_report.json").write_text("{}")
    (root / "timeline" / "audit" / "audit_summary.json").write_text("{}")
    return root


LEGACY_ANALYSIS = {
    "audio": {"stream_index": 0, "sample_rate": 16000, "channels": 1, "codec": "pcm_s16le"},
    "asr": {"model": "mlx-whisper-large-v3-turbo", "backend": "mlx_whisper", "language": "zh",
            "condition_on_previous_text": False},
    "transcript_filters": {"drop_patterns": ["优优独播剧场", "YoYo Television Series Exclusive"]},
    "diarization": {"engine": "pyannote", "model": "pyannote-speaker-diarization-community-1"},
    "shots": {"detector": "pyscenedetect_content", "threshold": 27.0, "min_scene_len_frames": 15},
    "frames": {"strategy": "midpoint", "fps": None, "width": 480, "jpeg_quality": 80},
    "face_detection": {"model": "opencv-yunet-2023mar", "confidence_threshold": 0.75, "nms_threshold": 0.3,
                       "top_k": 5000, "padding": 0.35, "crop_size": 256, "jpeg_quality": 90},
    "face_embedding": {"model": "insightface-buffalo-l", "det_size": 320,
                       "execution_provider": "CPUExecutionProvider", "face_selection": "largest"},
    "clustering": {"algorithm": "dbscan", "eps": 0.35, "min_samples": 2, "metric": "cosine"},
    "character_database": {"representatives_per_cluster": 5},
    "alignment": {"strong_ratio": 0.7, "moderate_ratio": 0.3},
}


@pytest.fixture
def legacy_project(tmp_path, monkeypatch, legacy_workspace):
    """A project whose config equals project_001's analysis settings, without a source yet."""
    monkeypatch.setenv("VP_PLATFORM_ROOT", str(REPO_ROOT))
    data = yaml.safe_load(PROJECT_001_CONFIG.read_text(encoding="utf-8"))
    data["project"]["id"] = "legacy_test"
    data["analysis"] = LEGACY_ANALYSIS  # the settings the legacy scripts used
    project_dir = tmp_path / "projects" / "legacy_test"
    project_dir.mkdir(parents=True)
    return write_yaml(project_dir / "config.yaml", data)


def copy_tree(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst)
