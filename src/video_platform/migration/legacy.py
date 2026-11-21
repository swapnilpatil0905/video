"""Import the pre-platform (legacy) analysis workspace as cached platform artifacts.

The legacy scripts in ``pipeline/`` produced valid analysis results with hardcoded
parameters. This importer:

1. Writes a sha256 manifest of every legacy file (read-only pass).
2. For each analysis stage whose *configured* parameters equal the legacy parameters,
   adopts the legacy files (hardlinked, no extra disk) under the artifact id the planner
   expects, converts them to versioned platform schemas, and records provenance.
3. Never modifies or deletes legacy files.

Because the imported artifacts carry exactly the ids the planner computes, re-running
analysis afterwards is a cache hit for transcription, diarization, shot detection, etc.
If a configured parameter differs from the legacy run, that stage (and everything
downstream) is *not* imported and will be planned for re-execution instead.
"""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config.loader import LoadedConfig
from ..core.errors import ConfigError, StageFailure
from ..core.hashing import sha256_bytes, sha256_file
from ..core.log import get_logger
from ..ingestion.probe import ProbeFn, ffprobe_json, probe_video
from ..orchestration.graph import StageGraph
from ..orchestration.planner import Planner
from ..schemas.audio import AudioSegment, SpeakerTurn, TranscriptSegment, segment_id, speaker_id
from ..schemas.characters import (
    BoundingBox,
    Evidence,
    FaceDetection,
    RepresentativeFace,
    SourceCharacter,
    SpeakerCharacterMapping,
    source_character_id,
)
from ..schemas.common import ArtifactRecord, ArtifactRef, Metric
from ..schemas.media import Keyframe, Shot, shot_id
from ..storage.index import ArtifactIndex
from ..storage.store import ArtifactStore, LocalArtifactStore

log = get_logger(__name__)

# Directories of the legacy workspace covered by the manifest (code included for provenance).
LEGACY_DIRS = (
    "input", "audio", "transcripts", "diarization", "scenes", "faces",
    "characters", "timeline", "verification", "models", "pipeline",
)
_IGNORED_NAMES = {".DS_Store", "__pycache__"}

# Parameters actually used by the legacy scripts (copied from their constants).
# The configured values must equal these for a stage to be adopted.
LEGACY_PARAMS: dict[str, dict[str, Any]] = {
    "extract_audio": {"stream_index": 0, "sample_rate": 16000, "channels": 1, "codec": "pcm_s16le"},
    "asr": {
        "model": "mlx-whisper-large-v3-turbo",
        "backend": "mlx_whisper",
        "language": "zh",
        "condition_on_previous_text": False,
    },
    "transcript_filter": {"drop_patterns": ["优优独播剧场", "YoYo Television Series Exclusive"]},
    "diarization": {"model": "pyannote-speaker-diarization-community-1", "min_speakers": None, "max_speakers": None},
    "shot_detection": {"detector": "pyscenedetect_content", "threshold": 27.0, "min_scene_len_frames": 15},
    "frame_sampling": {"strategy": "midpoint", "fps": None, "width": 480, "jpeg_quality": 80},
    "face_detection": {
        "model": "opencv-yunet-2023mar",
        "confidence_threshold": 0.75,
        "nms_threshold": 0.3,
        "top_k": 5000,
        "padding": 0.35,
        "crop_size": 256,
        "jpeg_quality": 90,
    },
    "face_embedding": {
        "model": "insightface-buffalo-l",
        "det_size": 320,
        "execution_provider": "CPUExecutionProvider",
        "face_selection": "largest",
    },
    "face_clustering": {"algorithm": "dbscan", "eps": 0.35, "min_samples": 2, "metric": "cosine"},
    "character_database": {"representatives_per_cluster": 5},
    "alignment": {"strong_ratio": 0.7, "moderate_ratio": 0.3},
    "speaker_mapping_prior": {
        "confidence_weights": {"strong": 1.0, "moderate": 0.7, "weak": 0.35, "unknown": 0.5},
        "stronger_candidate": {
            "min_single_character_seconds": 30,
            "min_single_character_percentage": 50,
            "min_unique_scenes": 10,
        },
        "moderate_candidate": {
            "min_single_character_seconds": 10,
            "min_single_character_percentage": 0,
            "min_unique_scenes": 5,
        },
    },
}

# Legacy files adopted per stage (relative to the legacy root). Trailing "/" = directory.
LEGACY_FILES: dict[str, list[str]] = {
    "ingest": [],
    "extract_audio": ["audio/source.wav"],
    "asr": ["transcripts/chinese_transcript.json"],
    "transcript_filter": ["transcripts/chinese_transcript_clean.json"],
    "diarization": ["diarization/speaker_turns.json", "diarization/speaker_turns.csv", "diarization/speaker_summary.json"],
    "shot_detection": ["scenes/scene_timeline.csv"],
    "frame_sampling": ["scenes/frame_manifest.csv", "scenes/frames/"],
    "face_detection": ["faces/face_manifest.csv", "faces/crops/"],
    "face_embedding": ["faces/face_embeddings.npz", "faces/embedding_manifest.csv"],
    "face_clustering": [
        "faces/clustered_faces.csv",
        "faces/cluster_summary.csv",
        "faces/cluster_audit.csv",
        "faces/cluster_representatives.csv",
        "faces/clustering_evaluation.csv",
    ],
    "character_database": ["characters/character_database.json", "characters/character_scene_mapping.csv"],
    "alignment": [
        "timeline/master_timeline.json",
        "timeline/master_timeline.csv",
        "timeline/timeline_quality_report.json",
        "timeline/audit/",
    ],
    "speaker_mapping_prior": [
        "characters/speaker_character_refined.json",
        "characters/speaker_character_refined.csv",
        "characters/speaker_character_candidates.json",
        "characters/speaker_character_candidates.csv",
    ],
}

LEGACY_SCRIPTS: dict[str, str] = {
    "extract_audio": "pipeline/extract_audio.py",
    "asr": "pipeline/transcribe.py",
    "transcript_filter": "pipeline/clean_transcript.py",
    "diarization": "pipeline/diarize_speakers_2.py",
    "shot_detection": "pipeline/detect_scenes.py",
    "frame_sampling": "pipeline/extract_scene_frames.py",
    "face_detection": "pipeline/detect_faces.py",
    "face_embedding": "pipeline/extract_face_embeddings.py",
    "face_clustering": "pipeline/cluster_characters.py + pipeline/audit_character_clusters.py",
    "character_database": "pipeline/build_character_database.py",
    "alignment": "pipeline/build_master_timeline.py",
    "speaker_mapping_prior": "pipeline/refine_speaker_character_matching.py",
}

# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def iter_legacy_files(root: Path) -> Iterator[Path]:
    for name in LEGACY_DIRS:
        base = root / name
        if not base.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in _IGNORED_NAMES)
            for filename in sorted(filenames):
                if filename not in _IGNORED_NAMES and not filename.endswith(".pyc"):
                    yield Path(dirpath) / filename


def build_manifest(root: Path, index: ArtifactIndex | None = None) -> dict[str, Any]:
    files = {}
    for path in iter_legacy_files(root):
        rel = path.relative_to(root).as_posix()
        digest = index.file_sha256(path) if index else sha256_file(path)
        files[rel] = {"sha256": digest, "size": path.stat().st_size}
    return {
        "schema_version": 1,
        "legacy_root": str(root),
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "file_count": len(files),
        "total_bytes": sum(f["size"] for f in files.values()),
        "files": files,
    }


def compare_manifests(expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, list[str]]:
    exp, act = expected["files"], actual["files"]
    return {
        "missing": sorted(set(exp) - set(act)),
        "added": sorted(set(act) - set(exp)),
        "changed": sorted(k for k in set(exp) & set(act) if exp[k]["sha256"] != act[k]["sha256"]),
    }


# ---------------------------------------------------------------------------
# Converters: legacy formats → platform schemas
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _scene_number(frame_name: str) -> int:
    """'scene_00042.jpg' → 42."""
    return int(Path(frame_name).stem.split("_")[-1])


def _shot_ref(legacy_scene: str | int) -> str:
    return shot_id(int(legacy_scene))


def convert_transcript(raw: dict[str, Any], language: str | None) -> list[TranscriptSegment]:
    lang = raw.get("language") or language
    segments = []
    for seg in raw.get("segments", []):
        start, end = float(seg["start"]), float(seg["end"])
        segments.append(
            TranscriptSegment(
                segment_id=segment_id(int(seg["id"])),
                start_s=start,
                end_s=end,
                text=str(seg.get("text", "")).strip(),
                language=lang,
                avg_logprob=seg.get("avg_logprob"),
                no_speech_prob=seg.get("no_speech_prob"),
                compression_ratio=seg.get("compression_ratio"),
                flags=["invalid_timestamp"] if end <= start else [],
            )
        )
    return segments


def convert_speaker_turns(data: dict[str, Any]) -> list[SpeakerTurn]:
    return [
        SpeakerTurn(start_s=float(t["start"]), end_s=float(t["end"]), speaker_id=speaker_id(str(t["speaker_id"])))
        for t in data.get("turns", [])
    ]


def convert_shots(rows: list[dict[str, str]]) -> list[Shot]:
    return [
        Shot(
            shot_id=_shot_ref(row["scene_id"]),
            start_s=float(row["start_seconds"]),
            end_s=float(row["end_seconds"]),
            legacy_scene_number=int(row["scene_id"]),
        )
        for row in rows
    ]


def convert_keyframes(rows: list[dict[str, str]], uri_for: dict[str, str]) -> list[Keyframe]:
    return [
        Keyframe(
            shot_id=_shot_ref(row["scene_id"]),
            time_s=float(row["midpoint_seconds"]),
            uri=uri_for.get(row["frame_path"], row["frame_path"]),
        )
        for row in rows
    ]


def convert_face_detections(rows: list[dict[str, str]], uri_for: dict[str, str]) -> list[FaceDetection]:
    return [
        FaceDetection(
            face_id=f"FACE_{int(row['face_id']):06d}",
            shot_id=_shot_ref(_scene_number(row["scene"])),
            bbox=BoundingBox(
                x=float(row["x"]), y=float(row["y"]), width=float(row["width"]), height=float(row["height"])
            ),
            confidence=float(row["confidence"]),
            crop_uri=uri_for.get(row["face_crop"], row["face_crop"]),
        )
        for row in rows
    ]


def convert_characters(database: dict[str, Any], uri_for: dict[str, str]) -> list[SourceCharacter]:
    characters = []
    for item in database.get("characters", []):
        shots = [shot_id(int(s)) for s in item.get("scene_ids", [])]
        characters.append(
            SourceCharacter(
                character_id=source_character_id(item["character_id"]),
                legacy_ids=[item["character_id"]],
                status="provisional",
                face_count=int(item.get("face_count", 0)),
                shot_ids=shots,
                first_shot_id=shot_id(int(item["first_scene"])) if item.get("first_scene") is not None else None,
                last_shot_id=shot_id(int(item["last_scene"])) if item.get("last_scene") is not None else None,
                mean_similarity=item.get("mean_similarity"),
                p10_similarity=item.get("p10_similarity"),
                representative_faces=[
                    RepresentativeFace(
                        face_id=f"FACE_{int(face['face_id']):06d}",
                        shot_id=_shot_ref(_scene_number(face["scene"])),
                        uri=uri_for.get(face.get("image", ""), face.get("image")),
                        centroid_similarity=face.get("similarity"),
                    )
                    for face in item.get("representative_faces", [])
                ],
            )
        )
    return characters


def convert_audio_segments(
    timeline: dict[str, Any], clean_transcript: dict[str, Any], language: str | None
) -> list[AudioSegment]:
    """Legacy timeline ``segment_id`` is an index into the cleaned transcript list."""
    clean = clean_transcript.get("segments", [])
    segments = []
    for item in timeline.get("segments", []):
        index = int(item["segment_id"])
        if not 0 <= index < len(clean):
            raise StageFailure(f"timeline segment_id {index} has no matching cleaned transcript segment")
        flags = ["outside_audio"] if item.get("outside_audio") else []
        raw_speaker = item.get("speaker_id")
        segments.append(
            AudioSegment(
                segment_id=segment_id(int(clean[index]["id"])),
                start_s=float(item["start"]),
                end_s=float(item["end"]),
                text=str(item.get("chinese_text") or item.get("text") or "").strip(),
                language=clean_transcript.get("language") or language,
                speaker_id=speaker_id(raw_speaker) if raw_speaker else None,
                speaker_overlap_ratio=item.get("speaker_overlap_ratio"),
                speaker_match=item.get("speaker_confidence") or "none",
                shot_id=_shot_ref(item["scene_id"]) if item.get("scene_id") else None,
                shot_match=item.get("scene_confidence") or "none",
                overlapping_shot_ids=[_shot_ref(s) for s in item.get("overlapping_scene_ids", [])],
                flags=flags,
            )
        )
    return segments


def convert_speaker_prior(refined: dict[str, Any]) -> list[SpeakerCharacterMapping]:
    """Co-occurrence candidates become *uncertain* mappings with explicit reasons."""
    mappings = []
    for raw_speaker, data in sorted(refined.get("speakers", {}).items()):
        for candidate in data.get("candidates", []):
            mappings.append(
                SpeakerCharacterMapping(
                    speaker_id=speaker_id(raw_speaker),
                    character_id=source_character_id(candidate["character_id"]),
                    status="uncertain",
                    confidence=None,
                    rank=candidate.get("rank"),
                    supporting_shots=int(candidate.get("unique_scenes", 0)),
                    supporting_segments=int(candidate.get("unique_segments", 0)),
                    supporting_speech_s=float(candidate.get("single_character_seconds", 0.0)),
                    evidence=[
                        Evidence(
                            kind="cooccurrence",
                            detail={
                                "evidence_level": candidate.get("evidence_level"),
                                "overlap_seconds": candidate.get("overlap_seconds"),
                                "single_character_seconds": candidate.get("single_character_seconds"),
                                "multi_character_seconds": candidate.get("multi_character_seconds"),
                                "ambiguous_percentage": candidate.get("ambiguous_percentage"),
                            },
                        )
                    ],
                    uncertainty_reason=(
                        "co-occurrence only: visible during speech is not proof of speaking; "
                        "active speaker detection required (M2)"
                    ),
                )
            )
    return mappings


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


@dataclass
class StageImport:
    stage: str
    artifact_id: str
    status: str  # imported | already_cached | skipped_param_mismatch | skipped_upstream | skipped_missing_files
    detail: str = ""


@dataclass
class MigrationReport:
    project_id: str
    legacy_root: str
    manifest_path: str
    manifest_file_count: int
    manifest_unchanged: bool | None
    stages: list[StageImport] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "legacy_root": self.legacy_root,
            "manifest_path": self.manifest_path,
            "manifest_file_count": self.manifest_file_count,
            "manifest_unchanged": self.manifest_unchanged,
            "stages": [s.__dict__ for s in self.stages],
        }


def _expand(root: Path, entries: list[str]) -> list[Path]:
    files: list[Path] = []
    for entry in entries:
        path = root / entry
        if entry.endswith("/"):
            if path.is_dir():
                files.extend(sorted(p for p in path.rglob("*") if p.is_file() and p.name not in _IGNORED_NAMES))
        elif path.is_file():
            files.append(path)
    return files


def _diff(configured: Any, legacy: Any, prefix: str = "") -> list[str]:
    if isinstance(configured, dict) and isinstance(legacy, dict):
        out = []
        for key in sorted(set(configured) | set(legacy)):
            out += _diff(configured.get(key), legacy.get(key), f"{prefix}.{key}" if prefix else key)
        return out
    return [] if configured == legacy else [f"{prefix}: configured={configured!r} legacy={legacy!r}"]


class LegacyImporter:
    def __init__(
        self,
        loaded: LoadedConfig,
        graph: StageGraph,
        planner: Planner,
        index: ArtifactIndex,
        store: ArtifactStore,
        legacy_root: Path,
        *,
        probe: ProbeFn = ffprobe_json,
    ) -> None:
        self.loaded = loaded
        self.graph = graph
        self.planner = planner
        self.index = index
        self.store = store
        self.root = Path(legacy_root).resolve()
        self.probe = probe
        self.language = loaded.config.languages.source
        self._manifest: dict[str, Any] | None = None

    # -- manifest -------------------------------------------------------------

    @property
    def manifest_path(self) -> Path:
        return self.loaded.project_dir / "migration" / "manifest_v0.json"

    def write_or_verify_manifest(self) -> tuple[dict[str, Any], bool | None]:
        current = build_manifest(self.root, self.index)
        path = self.manifest_path
        if path.exists():
            previous = _read_json(path)
            diff = compare_manifests(previous, current)
            unchanged = not any(diff.values())
            if not unchanged:
                log.warning("legacy workspace changed since the first manifest: %s", {k: v[:5] for k, v in diff.items()})
            self._manifest = previous
            return previous, unchanged
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(current, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        self._manifest = current
        return current, None

    def _sha(self, path: Path) -> str:
        rel = path.relative_to(self.root).as_posix()
        if self._manifest and rel in self._manifest["files"]:
            return self._manifest["files"][rel]["sha256"]
        return self.index.file_sha256(path)

    # -- source video -----------------------------------------------------------

    def ensure_project_source(self) -> Path:
        """Hardlink legacy input/source.mp4 to the project's configured source path."""
        target = self.loaded.source_video_path
        legacy_source = self.root / "input" / "source.mp4"
        if target.exists():
            return target
        if not legacy_source.exists():
            raise ConfigError(f"neither {target} nor legacy {legacy_source} exists")
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(legacy_source, target)
        except OSError as exc:
            raise ConfigError(
                f"could not hardlink {legacy_source} → {target} ({exc}). Copy it manually (never modify the original)."
            ) from exc
        return target

    # -- per stage ----------------------------------------------------------------

    def _adopt_files(self, stage: str, artifact_id: str) -> tuple[list[ArtifactRef], dict[str, str]]:
        refs: list[ArtifactRef] = []
        uri_for: dict[str, str] = {}
        for path in _expand(self.root, LEGACY_FILES[stage]):
            rel = path.relative_to(self.root).as_posix()
            key = f"{stage}/{artifact_id}/legacy/{rel}"
            uri = self.store.put_file(path, key, link=True)
            uri_for[rel] = uri
            refs.append(ArtifactRef(artifact_id=artifact_id, uri=uri, sha256=self._sha(path), role="legacy_raw"))
        return refs, uri_for

    def _write_json(self, stage: str, artifact_id: str, name: str, payload: Any) -> ArtifactRef:
        data = json.dumps(payload, ensure_ascii=False, indent=1, default=str).encode("utf-8")
        key = f"{stage}/{artifact_id}/{name}"
        uri = self.store.put_bytes(data, key)
        return ArtifactRef(artifact_id=artifact_id, uri=uri, sha256=sha256_bytes(data), media_type="application/json", role="normalized")

    @staticmethod
    def _dump(models: list[Any]) -> list[dict[str, Any]]:
        return [m.model_dump(mode="json", exclude_none=True) for m in models]

    def _normalize(self, stage: str, artifact_id: str, uri_for: dict[str, str]) -> tuple[list[ArtifactRef], dict[str, Metric], dict[str, Any]]:
        root = self.root
        outputs: list[ArtifactRef] = []
        metrics: dict[str, Metric] = {}
        extra: dict[str, Any] = {}

        if stage == "ingest":
            source = self.loaded.source_video_path
            meta = probe_video(source, sha256=self.planner.source_hash(), probe=self.probe)
            outputs.append(self._write_json(stage, artifact_id, "video_metadata.json", meta.model_dump(mode="json")))
            metrics["duration_s"] = Metric(value=meta.duration_s, unit="s")
            metrics["resolution"] = Metric(value=f"{meta.width}x{meta.height}")
            metrics["fps"] = Metric(value=round(meta.fps, 3), unit="fps")
        elif stage == "asr":
            segs = convert_transcript(_read_json(root / "transcripts/chinese_transcript.json"), self.language)
            outputs.append(self._write_json(stage, artifact_id, "transcript.json", self._dump(segs)))
            metrics["segment_count"] = Metric(value=len(segs))
            metrics["invalid_timestamps"] = Metric(value=sum("invalid_timestamp" in s.flags for s in segs))
        elif stage == "transcript_filter":
            raw = _read_json(root / "transcripts/chinese_transcript.json")
            clean = _read_json(root / "transcripts/chinese_transcript_clean.json")
            segs = convert_transcript(clean, self.language)
            removed = sorted(
                {int(s["id"]) for s in raw.get("segments", [])} - {int(s["id"]) for s in clean.get("segments", [])}
            )
            payload = {"segments": self._dump(segs), "removed_segment_ids": [segment_id(i) for i in removed]}
            outputs.append(self._write_json(stage, artifact_id, "transcript_clean.json", payload))
            metrics["segment_count"] = Metric(value=len(segs))
            metrics["removed_by_filter"] = Metric(value=len(removed))
        elif stage == "diarization":
            data = _read_json(root / "diarization/speaker_turns.json")
            turns = convert_speaker_turns(data)
            outputs.append(self._write_json(stage, artifact_id, "speaker_turns.json", self._dump(turns)))
            metrics["turn_count"] = Metric(value=len(turns))
            metrics["speaker_count"] = Metric(value=len({t.speaker_id for t in turns}))
            summary_path = root / "diarization/speaker_summary.json"
            if summary_path.exists():
                extra["processing_seconds"] = _read_json(summary_path).get("processing_seconds")
        elif stage == "shot_detection":
            shots = convert_shots(_read_csv(root / "scenes/scene_timeline.csv"))
            outputs.append(self._write_json(stage, artifact_id, "shots.json", self._dump(shots)))
            metrics["shot_count"] = Metric(value=len(shots))
        elif stage == "frame_sampling":
            frames = convert_keyframes(_read_csv(root / "scenes/frame_manifest.csv"), uri_for)
            outputs.append(self._write_json(stage, artifact_id, "keyframes.json", self._dump(frames)))
            metrics["keyframe_count"] = Metric(value=len(frames))
        elif stage == "face_detection":
            faces = convert_face_detections(_read_csv(root / "faces/face_manifest.csv"), uri_for)
            outputs.append(self._write_json(stage, artifact_id, "face_detections.json", self._dump(faces)))
            metrics["face_count"] = Metric(value=len(faces))
        elif stage == "face_embedding":
            rows = _read_csv(root / "faces/embedding_manifest.csv")
            metrics["embedding_count"] = Metric(value=len(rows))
        elif stage == "face_clustering":
            rows = _read_csv(root / "faces/cluster_summary.csv")
            metrics["cluster_count"] = Metric(value=sum(r["cluster_id"] != "UNASSIGNED" for r in rows))
            unassigned = next((r for r in rows if r["cluster_id"] == "UNASSIGNED"), None)
            metrics["unassigned_faces"] = Metric(value=int(unassigned["face_count"]) if unassigned else 0)
        elif stage == "character_database":
            characters = convert_characters(_read_json(root / "characters/character_database.json"), uri_for)
            aliases = {c.legacy_ids[0]: c.character_id for c in characters}
            outputs.append(self._write_json(stage, artifact_id, "source_characters.json", self._dump(characters)))
            outputs.append(self._write_json(stage, artifact_id, "legacy_character_aliases.json", aliases))
            metrics["character_count"] = Metric(value=len(characters))
        elif stage == "alignment":
            segs = convert_audio_segments(
                _read_json(root / "timeline/master_timeline.json"),
                _read_json(root / "transcripts/chinese_transcript_clean.json"),
                self.language,
            )
            outputs.append(self._write_json(stage, artifact_id, "audio_segments.json", self._dump(segs)))
            metrics["segment_count"] = Metric(value=len(segs))
            for level in ("strong", "moderate", "weak", "none"):
                metrics[f"speaker_match_{level}"] = Metric(value=sum(s.speaker_match == level for s in segs))
        elif stage == "speaker_mapping_prior":
            mappings = convert_speaker_prior(_read_json(root / "characters/speaker_character_refined.json"))
            outputs.append(self._write_json(stage, artifact_id, "speaker_character_prior.json", self._dump(mappings)))
            metrics["candidate_count"] = Metric(value=len(mappings))
            metrics["accepted_mappings"] = Metric(value=0, note="co-occurrence alone never yields 'accepted'")
        return outputs, metrics, extra

    def import_all(self) -> MigrationReport:
        manifest, unchanged = self.write_or_verify_manifest()
        self.ensure_project_source()
        expected = self.planner.expected_ids()
        report = MigrationReport(
            project_id=self.loaded.project_id,
            legacy_root=str(self.root),
            manifest_path=str(self.manifest_path),
            manifest_file_count=manifest["file_count"],
            manifest_unchanged=unchanged,
        )
        not_imported: set[str] = set()

        for stage in self.graph.order:
            if stage not in LEGACY_FILES:
                continue
            spec = self.graph.get(stage)
            artifact_id = expected[stage]
            blocked = [d for d in spec.depends_on if d in not_imported]
            if blocked:
                not_imported.add(stage)
                report.stages.append(StageImport(stage, artifact_id, "skipped_upstream", f"upstream not imported: {blocked}"))
                continue
            if self.index.find_valid(artifact_id):
                report.stages.append(StageImport(stage, artifact_id, "already_cached"))
                continue
            if stage in LEGACY_PARAMS:
                differences = _diff(self.planner.stage_params(spec), LEGACY_PARAMS[stage])
                if differences:
                    not_imported.add(stage)
                    report.stages.append(
                        StageImport(stage, artifact_id, "skipped_param_mismatch", "; ".join(differences))
                    )
                    continue
            missing = [e for e in LEGACY_FILES[stage] if not (self.root / e.rstrip("/")).exists()]
            if missing:
                not_imported.add(stage)
                report.stages.append(StageImport(stage, artifact_id, "skipped_missing_files", f"missing: {missing}"))
                continue

            raw_refs, uri_for = self._adopt_files(stage, artifact_id)
            normalized, metrics, extra = self._normalize(stage, artifact_id, uri_for)
            model_id = (self.planner.model_ids(spec) or [None])[0]
            inputs = [ArtifactRef(artifact_id=expected[d], role="input") for d in spec.depends_on]
            if not spec.depends_on:
                inputs = [ArtifactRef(artifact_id=f"src:{self.planner.source_hash()}", role="source")]
            record = ArtifactRecord(
                project_id=self.loaded.project_id,
                stage=stage,
                stage_version=spec.version,
                artifact_id=artifact_id,
                model_id=model_id,
                model_fingerprint=self.planner.model_fingerprint(spec),
                inputs=inputs,
                outputs=normalized + raw_refs,
                params=self.planner.stage_params(spec),
                params_hash=self.planner.stage_identity(spec, [expected[d] for d in spec.depends_on])[1],
                processing_seconds=extra.get("processing_seconds"),
                metrics=metrics,
                status="valid",
                rights_basis="personal_learning_private",
                provenance={
                    "imported_from": "legacy_workspace",
                    "legacy_script": LEGACY_SCRIPTS.get(stage, "ffprobe (import time)"),
                    "legacy_files": [r.uri for r in raw_refs][:50],
                    "legacy_file_count": len(raw_refs),
                    "imported_at": datetime.now(tz=timezone.utc).isoformat(),
                },
            )
            self.index.put(record)
            report.stages.append(StageImport(stage, artifact_id, "imported", f"{len(raw_refs)} legacy files"))
            log.info("imported %s as %s (%d files)", stage, artifact_id, len(raw_refs))
        return report


def default_local_store(loaded: LoadedConfig) -> LocalArtifactStore:
    return LocalArtifactStore(loaded.artifact_root)
