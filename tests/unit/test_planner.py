from pathlib import Path

from video_platform.config.loader import load_project_config
from video_platform.models.registry import ModelRegistry
from video_platform.orchestration.graph import load_pipeline
from video_platform.orchestration.planner import Planner
from video_platform.schemas.common import ArtifactRecord
from video_platform.storage.index import ArtifactIndex

from ..conftest import REPO_ROOT, minimal_config


def _planner(config_path: Path, overrides=()):
    loaded = load_project_config(config_path, overrides=list(overrides))
    graph = load_pipeline(REPO_ROOT / "configs" / "pipeline.yaml")
    registry = ModelRegistry.load(REPO_ROOT / "model_registry")
    index = ArtifactIndex(loaded.state_dir / "artifacts.sqlite")
    return Planner(graph, loaded, registry, index), index, loaded


def _mark_all_valid(planner, index, loaded):
    for stage, artifact_id in planner.expected_ids().items():
        index.put(ArtifactRecord(project_id=loaded.project_id, stage=stage, stage_version="1",
                                 artifact_id=artifact_id, params_hash="x"))


def test_fresh_project_everything_pending(make_project):
    planner, _, _ = _planner(make_project())
    items = planner.plan(["speaker_mapping_prior"])
    assert {i.status for i in items} == {"pending"}
    assert items[0].reason == "never run"


def test_cached_after_artifacts_exist(make_project):
    path = make_project()
    planner, index, loaded = _planner(path)
    _mark_all_valid(planner, index, loaded)
    planner2, _, _ = _planner(path)
    assert {i.status for i in planner2.plan()} == {"cached"}


def test_param_change_invalidates_only_descendants(make_project):
    path = make_project()
    planner, index, loaded = _planner(path)
    _mark_all_valid(planner, index, loaded)
    before = planner.expected_ids()

    changed, _, _ = _planner(path, ["analysis.clustering.eps=0.4"])
    after = changed.expected_ids()
    differing = {stage for stage in before if before[stage] != after[stage]}
    assert differing == {"face_clustering"} | changed.graph.descendants("face_clustering")
    assert "asr" not in differing and "narrative" in differing
    status = {i.stage: i.status for i in changed.plan()}
    assert status["asr"] == "cached" and status["diarization"] == "cached" and status["shot_detection"] == "cached"
    assert status["face_clustering"] == "pending"


def test_transcript_filter_change_does_not_rerun_vision(make_project):
    path = make_project()
    planner, index, loaded = _planner(path)
    _mark_all_valid(planner, index, loaded)
    changed, _, _ = _planner(path, ['analysis.transcript_filters.drop_patterns=["x"]'])
    status = {i.stage: i.status for i in changed.plan()}
    for vision in ("shot_detection", "frame_sampling", "face_detection", "face_embedding", "face_clustering"):
        assert status[vision] == "cached"
    assert status["transcript_filter"] == "pending" and status["alignment"] == "pending"


def test_source_change_invalidates_everything(make_project):
    path = make_project()
    planner, index, loaded = _planner(path)
    _mark_all_valid(planner, index, loaded)
    loaded.source_video_path.write_bytes(b"different source video")
    planner2, _, _ = _planner(path)
    assert {i.status for i in planner2.plan()} == {"pending"}


def test_model_revision_is_part_of_identity(make_project):
    planner, _, _ = _planner(make_project(minimal_config()))
    before = planner.expected_ids()
    record = planner.registry.get("faster-whisper-large-v3")
    planner.registry.records["faster-whisper-large-v3"] = record.model_copy(
        update={"weights": record.weights.model_copy(update={"pinned_revision": "rev2"})}
    )
    planner._expected = None
    after = planner.expected_ids()
    assert before["asr"] != after["asr"] and before["diarization"] == after["diarization"]
