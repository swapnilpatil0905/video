"""End-to-end legacy migration on a miniature legacy workspace (same formats as the real one)."""

import json
import os

from video_platform.cli import main
from video_platform.config.loader import load_project_config
from video_platform.migration.legacy import LegacyImporter
from video_platform.models.registry import ModelRegistry
from video_platform.orchestration.graph import load_pipeline
from video_platform.orchestration.planner import Planner
from video_platform.storage.index import ArtifactIndex
from video_platform.storage.store import LocalArtifactStore

from ..conftest import REPO_ROOT


def _importer(config_path, legacy_root, probe, overrides=()):
    loaded = load_project_config(config_path, overrides=list(overrides))
    graph = load_pipeline(REPO_ROOT / "configs" / "pipeline.yaml")
    registry = ModelRegistry.load(REPO_ROOT / "model_registry")
    index = ArtifactIndex(loaded.state_dir / "artifacts.sqlite")
    planner = Planner(graph, loaded, registry, index)
    store = LocalArtifactStore(loaded.artifact_root)
    return LegacyImporter(loaded, graph, planner, index, store, legacy_root, probe=probe), planner, index, loaded


def test_import_makes_analysis_a_cache_hit(legacy_project, legacy_workspace, fake_probe):
    importer, planner, index, loaded = _importer(legacy_project, legacy_workspace, fake_probe)
    report = importer.import_all()
    statuses = {s.stage: s.status for s in report.stages}
    assert set(statuses.values()) == {"imported"}, report.as_dict()
    assert report.manifest_unchanged is None  # first manifest created

    # Re-planning: everything cached → nothing would re-run.
    assert {i.status for i in planner.plan(["speaker_mapping_prior"])} == {"cached"}

    # Source adopted by hardlink; legacy file untouched.
    legacy_src = legacy_workspace / "input" / "source.mp4"
    assert os.stat(loaded.source_video_path).st_ino == os.stat(legacy_src).st_ino
    assert legacy_src.read_bytes() == b"legacy source video"

    # Normalised outputs use platform ids.
    asr = index.find_valid(planner.expected_ids()["asr"])
    transcript = json.loads(importer.store.read_bytes(asr.outputs[0].uri.removeprefix("local://")))
    assert [s["segment_id"] for s in transcript] == ["SEG_000000", "SEG_000001", "SEG_000002", "SEG_000003"]
    assert "invalid_timestamp" in transcript[2]["flags"]

    align = index.find_valid(planner.expected_ids()["alignment"])
    segments = json.loads(importer.store.read_bytes(align.outputs[0].uri.removeprefix("local://")))
    # timeline segment_id 2 → cleaned index 2 → raw id 3
    assert [s["segment_id"] for s in segments] == ["SEG_000000", "SEG_000003"]
    assert segments[0]["speaker_id"] == "SPK_00" and segments[1]["shot_id"] == "SH_00002"

    prior = index.find_valid(planner.expected_ids()["speaker_mapping_prior"])
    mappings = json.loads(importer.store.read_bytes(prior.outputs[0].uri.removeprefix("local://")))
    assert mappings[0]["status"] == "uncertain" and mappings[0]["character_id"] == "SRC_CHAR_0001"

    diar = index.find_valid(planner.expected_ids()["diarization"])
    assert diar.processing_seconds == 12.5
    assert diar.provenance["imported_from"] == "legacy_workspace"


def test_import_is_idempotent_and_manifest_verified(legacy_project, legacy_workspace, fake_probe):
    _importer(legacy_project, legacy_workspace, fake_probe)[0].import_all()
    importer, *_ = _importer(legacy_project, legacy_workspace, fake_probe)
    report = importer.import_all()
    assert {s.status for s in report.stages} == {"already_cached"}
    assert report.manifest_unchanged is True


def test_param_mismatch_skips_stage_and_descendants(legacy_project, legacy_workspace, fake_probe):
    importer, planner, _, _ = _importer(legacy_project, legacy_workspace, fake_probe, ["analysis.clustering.eps=0.4"])
    statuses = {s.stage: s.status for s in importer.import_all().stages}
    assert statuses["face_embedding"] == "imported"
    assert statuses["face_clustering"] == "skipped_param_mismatch"
    assert statuses["character_database"] == "skipped_upstream"
    assert statuses["speaker_mapping_prior"] == "skipped_upstream"
    status = {i.stage: i.status for i in planner.plan()}
    assert status["asr"] == "cached" and status["face_clustering"] == "pending"


def test_cli_validate_status_and_mode_analyze(legacy_project, legacy_workspace, fake_probe, monkeypatch, capsys):
    _importer(legacy_project, legacy_workspace, fake_probe)[0].import_all()
    assert main(["--project", str(legacy_project), "--validate-only"]) == 0
    assert main(["--project", str(legacy_project), "--stage", "speaker_mapping_prior"]) == 0
    assert "Nothing to do" in capsys.readouterr().out
    assert main(["--project", str(legacy_project), "--plan", "--mode", "full"]) == 0
    capsys.readouterr()
    assert main(["--project", str(legacy_project), "--approve-sample"]) == 3  # no sample yet
    assert main(["--project", str(legacy_project), "--status", "--json"]) == 0
    status = {i["stage"]: i["status"] for i in json.loads(capsys.readouterr().out)["plan"]}
    for legacy_stage in ("asr", "diarization", "shot_detection", "face_clustering", "alignment"):
        assert status[legacy_stage] == "cached"
    assert status["sample_production"] == "pending"


def test_cli_init_creates_valid_project(tmp_path, monkeypatch):
    monkeypatch.setenv("VP_PLATFORM_ROOT", str(REPO_ROOT))
    target = tmp_path / "projects" / "project_002"
    assert main(["--init", str(target)]) == 0
    assert main(["--project", str(target / "config.yaml"), "--validate-only"]) == 2  # no source video yet
    (target / "input" / "source.mp4").write_bytes(b"v")
    assert main(["--project", str(target / "config.yaml"), "--validate-only"]) == 0
    assert main(["--init", str(target)]) == 3  # refuses to overwrite


def test_cli_config_error_exit_code(tmp_path):
    bad = tmp_path / "config.yaml"
    bad.write_text("project: [not, a, mapping]\n")
    assert main(["--project", str(bad), "--validate-only"]) == 2
