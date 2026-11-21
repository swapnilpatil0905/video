from video_platform.models.registry import ModelRecord, ModelRegistry, check_usage_policy


def test_repo_registry_loads_and_covers_configured_models(repo_root):
    registry = ModelRegistry.load(repo_root / "model_registry")
    assert len(registry.records) >= 20
    for model_id in ("mlx-whisper-large-v3-turbo", "pyannote-speaker-diarization-community-1",
                     "opencv-yunet-2023mar", "insightface-buffalo-l", "faster-whisper-large-v3"):
        assert registry.get(model_id).model_id == model_id
    for record in registry:
        assert record.licence.sources, f"{record.model_id} must cite at least one licence source"


def _rec(**licence):
    return ModelRecord.model_validate({"model_id": "m", "name": "M", "tasks": ["t"], "licence": licence})


def test_policy_personal_learning_allows_non_commercial():
    decision = check_usage_policy(_rec(personal_use="yes", commercial_use="no"), "personal_learning")
    assert decision.allowed
    assert any("not yet verified" in w for w in decision.warnings)


def test_policy_commercial_blocks_non_commercial():
    decision = check_usage_policy(_rec(personal_use="yes", commercial_use="no", weights="CC-BY-NC-4.0"), "commercial")
    assert not decision.allowed


def test_policy_blocks_when_personal_use_forbidden():
    assert not check_usage_policy(_rec(personal_use="no"), "personal_learning").allowed


def test_fingerprint_uses_pinned_revision():
    record = ModelRecord.model_validate({"model_id": "m", "name": "M", "tasks": ["t"],
                                         "weights": {"pinned_revision": "abc123"}})
    assert record.fingerprint == "m@abc123"
    assert _rec().fingerprint == "m@unpinned"
