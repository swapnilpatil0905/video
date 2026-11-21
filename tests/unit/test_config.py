import pytest

from video_platform.config.loader import apply_overrides, deep_merge, interpolate, load_project_config
from video_platform.core.errors import ConfigError

from ..conftest import PROJECT_001_CONFIG, minimal_config


def test_project_001_config_is_valid(monkeypatch, repo_root):
    monkeypatch.setenv("VP_PLATFORM_ROOT", str(repo_root))
    loaded = load_project_config(PROJECT_001_CONFIG)
    cfg = loaded.config
    assert cfg.project.id == "project_001"
    assert (cfg.languages.source, cfg.languages.target) == ("zh", "hi")
    assert cfg.licensing.usage == "personal_learning"
    assert cfg.budget.max_total_spend_usd is None  # open
    assert cfg.notifications.approval.allow_cli_override is True
    assert loaded.source_video_path == (PROJECT_001_CONFIG.parent / "input" / "source.mp4").resolve()


def test_minimal_config_gets_schema_defaults(make_project):
    loaded = load_project_config(make_project())
    assert loaded.config.sample.duration_seconds == 90
    assert loaded.config.analysis.asr.backend == "faster_whisper"
    assert loaded.config.packages.pip_index_url == "https://pypi.org/simple"


def test_unknown_key_rejected(make_project):
    cfg = minimal_config()
    cfg["sample"] = {"duration_secnds": 60}
    with pytest.raises(ConfigError, match="duration_secnds"):
        load_project_config(make_project(cfg))


def test_literal_secret_rejected(make_project):
    cfg = minimal_config(notifications={"smtp": {"host": "h", "username": "u", "password": "hunter2"}})
    with pytest.raises(ConfigError, match="notifications.smtp.password"):
        load_project_config(make_project(cfg))


def test_env_reference_resolved_and_hidden(make_project, monkeypatch):
    monkeypatch.setenv("SMTP_PASSWORD", "s3cret")
    cfg = minimal_config(notifications={
        "enabled": True, "approval_email": "me@example.com", "provider": "smtp",
        "smtp": {"host": "smtp.example.com", "username": "u", "password": "${SMTP_PASSWORD}"},
    })
    loaded = load_project_config(make_project(cfg))
    secret = loaded.config.notifications.smtp.password
    assert secret.get_secret_value() == "s3cret"
    assert "s3cret" not in repr(loaded.config)


def test_missing_env_var_is_an_error(make_project, monkeypatch):
    monkeypatch.delenv("NOPE_NOT_SET", raising=False)
    cfg = minimal_config(project={"id": "test_project", "name": "${NOPE_NOT_SET}", "project_type": "full_remake",
                                  "source_video": "input/source.mp4"})
    with pytest.raises(ConfigError, match="NOPE_NOT_SET"):
        load_project_config(make_project(cfg))


def test_env_default_syntax():
    missing = []
    assert interpolate("${UNSET_X:-fallback}", {}, missing) == "fallback"
    assert missing == []


def test_email_override_and_validation(make_project):
    loaded = load_project_config(make_project(), email="someone@example.com")
    assert loaded.config.notifications.approval_email == "someone@example.com"
    with pytest.raises(ConfigError, match="approval_email"):
        load_project_config(make_project(), email="not-an-email")


def test_enabled_notifications_need_recipient(make_project):
    with pytest.raises(ConfigError, match="approval_email"):
        load_project_config(make_project(minimal_config(notifications={"enabled": True})))


def test_set_overrides(make_project):
    loaded = load_project_config(make_project(), overrides=["sample.duration_seconds=60", "compute.max_instances=2"])
    assert loaded.config.sample.duration_seconds == 60
    assert loaded.config.compute.max_instances == 2
    with pytest.raises(ConfigError):
        apply_overrides({}, ["no_equals_sign"])


def test_resolution_aspect_mismatch(make_project):
    cfg = minimal_config(video={"target_resolution": "1920x1080", "target_aspect_ratio": "9:16"})
    with pytest.raises(ConfigError, match="aspect"):
        load_project_config(make_project(cfg))


def test_target_language_must_match(make_project):
    cfg = minimal_config(audio={"target_language": "te"})
    with pytest.raises(ConfigError, match="target_language"):
        load_project_config(make_project(cfg))


def test_forbidden_package_index_rejected(make_project):
    cfg = minimal_config(packages={"pip_index_url": "https://mirror.corp.example/simple",
                                   "forbid_hosts_matching": ["*.corp.example"]})
    with pytest.raises(ConfigError, match="forbidden"):
        load_project_config(make_project(cfg))


def test_source_voices_never_cloned(make_project):
    with pytest.raises(ConfigError, match="reuse_source_voices"):
        load_project_config(make_project(minimal_config(rights={"reuse_source_voices": True})))


def test_deep_merge_replaces_lists_and_merges_dicts():
    merged = deep_merge({"a": {"x": 1, "l": [1]}, "b": 1}, {"a": {"y": 2, "l": [2]}})
    assert merged == {"a": {"x": 1, "y": 2, "l": [2]}, "b": 1}
