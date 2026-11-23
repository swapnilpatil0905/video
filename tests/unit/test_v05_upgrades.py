"""v0.5: engine choice, clip-level presence matching, separation output discovery, optional models."""

from video_platform.stages.analysis import find_dnr_stems, find_separation_outputs
from video_platform.stages.production import choose_tts_engine, match_presence


def test_choose_tts_engine():
    assert choose_tts_engine({"chatterbox": 0.20, "indicf5": 0.08}) == "indicf5"
    assert choose_tts_engine({"chatterbox": 0.10, "indicf5": 0.08}) == "chatterbox"   # within margin: keep emotion control
    assert choose_tts_engine({"chatterbox": None, "indicf5": 0.3}) == "indicf5"
    assert choose_tts_engine({"chatterbox": None, "indicf5": None}) == "chatterbox"


def test_match_presence():
    centroids = {"SRC_CHAR_0001": [1.0, 0.0], "SRC_CHAR_0003": [0.0, 1.0]}
    assert match_presence([0.9, 0.1], centroids, 0.35) == "SRC_CHAR_0001"
    assert match_presence([0.2, 0.2], centroids, 0.35) is None
    assert match_presence(None, centroids, 0.35) is None


def test_find_separation_outputs(tmp_path):
    (tmp_path / "full_48k_(Vocals)_model_bs_roformer.wav").write_bytes(b"v")
    (tmp_path / "full_48k_(Instrumental)_model_bs_roformer.wav").write_bytes(b"i")
    vocals, rest = find_separation_outputs(tmp_path)
    assert vocals.name.startswith("full_48k_(Vocals)") and rest.name.startswith("full_48k_(Instrumental)")


def test_find_dnr_stems(tmp_path):
    folder = tmp_path / "music_effects"
    folder.mkdir()
    for name in ("speech", "music", "effects"):
        (folder / f"{name}.wav").write_bytes(b"x")
    stems = find_dnr_stems(tmp_path)
    assert {k: v.name for k, v in stems.items()} == {"speech": "speech.wav", "music": "music.wav", "effects": "effects.wav"}
    other = tmp_path / "other"
    other.mkdir()
    (other / "dialog.wav").write_bytes(b"x")
    (other / "sfx.wav").write_bytes(b"x")
    assert set(find_dnr_stems(other)) == {"speech", "effects"}


def test_optional_polish_model_can_be_disabled(make_project):
    from video_platform.config.loader import load_project_config
    from video_platform.models.registry import ModelRegistry
    from video_platform.orchestration.graph import load_pipeline
    from video_platform.orchestration.planner import Planner
    from video_platform.storage.index import ArtifactIndex

    from ..conftest import REPO_ROOT

    loaded = load_project_config(make_project(), overrides=["production.llm.polish_model=null"])
    planner = Planner(load_pipeline(REPO_ROOT / "configs" / "pipeline.yaml"), loaded,
                      ModelRegistry.load(REPO_ROOT / "model_registry"), ArtifactIndex(loaded.state_dir / "a.sqlite"))
    assert planner.model_ids(planner.graph.get("localization")) == ["qwen3-14b"]


def test_emotion2vec_label_parsing():
    from video_platform.workers.emotion import label_from_emotion2vec

    assert label_from_emotion2vec(["生气/angry", "开心/happy", "中立/neutral"], [0.7, 0.2, 0.1]) == "angry"
    assert label_from_emotion2vec(["其他/other"], [0.9]) == "unknown"
    assert label_from_emotion2vec([], []) == "unknown"
