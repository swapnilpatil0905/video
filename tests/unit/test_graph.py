import pytest

from video_platform.core.errors import ConfigError
from video_platform.orchestration.graph import PipelineSpec, StageGraph, load_pipeline


def _graph(stages, modes=None):
    return StageGraph(PipelineSpec.model_validate({"stages": stages, "modes": modes or {}}))


def _s(name, deps=()):
    return {"name": name, "version": "1", "depends_on": list(deps), "milestone": "M2"}


def test_repo_pipeline_loads(repo_root):
    graph = load_pipeline(repo_root / "configs" / "pipeline.yaml")
    order = graph.order
    assert order[0] == "ingest"
    assert order.index("asr") > order.index("extract_audio")
    assert order.index("alignment") > max(order.index(s) for s in ("transcript_filter", "diarization", "shot_detection"))
    assert "speaker_mapping" in graph.mode_targets("analyze")
    assert set(graph.closure(["full_production"])) == set(order)
    assert "sample_production" in graph.ancestors("full_production")
    assert {"ingest", "asr"} <= graph.ancestors("speaker_mapping_prior")


def test_cycle_detected():
    with pytest.raises(ConfigError, match="cycle"):
        _graph([_s("a", ["b"]), _s("b", ["a"])])


def test_unknown_dependency():
    with pytest.raises(ConfigError, match="unknown stage"):
        _graph([_s("a", ["missing"])])


def test_duplicate_stage():
    with pytest.raises(ConfigError, match="duplicate"):
        _graph([_s("a"), _s("a")])


def test_ancestors_descendants():
    graph = _graph([_s("a"), _s("b", ["a"]), _s("c", ["b"]), _s("d", ["a"])])
    assert graph.ancestors("c") == {"a", "b"}
    assert graph.descendants("a") == {"b", "c", "d"}
    assert graph.closure(["c"]) == ["a", "b", "c"]
