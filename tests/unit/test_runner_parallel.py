"""The DAG scheduler: dependencies respected, independent GPU stages overlap, exclusive stages run alone."""

import time
from types import SimpleNamespace

from video_platform.config.loader import load_project_config
from video_platform.models.registry import ModelRegistry
from video_platform.orchestration.graph import PipelineSpec, StageGraph
from video_platform.orchestration.planner import Planner
from video_platform.orchestration.runner import Runner, executor
from video_platform.storage.index import ArtifactIndex
from video_platform.storage.store import LocalArtifactStore

STAGES = [
    {"name": "t_root", "version": "1", "milestone": "T"},
    {"name": "t_gpu_a", "version": "1", "depends_on": ["t_root"], "runtime": "analysis-gpu", "milestone": "T"},
    {"name": "t_gpu_b", "version": "1", "depends_on": ["t_root"], "runtime": "audio-gpu", "milestone": "T"},
    {"name": "t_cpu", "version": "1", "depends_on": ["t_root"], "milestone": "T"},
    {"name": "t_llm", "version": "1", "depends_on": ["t_gpu_a", "t_gpu_b", "t_cpu"], "runtime": "llm-gpu", "milestone": "T"},
    {"name": "t_after", "version": "1", "depends_on": ["t_root"], "runtime": "analysis-gpu", "milestone": "T"},
]
SPANS: dict[str, tuple[float, float]] = {}

for _spec in STAGES:
    def _make(name: str):
        @executor(name)
        def _run(s):
            start = time.time()
            time.sleep(0.25)
            SPANS[name] = (start, time.time())
            s.put_json("out.json", {"stage": name})
    _make(_spec["name"])


def _ctx(config_path):
    loaded = load_project_config(config_path)
    graph = StageGraph(PipelineSpec.model_validate({"stages": STAGES}))
    index = ArtifactIndex(loaded.state_dir / "artifacts.sqlite")
    registry = ModelRegistry({})
    planner = Planner(graph, loaded, registry, index)
    return SimpleNamespace(loaded=loaded, graph=graph, index=index, planner=planner, registry=registry,
                           store=LocalArtifactStore(loaded.artifact_root))


def _overlap(a, b):
    return SPANS[a][0] < SPANS[b][1] and SPANS[b][0] < SPANS[a][1]


def test_parallel_scheduler(make_project):
    ctx = _ctx(make_project())
    SPANS.clear()
    executed = Runner(ctx, SimpleNamespace(gpu_memory_gb=96.0), max_parallel=3).run(None)
    assert set(executed) == {s["name"] for s in STAGES}
    assert _overlap("t_gpu_a", "t_gpu_b")                       # independent GPU stages share the GPU
    for other in ("t_gpu_a", "t_gpu_b", "t_cpu"):
        assert SPANS["t_llm"][0] >= SPANS[other][1] - 1e-3      # dependencies finished first
    for other in SPANS:
        if other != "t_llm":
            assert not _overlap("t_llm", other)                 # exclusive stage runs alone
    assert Runner(ctx, SimpleNamespace(gpu_memory_gb=96.0), max_parallel=3).run(None) == []  # all cached


def test_unknown_gpu_runs_gpu_stages_one_at_a_time(make_project):
    ctx = _ctx(make_project())
    SPANS.clear()
    Runner(ctx, SimpleNamespace(gpu_memory_gb=0.0), max_parallel=3).run(None)
    assert not _overlap("t_gpu_a", "t_gpu_b")


def test_before_stage_error_is_reported_at_once_and_running_stages_still_finish(make_project, caplog):
    import pytest

    from video_platform.core.errors import IllegalStateError

    ctx = _ctx(make_project())
    SPANS.clear()

    def before(spec, _artifact_id):
        if spec.name == "t_cpu":
            raise IllegalStateError("illegal transition ANALYZING → SAMPLE_GENERATING")

    with caplog.at_level("ERROR"), pytest.raises(IllegalStateError):
        Runner(ctx, SimpleNamespace(gpu_memory_gb=96.0), max_parallel=3, before_stage=before).run(None)
    assert any("cannot start stage t_cpu" in r.getMessage() for r in caplog.records)
    assert "t_llm" not in SPANS                                   # nothing new starts after the error
    assert ctx.index.find_valid(ctx.planner.expected_ids()["t_root"]) is not None
