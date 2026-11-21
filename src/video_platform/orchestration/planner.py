"""Execution planner: expected artifact ids and cache status per stage.

For every stage the planner derives the artifact id it *would* produce from:
    stage name + version, the expected ids of its dependencies, the hash of its
    configured parameters, and the pinned model fingerprint.
If a valid artifact with that id exists in the index, the stage is a cache hit and
will not run. Changing a parameter, model revision or upstream artifact changes the
id, so only the affected stage and its descendants are re-planned.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..config.loader import LoadedConfig
from ..core.errors import ConfigError
from ..core.hashing import compute_artifact_id, params_hash
from ..models.registry import ModelRegistry
from ..storage.index import ArtifactIndex
from .graph import StageGraph, StageSpec

SOURCE_INPUT_PREFIX = "src:"


def get_config_path(loaded: LoadedConfig, dotted: str) -> Any:
    node: Any = loaded.config.model_dump(mode="json")
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            raise ConfigError(f"config path '{dotted}' does not exist (missing '{key}')")
        node = node[key]
    return node


@dataclass
class PlanItem:
    stage: str
    artifact_id: str
    status: str  # "cached" | "pending"
    milestone: str
    reason: str
    depends_on: list[str] = field(default_factory=list)
    params_hash: str = ""
    model_fingerprint: str = ""

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class Planner:
    def __init__(
        self,
        graph: StageGraph,
        loaded: LoadedConfig,
        registry: ModelRegistry,
        index: ArtifactIndex,
        *,
        source_hash: Callable[[], str] | None = None,
    ) -> None:
        self.graph = graph
        self.loaded = loaded
        self.registry = registry
        self.index = index
        self._source_hash_fn = source_hash or (lambda: index.file_sha256(loaded.source_video_path))
        self._source_hash: str | None = None
        self._expected: dict[str, str] | None = None

    # ---- inputs to the id ------------------------------------------------

    def source_hash(self) -> str:
        if self._source_hash is None:
            path = self.loaded.source_video_path
            if not path.exists():
                raise ConfigError(f"source video not found: {path}")
            self._source_hash = self._source_hash_fn()
        return self._source_hash

    def stage_params(self, spec: StageSpec) -> dict[str, Any]:
        if not spec.params_from:
            return {}
        if isinstance(spec.params_from, list):  # multiple sections: keyed by path
            return {path: get_config_path(self.loaded, path) for path in spec.params_from}
        value = get_config_path(self.loaded, spec.params_from)
        return value if isinstance(value, dict) else {"value": value}

    def model_ids(self, spec: StageSpec) -> list[str]:
        if not spec.model_from:
            return []
        paths = spec.model_from if isinstance(spec.model_from, list) else [spec.model_from]
        out: list[str] = []
        for path in paths:  # optional models (null) are skipped; a list is a preference order
            value = get_config_path(self.loaded, path)
            for v in (value if isinstance(value, list) else [value]):
                if v and str(v) not in out:
                    out.append(str(v))
        return out

    def model_fingerprint(self, spec: StageSpec) -> str:
        return "+".join(self.registry.get(model_id).fingerprint for model_id in self.model_ids(spec))

    def stage_identity(self, spec: StageSpec, dependency_ids: list[str]) -> tuple[str, str, str]:
        p_hash = params_hash(self.stage_params(spec))
        fingerprint = self.model_fingerprint(spec)
        inputs = list(dependency_ids) or [SOURCE_INPUT_PREFIX + self.source_hash()]
        artifact_id = compute_artifact_id(
            stage=spec.name,
            stage_version=spec.version,
            input_ids=inputs,
            params_digest=p_hash,
            model_fingerprint=fingerprint,
        )
        return artifact_id, p_hash, fingerprint

    def expected_ids(self) -> dict[str, str]:
        if self._expected is None:
            expected: dict[str, str] = {}
            for name in self.graph.order:
                spec = self.graph.get(name)
                expected[name] = self.stage_identity(spec, [expected[d] for d in spec.depends_on])[0]
            self._expected = expected
        return dict(self._expected)

    # ---- plan --------------------------------------------------------------

    def plan(self, targets: list[str] | None = None) -> list[PlanItem]:
        stages = self.graph.closure(targets) if targets else self.graph.order
        expected = self.expected_ids()
        items: list[PlanItem] = []
        status_by_stage: dict[str, str] = {}
        for name in stages:
            spec = self.graph.get(name)
            artifact_id, p_hash, fingerprint = self.stage_identity(spec, [expected[d] for d in spec.depends_on])
            cached = self.index.find_valid(artifact_id) is not None
            if cached:
                status, reason = "cached", "valid artifact exists"
            else:
                pending_deps = [d for d in spec.depends_on if status_by_stage.get(d) == "pending"]
                previous = [r for r in self.index.list_records(self.loaded.project_id, name) if r.status == "valid"]
                if pending_deps:
                    reason = f"upstream pending: {', '.join(pending_deps)}"
                elif previous:
                    reason = "inputs, params, model or stage version changed since the last valid run"
                else:
                    reason = "never run"
                status = "pending"
            status_by_stage[name] = status
            items.append(
                PlanItem(
                    stage=name,
                    artifact_id=artifact_id,
                    status=status,
                    milestone=spec.milestone,
                    reason=reason,
                    depends_on=list(spec.depends_on),
                    params_hash=p_hash,
                    model_fingerprint=fingerprint,
                )
            )
        return items
