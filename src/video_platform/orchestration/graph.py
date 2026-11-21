"""Stage graph: the declared pipeline DAG (configs/pipeline.yaml)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..core.errors import ConfigError

WorkItemKind = Literal["project", "shot", "segment", "character", "line", "location"]
Runtime = Literal["cpu", "analysis-gpu", "generation-gpu", "audio-gpu", "llm-gpu", "image-gpu"]


class StageSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    version: str = Field(description="Bump when the stage's logic changes; invalidates its artifacts.")
    description: str = ""
    depends_on: list[str] = Field(default_factory=list)
    per: WorkItemKind = "project"
    runtime: Runtime = "cpu"
    params_from: str | list[str] | None = Field(
        default=None, description="Dotted config path(s) hashed into the artifact id."
    )
    model_from: str | list[str] | None = Field(
        default=None, description="Dotted config path(s) holding model registry ids."
    )
    outputs: list[str] = Field(default_factory=list)
    milestone: str = Field(description="Milestone in which an executor is implemented, e.g. 'M2'.")


class PipelineSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    stages: list[StageSpec]
    modes: dict[str, list[str]] = Field(default_factory=dict, description="mode name → target stages")


class StageGraph:
    def __init__(self, spec: PipelineSpec) -> None:
        self.spec = spec
        self._stages: dict[str, StageSpec] = {}
        for stage in spec.stages:
            if stage.name in self._stages:
                raise ConfigError(f"duplicate stage '{stage.name}' in pipeline")
            self._stages[stage.name] = stage
        for stage in spec.stages:
            for dep in stage.depends_on:
                if dep not in self._stages:
                    raise ConfigError(f"stage '{stage.name}' depends on unknown stage '{dep}'")
        for mode, targets in spec.modes.items():
            for target in targets:
                if target not in self._stages:
                    raise ConfigError(f"mode '{mode}' targets unknown stage '{target}'")
        self._order = self._toposort()

    def _toposort(self) -> list[str]:
        order: list[str] = []
        state: dict[str, int] = {}  # 1 = visiting, 2 = done

        def visit(name: str, chain: list[str]) -> None:
            if state.get(name) == 2:
                return
            if state.get(name) == 1:
                cycle = " → ".join([*chain[chain.index(name):], name])
                raise ConfigError(f"pipeline contains a cycle: {cycle}")
            state[name] = 1
            for dep in self._stages[name].depends_on:
                visit(dep, [*chain, name])
            state[name] = 2
            order.append(name)

        for stage in self.spec.stages:  # declaration order keeps output deterministic
            visit(stage.name, [])
        return order

    def __contains__(self, name: str) -> bool:
        return name in self._stages

    def get(self, name: str) -> StageSpec:
        try:
            return self._stages[name]
        except KeyError:
            raise ConfigError(f"unknown stage '{name}'") from None

    @property
    def order(self) -> list[str]:
        return list(self._order)

    def ancestors(self, name: str) -> set[str]:
        result: set[str] = set()
        stack = list(self.get(name).depends_on)
        while stack:
            current = stack.pop()
            if current not in result:
                result.add(current)
                stack.extend(self._stages[current].depends_on)
        return result

    def descendants(self, name: str) -> set[str]:
        self.get(name)
        result: set[str] = set()
        frontier = {name}
        while frontier:
            children = {s.name for s in self.spec.stages if frontier & set(s.depends_on)} - result
            result |= children
            frontier = children
        return result

    def closure(self, targets: list[str]) -> list[str]:
        """Targets plus all their ancestors, in topological order."""
        wanted: set[str] = set()
        for target in targets:
            wanted.add(self.get(target).name)
            wanted |= self.ancestors(target)
        return [name for name in self._order if name in wanted]

    def mode_targets(self, mode: str) -> list[str]:
        if mode not in self.spec.modes:
            raise ConfigError(f"unknown mode '{mode}'. Known: {sorted(self.spec.modes)}")
        return list(self.spec.modes[mode])


def load_pipeline(path: Path) -> StageGraph:
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        spec = PipelineSpec.model_validate(data)
    except FileNotFoundError as exc:
        raise ConfigError(f"pipeline definition not found: {path}") from exc
    except (yaml.YAMLError, ValidationError) as exc:
        raise ConfigError(f"invalid pipeline definition {path}: {exc}") from exc
    return StageGraph(spec)
