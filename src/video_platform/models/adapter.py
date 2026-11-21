"""Model adapter contract.

Adapters isolate model-specific code. Business logic only sees normalised schemas.
Concrete adapters arrive with the stages that need them (M2+). Research repositories
that are not pip-installable are wrapped as subprocess adapters with a pinned commit
and a JSON request/response contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from .registry import ModelRecord


class ModelAdapter(ABC):
    """Load → run (batched) → unload. One adapter instance per loaded model."""

    task: ClassVar[str]
    runtime: ClassVar[str] = "cpu"

    def __init__(self, record: ModelRecord, **options: Any) -> None:
        self.record = record
        self.options = options
        self.loaded = False

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def run(self, batch: list[Any]) -> list[Any]: ...

    def unload(self) -> None:
        """Release memory (GPU memory in particular). Safe to call twice."""
        self.loaded = False

    def __enter__(self) -> ModelAdapter:
        self.load()
        self.loaded = True
        return self

    def __exit__(self, *exc: object) -> None:
        self.unload()
