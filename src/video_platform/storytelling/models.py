"""ModelSelector: one model per role, chosen by preference order and what fits the GPU."""

from __future__ import annotations

from typing import Any, Callable

ROLES = ("analyzer", "writer", "naturalness_reviewer", "screenplay_reviewer")


class ModelSelector:
    def __init__(self, roles: dict[str, Any], defaults: dict[str, str | None], registry_get: Callable[[str], Any],
                 gpu_memory_gb: float | None) -> None:
        self.roles = roles
        self.defaults = defaults
        self.registry_get = registry_get
        self.gpu = gpu_memory_gb

    def candidates(self, role: str) -> list[str]:
        value = self.roles.get(role)
        if value is None:
            value = self.defaults.get(role)
        if value is None:
            return []
        return [value] if isinstance(value, str) else [v for v in value if v]

    def fits(self, model_id: str) -> bool:
        if not self.gpu:
            return True
        try:
            need = float(getattr(self.registry_get(model_id).hardware, "min_vram_gb", 0) or 0)
        except Exception:  # noqa: BLE001 - unknown model: let the registry lookup fail later with a clear error
            return True
        return need <= self.gpu

    def pick(self, role: str) -> str | None:
        options = self.candidates(role)
        for model_id in options:
            if self.fits(model_id):
                return model_id
        return options[-1] if options else None  # nothing fits: smallest listed last (the run will offload or fail)

    def all_models(self) -> list[str]:
        return sorted({m for r in ROLES for m in self.candidates(r)})
