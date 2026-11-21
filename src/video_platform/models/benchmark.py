"""Benchmark harness skeleton.

Runs candidate model adapters over representative project cases and records wall
time, peak memory and task metrics, so model selection is evidence-based. Real
cases (e.g. 10-minute ASR slice, labelled ASD set) are added in M2+.
"""

from __future__ import annotations

import json
import platform
import time
import tracemalloc
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class BenchmarkCase(BaseModel):
    case_id: str
    inputs: dict[str, Any]
    reference: dict[str, Any] | None = None


class BenchmarkResult(BaseModel):
    model_id: str
    case_id: str
    started_at: datetime
    wall_seconds: float
    peak_host_memory_mb: float
    peak_vram_gb: float | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    error: str | None = None
    hardware: dict[str, str] = Field(default_factory=dict)


MetricFn = Callable[[Any, BenchmarkCase], dict[str, float]]


def _hardware() -> dict[str, str]:
    info = {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version()}
    try:  # optional: torch may not be installed
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # pragma: no cover - environment dependent
        pass
    return info


def _peak_vram_gb() -> float | None:
    try:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**3
    except Exception:  # pragma: no cover
        pass
    return None


def run_benchmark(
    model_id: str,
    run_case: Callable[[BenchmarkCase], Any],
    cases: Iterable[BenchmarkCase],
    metrics: Iterable[MetricFn] = (),
) -> list[BenchmarkResult]:
    metric_fns = list(metrics)
    hardware = _hardware()
    results = []
    for case in cases:
        started = datetime.now(tz=timezone.utc)
        tracemalloc.start()
        t0 = time.perf_counter()
        output, error = None, None
        try:
            output = run_case(case)
        except Exception as exc:  # benchmark must record failures, not crash
            error = f"{type(exc).__name__}: {exc}"
        wall = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        values: dict[str, float] = {}
        if error is None:
            for fn in metric_fns:
                values.update(fn(output, case))
        results.append(
            BenchmarkResult(
                model_id=model_id,
                case_id=case.case_id,
                started_at=started,
                wall_seconds=round(wall, 4),
                peak_host_memory_mb=round(peak / 1024**2, 2),
                peak_vram_gb=_peak_vram_gb(),
                metrics=values,
                error=error,
                hardware=hardware,
            )
        )
    return results


def save_results(results: Iterable[BenchmarkResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.model_dump(mode="json"), ensure_ascii=False) + "\n")
