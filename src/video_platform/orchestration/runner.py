"""Stage execution: runs pending stages in dependency order and records artifacts.

Executors are plain functions registered with ``@executor("<stage>")``. Each receives a
``StageContext`` that gives it typed access to upstream outputs, the artifact store, the
runtime launcher, and an *item cache* for fine-grained resume / regeneration (one item
per shot, dialogue line, character, ...).
"""

from __future__ import annotations

import json
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..core.errors import NotImplementedYet, PlatformError, StageFailure
from ..core.hashing import canonical_json, sha256_bytes, sha256_file
from ..core.log import get_logger
from ..schemas.common import ArtifactRecord, ArtifactRef, Metric
from .graph import StageSpec

log = get_logger(__name__)

Executor = Callable[["StageContext"], None]
EXECUTORS: dict[str, Executor] = {}


def executor(name: str) -> Callable[[Executor], Executor]:
    def register(fn: Executor) -> Executor:
        EXECUTORS[name] = fn
        return fn

    return register


class ItemHandle:
    """A cacheable unit of work inside a stage (e.g. one shot, one line)."""

    def __init__(self, sctx: StageContext, kind: str, payload: Any, work_item: str) -> None:
        self.sctx = sctx
        self.kind = kind
        self.work_item = work_item
        self.item_id = "itm_" + sha256_bytes(canonical_json({"kind": kind, "payload": payload}))[:40]
        self.record = sctx.index.find_valid(self.item_id)

    @property
    def done(self) -> bool:
        return self.record is not None

    def key(self, name: str) -> str:
        return f"items/{self.kind}/{self.item_id}/{name}"

    def path(self, name: str) -> Path:
        """Local scratch path for producing an item file."""
        path = self.sctx.workdir / "items" / self.kind / self.item_id / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def output(self, name: str) -> Path:
        """Local path of a finished item output."""
        if self.record is None:
            raise StageFailure(f"item {self.item_id} ({self.work_item}) has no outputs yet")
        for ref in self.record.outputs:
            if ref.uri and ref.uri.rsplit("/", 1)[-1] == name:
                return self.sctx.store.materialize(self.sctx.store.key_from_uri(ref.uri), self.sctx.cache_dir)
        raise StageFailure(f"item {self.item_id} has no output '{name}'")

    @property
    def data(self) -> dict[str, Any]:
        return dict(self.record.provenance.get("data", {})) if self.record else {}

    def finish(self, files: dict[str, Path], data: dict[str, Any] | None = None, *, status: str = "valid") -> None:
        refs = []
        for name, path in files.items():
            uri = self.sctx.store.put_file(path, self.key(name))
            refs.append(ArtifactRef(artifact_id=self.item_id, uri=uri, sha256=sha256_file(path), role="item"))
        self.record = ArtifactRecord(
            project_id=self.sctx.loaded.project_id,
            stage=f"item:{self.kind}",
            stage_version=self.sctx.spec.version,
            artifact_id=self.item_id,
            work_item=self.work_item,
            outputs=refs,
            params_hash="",
            status=status,  # type: ignore[arg-type]
            provenance={"data": data or {}, "parent_stage": self.sctx.spec.name},
        )
        self.sctx.index.put(self.record)


class StageContext:
    def __init__(self, ctx: Any, spec: StageSpec, artifact_id: str, deps: dict[str, ArtifactRecord], runtime: Any) -> None:
        self.ctx = ctx
        self.loaded = ctx.loaded
        self.config = ctx.loaded.config
        self.store = ctx.store
        self.index = ctx.index
        self.planner = ctx.planner
        self.spec = spec
        self.artifact_id = artifact_id
        self.deps = deps
        self.runtime = runtime
        self.workdir = self.loaded.state_dir / "work" / spec.name / artifact_id
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.loaded.state_dir / "cache"
        self.outputs: list[ArtifactRef] = []
        self.metrics: dict[str, Metric] = {}
        self.warnings: list[str] = []
        self.provenance: dict[str, Any] = {}

    # ---- params --------------------------------------------------------------

    @property
    def params(self) -> dict[str, Any]:
        return self.planner.stage_params(self.spec)

    # ---- upstream ----------------------------------------------------------------

    def dep_ref(self, stage: str, *names: str, required: bool = True) -> ArtifactRef | None:
        record = self.deps.get(stage)
        if record is None:
            raise StageFailure(f"{self.spec.name}: dependency '{stage}' is not available")
        for name in names:  # exact basename first, then suffix (legacy layouts)
            for ref in record.outputs:
                if ref.uri and ref.uri.rsplit("/", 1)[-1] == name:
                    return ref
        for name in names:
            for ref in record.outputs:
                if ref.uri and ref.uri.endswith(name):
                    return ref
        if required:
            raise StageFailure(f"{self.spec.name}: '{stage}' has none of the outputs {names}")
        return None

    def dep_path(self, stage: str, *names: str, required: bool = True) -> Path | None:
        ref = self.dep_ref(stage, *names, required=required)
        return self.store.materialize(self.store.key_from_uri(ref.uri), self.cache_dir) if ref else None

    def dep_json(self, stage: str, *names: str, required: bool = True) -> Any:
        ref = self.dep_ref(stage, *names, required=required)
        return json.loads(self.store.read_bytes(self.store.key_from_uri(ref.uri))) if ref else None

    # ---- outputs -------------------------------------------------------------------

    def key(self, name: str) -> str:
        return f"{self.spec.name}/{self.artifact_id}/{name}"

    def put_json(self, name: str, payload: Any) -> ArtifactRef:
        data = json.dumps(payload, ensure_ascii=False, indent=1, default=str).encode("utf-8")
        uri = self.store.put_bytes(data, self.key(name))
        ref = ArtifactRef(artifact_id=self.artifact_id, uri=uri, sha256=sha256_bytes(data),
                          media_type="application/json", role="output")
        self.outputs.append(ref)
        return ref

    def put_file(self, path: Path, name: str | None = None, *, link: bool = False) -> ArtifactRef:
        name = name or path.name
        uri = self.store.put_file(path, self.key(name), link=link)
        ref = ArtifactRef(artifact_id=self.artifact_id, uri=uri, sha256=sha256_file(path), role="output")
        self.outputs.append(ref)
        return ref

    def own_uri(self, name: str) -> str:
        """URI of an output this stage already wrote (latest match)."""
        for ref in reversed(self.outputs):
            if ref.uri and ref.uri.endswith("/" + name):
                return ref.uri
        raise StageFailure(f"{self.spec.name}: no output '{name}' written yet")

    def metric(self, name: str, value: Any, **kwargs: Any) -> None:
        self.metrics[name] = Metric(value=value, **kwargs)

    def warn(self, message: str) -> None:
        log.warning("%s: %s", self.spec.name, message)
        self.warnings.append(message)

    # ---- items -----------------------------------------------------------------------

    def item(self, kind: str, payload: Any, work_item: str) -> ItemHandle:
        return ItemHandle(self, kind, payload, work_item)


# Approximate GPU memory per runtime class (GB). Exclusive classes always run alone.
EXCLUSIVE_RUNTIMES = {"llm-gpu", "generation-gpu"}
RUNTIME_VRAM_GB = {"cpu": 0.0, "analysis-gpu": 16.0, "audio-gpu": 12.0, "image-gpu": 60.0}


class Runner:
    """Executes pending stages; independent stages run concurrently within a GPU memory budget."""

    def __init__(self, ctx: Any, runtime: Any, *, before_stage: Callable[[StageSpec, str], None] | None = None,
                 max_parallel: int | None = None, after_stage: Callable[[StageSpec, list[str]], None] | None = None) -> None:
        self.ctx = ctx
        self.runtime = runtime
        self.before_stage = before_stage
        self.after_stage = after_stage
        self.max_parallel = max_parallel or getattr(ctx.loaded.config.runtime, "max_parallel_stages", 1)

    def _can_start(self, spec: StageSpec, running: dict[str, StageSpec], gpu_gb: float) -> bool:
        if len(running) >= self.max_parallel:
            return False
        if any(r.runtime in EXCLUSIVE_RUNTIMES for r in running.values()):
            return False
        if spec.runtime in EXCLUSIVE_RUNTIMES:
            return not running
        need = RUNTIME_VRAM_GB.get(spec.runtime, 0.0)
        if need == 0.0:
            return True
        gpu_running = [r for r in running.values() if RUNTIME_VRAM_GB.get(r.runtime, 0.0) > 0]
        if gpu_gb <= 0:
            return not gpu_running  # unknown GPU: one GPU stage at a time
        used = sum(RUNTIME_VRAM_GB[r.runtime] for r in gpu_running)
        return used + need <= gpu_gb * 0.9 or not gpu_running

    def run(self, targets: list[str] | None) -> list[str]:
        from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

        from .. import stages  # noqa: F401  (registers executors)

        plan = self.ctx.planner.plan(targets)
        done = {i.stage for i in plan if i.status == "cached"}
        pending = [i for i in plan if i.status == "pending"]
        for item in pending:
            if item.stage not in EXECUTORS:
                spec = self.ctx.graph.get(item.stage)
                raise NotImplementedYet(f"stage '{spec.name}'", spec.milestone)
        gpu_gb = getattr(self.runtime, "gpu_memory_gb", 0.0) if self.runtime is not None else 0.0
        executed: list[str] = []
        errors: list[BaseException] = []
        running: dict[Any, StageSpec] = {}
        with ThreadPoolExecutor(max_workers=max(1, self.max_parallel)) as pool:
            while pending or running:
                if not errors:
                    for item in list(pending):
                        spec = self.ctx.graph.get(item.stage)
                        if not all(dep in done for dep in spec.depends_on):
                            continue
                        if not self._can_start(spec, {f: s for f, s in running.items()}, gpu_gb):
                            continue
                        if self.before_stage:
                            self.before_stage(spec, item.artifact_id)
                        pending.remove(item)
                        running[pool.submit(self._execute, item)] = spec
                if not running:
                    if pending and not errors:
                        raise StageFailure(f"scheduler stalled with pending stages: {[i.stage for i in pending]}")
                    break
                finished, _ = wait(list(running), return_when=FIRST_COMPLETED)
                for future in finished:
                    spec = running.pop(future)
                    try:
                        future.result()
                        done.add(spec.name)
                        executed.append(spec.name)
                        if self.after_stage:
                            remaining = [i.stage for i in pending] + [s.name for s in running.values()]
                            try:
                                self.after_stage(spec, remaining)
                            except Exception as exc:  # noqa: BLE001 - housekeeping must never fail a run
                                log.warning("after-stage hook failed: %s", exc)
                    except BaseException as exc:  # noqa: BLE001 - collected, re-raised after running stages finish
                        errors.append(exc)
        if errors:
            raise errors[0]
        return executed

    def _execute(self, item: Any) -> None:
        planner, index, graph = self.ctx.planner, self.ctx.index, self.ctx.graph
        expected = planner.expected_ids()
        spec = graph.get(item.stage)
        fn = EXECUTORS[spec.name]
        deps = {}
        for dep in spec.depends_on:
            record = index.find_valid(expected[dep])
            if record is None:
                raise StageFailure(f"{spec.name}: upstream '{dep}' has no valid artifact")
            deps[dep] = record
        sctx = StageContext(self.ctx, spec, item.artifact_id, deps, self.runtime)
        log.info("▶ stage %s (%s)", spec.name, item.artifact_id)
        started = time.time()
        try:
            fn(sctx)
        except PlatformError:
            self._write_traceback(spec.name)
            raise
        except Exception as exc:
            path = self._write_traceback(spec.name)
            raise StageFailure(f"stage {spec.name} failed: {type(exc).__name__}: {exc} (traceback: {path})") from exc
        inputs = [ArtifactRef(artifact_id=expected[d], role="input") for d in spec.depends_on] or [
            ArtifactRef(artifact_id=f"src:{planner.source_hash()}", role="source")
        ]
        record = ArtifactRecord(
            project_id=self.ctx.loaded.project_id,
            stage=spec.name,
            stage_version=spec.version,
            artifact_id=item.artifact_id,
            model_id=(planner.model_ids(spec) or [None])[0],
            model_fingerprint=item.model_fingerprint,
            inputs=inputs,
            outputs=sctx.outputs,
            params=planner.stage_params(spec),
            params_hash=item.params_hash,
            processing_seconds=round(time.time() - started, 2),
            metrics=sctx.metrics,
            status="valid",
            rights_basis=self.ctx.loaded.config.rights.purpose,
            provenance={"warnings": sctx.warnings, **sctx.provenance},
        )
        index.put(record)
        summary = ", ".join(f"{k}={m.value}" for k, m in list(sctx.metrics.items())[:8])
        log.info("✔ stage %s in %.1fs (%d warnings)%s", spec.name, time.time() - started, len(sctx.warnings),
                 f" — {summary}" if summary else "")
        self._log_outputs(sctx, time.time() - started)

    def _log_outputs(self, sctx: StageContext, seconds: float) -> None:
        """Preview what the stage produced: '↳' lines in run.log, full report in state/logs/stages/."""
        rt = self.ctx.loaded.config.runtime
        if not getattr(rt, "log_outputs", True):
            return
        from .outputs import preview

        n = int(getattr(rt, "log_output_items", 4))
        name = sctx.spec.name
        report = [f"stage {name} ({sctx.artifact_id}) in {seconds:.1f}s", "",
                  "metrics: " + json.dumps({k: m.value for k, m in sctx.metrics.items()}, ensure_ascii=False, default=str),
                  f"provenance: {json.dumps(sctx.provenance, ensure_ascii=False, default=str)[:2000]}",
                  f"warnings ({len(sctx.warnings)}):", *[f"  - {w}" for w in sctx.warnings], "", "outputs:"]
        for ref in sctx.outputs:
            fname = (ref.uri or "").rsplit("/", 1)[-1]
            try:
                key = self.ctx.store.key_from_uri(ref.uri)
                data = self.ctx.store.read_bytes(key) if fname.endswith(".json") else None
                size = len(data) if data is not None else (
                    self.ctx.store.materialize(key, sctx.cache_dir).stat().st_size if self.ctx.store.exists(key) else None)
            except Exception:  # noqa: BLE001
                data, size = None, None
            size_txt = f"{size / 1e6:.2f} MB" if isinstance(size, (int, float)) else "?"
            report.append(f"- {fname} ({size_txt})")
            if data is None:
                continue
            try:
                payload = json.loads(data)
            except ValueError:
                continue
            short = preview(fname, payload, n)
            for line in short[: n + 2]:
                log.info("  ↳ %s %s: %s", name, fname, line)
            report += [f"    {line}" for line in preview(fname, payload, n * 3)]
        path = self.ctx.loaded.state_dir / "logs" / "stages" / f"{name}.txt"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(report) + "\n", encoding="utf-8")
        except OSError:
            pass

    def _write_traceback(self, stage: str) -> Path:
        path = self.ctx.loaded.state_dir / "logs" / f"{stage}.traceback.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(traceback.format_exc(), encoding="utf-8")
        return path
