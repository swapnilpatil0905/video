"""Run model work in per-model environments ("runtimes") as subprocesses.

Different models pin incompatible torch/transformers versions, so each runtime family
has its own virtualenv (created by scripts/setup_vast.sh under ``runtime.root/venvs``).
Workers exchange JSON files; each subprocess exits after its batch, which frees GPU
memory before the next model loads.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ..core.errors import ConfigError, StageFailure
from ..core.log import get_logger

log = get_logger(__name__)

SRC_DIR = Path(__file__).resolve().parents[2]  # .../src

# runtime family → venv directory name
RUNTIME_VENVS = {
    "core": None,          # the platform's own interpreter
    "diar": "diar",        # pyannote.audio
    "audio": "audio",      # demucs, funasr (SenseVoice)
    "llm": "llm",          # vLLM
    "parler": "parler",    # Indic Parler-TTS
    "tts": "tts",          # Chatterbox
    "image": "image",      # diffusers (Qwen-Image)
    "wan": "wan",          # Wan2.2 repository requirements
    "latentsync": "latentsync",
    "asd": "asd",          # Light-ASD (active speaker detection)
    "sep": "sep",          # audio-separator (BS-RoFormer)
    "indicf5": "indicf5",  # IndicF5 Hindi TTS
    "diarizen": "diarizen",  # DiariZen speaker diarization
}


def detect_gpu_memory_gb() -> float:
    """Total memory of GPU 0 in GB (0 when no NVIDIA GPU is visible)."""
    if os.environ.get("VP_GPU_MEMORY_GB"):
        return float(os.environ["VP_GPU_MEMORY_GB"])
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20)
        return float(out.stdout.strip().splitlines()[0]) / 1024 if out.returncode == 0 and out.stdout.strip() else 0.0
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return 0.0


def detect_gpu_ids() -> list[int]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
        return [int(x) for x in out.stdout.split()] if out.returncode == 0 else []
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return []


class RuntimeRunner:
    def __init__(self, runtime_spec: Any, log_dir: Path) -> None:
        self.spec = runtime_spec
        self.root = Path(runtime_spec.root)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._gpu_gb: float | None = None

    @property
    def gpu_memory_gb(self) -> float:
        if self._gpu_gb is None:
            self._gpu_gb = detect_gpu_memory_gb()
        return self._gpu_gb

    @property
    def gpu_ids(self) -> list[int]:
        if not hasattr(self, "_gpu_ids"):
            self._gpu_ids = detect_gpu_ids()
        return self._gpu_ids

    @property
    def cpu_count(self) -> int:
        return os.cpu_count() or 4

    # ---- paths ------------------------------------------------------------------

    def python(self, runtime: str) -> str:
        if runtime in self.spec.venvs:
            return self.spec.venvs[runtime]
        if runtime not in RUNTIME_VENVS:
            raise ConfigError(f"unknown runtime '{runtime}'")
        venv = RUNTIME_VENVS[runtime]
        if venv is None:
            return sys.executable
        path = self.root / "venvs" / venv / "bin" / "python"
        if not path.exists():
            raise ConfigError(f"runtime '{runtime}' is not installed ({path}). Run: bash scripts/setup_vast.sh")
        return str(path)

    def repo(self, name: str) -> Path:
        rel = self.spec.repos.get(name)
        if not rel:
            raise ConfigError(f"no repository configured for '{name}'")
        path = Path(rel) if Path(rel).is_absolute() else self.root / rel
        if not path.exists():
            raise ConfigError(f"repository '{name}' not found at {path}. Run: bash scripts/setup_vast.sh")
        return path

    @property
    def weights_dir(self) -> Path:
        path = Path(self.spec.weights_dir)
        return path if path.is_absolute() else self.root / path

    def env(self, runtime: str) -> dict[str, str]:
        env = dict(os.environ)
        env.setdefault("HF_HOME", str(self.root / "hf"))
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(SRC_DIR), env.get("PYTHONPATH", "")]))
        env["PYTHONUNBUFFERED"] = "1"
        env["VP_DEVICE"] = self.spec.device
        # CUDA libraries shipped as pip wheels (nvidia-*-cu12) are not on the loader path by default.
        python = Path(self.python(runtime))
        site = glob.glob(str(python.parent.parent / "lib" / "python3*" / "site-packages" / "nvidia" / "*" / "lib"))
        if site:
            env["LD_LIBRARY_PATH"] = os.pathsep.join([*site, env.get("LD_LIBRARY_PATH", "")])
        return env

    # ---- execution -------------------------------------------------------------------

    def run_command(
        self, runtime: str, argv: list[str], *, cwd: Path | None = None, log_name: str, timeout_minutes: float | None = None,
        env_extra: dict[str, str] | None = None,
    ) -> None:
        log_path = self.log_dir / f"{log_name}.log"
        timeout = (timeout_minutes or self.spec.worker_timeout_minutes) * 60
        start = time.time()
        log.info("run [%s] %s (log: %s)", runtime, " ".join(argv[:4]) + (" …" if len(argv) > 4 else ""), log_path)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n$ (cwd={cwd}) {' '.join(argv)}\n")
            handle.flush()
            try:
                result = subprocess.run(
                    argv, cwd=cwd, env={**self.env(runtime), **(env_extra or {})}, stdout=handle,
                    stderr=subprocess.STDOUT, timeout=timeout,
                )
            except subprocess.TimeoutExpired as exc:
                raise StageFailure(f"{log_name}: timed out after {timeout / 60:.0f} min (see {log_path})") from exc
        if result.returncode != 0:
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
            raise StageFailure(f"{log_name}: exit code {result.returncode} (see {log_path})\n  " + "\n  ".join(tail))
        log.info("done [%s] %s in %.1fs", runtime, log_name, time.time() - start)

    def call_worker(
        self, runtime: str, worker: str, request: dict[str, Any], *, workdir: Path, timeout_minutes: float | None = None,
        tag: str | None = None, env_extra: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        workdir.mkdir(parents=True, exist_ok=True)
        suffix = f".{tag}" if tag else ""
        req = workdir / f"{worker}{suffix}.request.json"
        resp = workdir / f"{worker}{suffix}.response.json"
        req.write_text(json.dumps(request, ensure_ascii=False, indent=1), encoding="utf-8")
        if resp.exists():
            resp.unlink()
        self.run_command(
            runtime,
            [self.python(runtime), "-m", "video_platform.workers", worker, str(req), str(resp)],
            log_name=f"worker_{worker}_{tag or workdir.name}",
            timeout_minutes=timeout_minutes,
            env_extra=env_extra,
        )
        if not resp.exists():
            raise StageFailure(f"worker {worker} produced no response file")
        return json.loads(resp.read_text(encoding="utf-8"))


def start_prefetch(hf_home: Path, repos: list[str], reserve_gb: float, token: str | None = None):
    """Background thread: download upcoming models in stage order, while disk space allows.

    Uses the shared Hugging Face cache (the stages find the files there; concurrent access is
    coordinated by huggingface_hub's lock files). Stops at the first model that would not fit.
    """
    import shutil
    import threading

    def worker() -> None:
        os.environ.setdefault("HF_HOME", str(hf_home))
        try:
            from huggingface_hub import HfApi, snapshot_download
        except ImportError:
            return
        api = HfApi(token=token)
        for repo in repos:
            try:
                folder = Path(hf_home) / "hub" / ("models--" + repo.replace("/", "--"))
                info = api.model_info(repo, files_metadata=True)
                size_gb = sum((s.size or 0) for s in (info.siblings or [])) / 1e9
                cached_gb = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()) / 1e9 if folder.exists() else 0
                free_gb = shutil.disk_usage(hf_home).free / 1e9
                if size_gb - cached_gb > free_gb - reserve_gb:
                    log.info("prefetch paused before %s (%.0f GB needed, %.0f GB free)", repo, size_gb - cached_gb, free_gb)
                    return
                log.info("prefetching %s (%.0f GB) in the background", repo, size_gb)
                snapshot_download(repo, max_workers=16, token=token)
            except Exception as exc:  # noqa: BLE001 - prefetch is best effort
                log.info("prefetch of %s skipped: %s", repo, str(exc)[:200])

    thread = threading.Thread(target=worker, name="weight-prefetch", daemon=True)
    thread.start()
    return thread


def prune_hf_model(hf_home: Path, repo_id: str) -> float:
    """Delete a model from the Hugging Face cache; returns GB freed (it re-downloads if needed again)."""
    import shutil

    folder = Path(hf_home) / "hub" / ("models--" + repo_id.replace("/", "--"))
    if not folder.exists():
        return 0.0
    size = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file() and not f.is_symlink())
    shutil.rmtree(folder, ignore_errors=True)
    return size / 1e9


def fill_template(template: list[str], values: dict[str, Any]) -> list[str]:
    """Substitute {placeholders} in a command template."""
    try:
        return [part.format(**values) for part in template]
    except KeyError as exc:
        raise ConfigError(f"command template uses unknown placeholder {exc}; available: {sorted(values)}") from exc
