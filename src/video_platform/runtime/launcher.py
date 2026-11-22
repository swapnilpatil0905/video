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
import threading
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


_ANSI = __import__("re").compile(r"\x1b\[[0-9;]*[A-Za-z]")


def last_log_line(path: Path, limit: int = 160) -> str:
    """Most recent meaningful line of a worker log (handles tqdm '\\r' progress bars)."""
    try:
        with open(path, "rb") as handle:
            handle.seek(0, 2)
            handle.seek(max(0, handle.tell() - 8192))
            text = handle.read().decode("utf-8", "replace")
    except OSError:
        return ""
    parts = [_ANSI.sub("", p).strip() for p in text.replace("\r", "\n").split("\n")]
    parts = [p for p in parts if p and not p.startswith("$ (cwd=")]
    return (parts[-1][:limit] if parts else "")


def gpu_summary() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        util, used, total = (x.strip() for x in out.stdout.splitlines()[0].split(","))
        return f"GPU {util}% {float(used) / 1024:.0f}/{float(total) / 1024:.0f} GB"
    except Exception:  # noqa: BLE001
        return "GPU n/a"


GPU_HEAVY_RUNTIMES = {"llm", "image", "wan", "latentsync", "parler", "tts", "indicf5"}


_OWN_PIDS: set[int] = set()  # model processes started by this run (parallel jobs are not "leftovers")
_OWN_LOCK = threading.Lock()


def _own_pids() -> set[int]:
    with _OWN_LOCK:
        live = {p for p in _OWN_PIDS if os.path.exists(f"/proc/{p}")}
        _OWN_PIDS.intersection_update(live)
        return set(live)


def gpu_busy_processes(min_mib: int = 2048, exclude: set[int] | None = None) -> str:
    """Processes holding GPU memory (pid, MiB, name), '' if none / no nvidia-smi."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
    except Exception:  # noqa: BLE001
        return ""
    rows = []
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if (len(parts) >= 2 and parts[1].isdigit() and int(parts[1]) >= min_mib
                and not (parts[0].isdigit() and _related(int(parts[0]), exclude or set()))):
            rows.append(f"pid {parts[0]} {int(parts[1]) / 1024:.0f} GB {parts[2] if len(parts) > 2 else ''}")
    return "; ".join(rows)


def _related(pid: int, ours: set[int]) -> bool:
    """True if pid is one of our model processes or a child of one (e.g. a model engine subprocess)."""
    for _ in range(8):
        if pid in ours:
            return True
        try:
            with open(f"/proc/{pid}/stat") as fh:
                pid = int(fh.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            return False
        if pid <= 1:
            return False
    return False


def _fmt_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


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
        env["HF_HOME"] = os.environ.get("VP_HF_HOME") or str(self.root / "hf")  # one cache, whatever the template sets
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
        heartbeat = float(getattr(self.spec, "heartbeat_seconds", 60) or 0)
        if runtime in GPU_HEAVY_RUNTIMES:
            busy = gpu_busy_processes(exclude=_own_pids())
            if busy:
                log.warning("GPU memory already in use before %s: %s — leftover model processes? "
                            "Stop them with: bash start.sh stop", log_name, busy)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n$ (cwd={cwd}) {' '.join(argv)}\n")
            handle.flush()
            proc = subprocess.Popen(argv, cwd=cwd, env={**self.env(runtime), **(env_extra or {})}, stdout=handle,
                                    stderr=subprocess.STDOUT)
            with _OWN_LOCK:
                _OWN_PIDS.add(proc.pid)
            while True:
                try:
                    proc.wait(timeout=heartbeat if heartbeat > 0 else None)
                    break
                except subprocess.TimeoutExpired:
                    elapsed = time.time() - start
                    if elapsed > timeout:
                        proc.kill()
                        raise StageFailure(f"{log_name}: timed out after {timeout / 60:.0f} min (see {log_path})") from None
                    log.info("⏳ %s running %s | %s | %s", log_name, _fmt_elapsed(elapsed), gpu_summary(),
                             last_log_line(log_path))
        result = proc
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
        reused = reusable_response(req, resp, request)
        if reused is not None:
            log.info("reusing finished %s output from an earlier attempt (%s)", worker, resp.name)
            return reused
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


_REUSE_IGNORED_KEYS = {"split", "answers_path"}  # request details that do not change the answers already received


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _canonical(v) for k, v in sorted(value.items()) if k not in _REUSE_IGNORED_KEYS}
    if isinstance(value, list):
        return [_canonical(v) for v in value]
    return value


def reusable_response(req: Path, resp: Path, request: dict[str, Any]) -> dict[str, Any] | None:
    """The response of an identical, completed worker call in the same work dir (e.g. the stage
    failed later, in code after the worker). Saves re-running long model calls."""
    if not (req.exists() and resp.exists()):
        return None
    try:
        if _canonical(json.loads(req.read_text(encoding="utf-8"))) != _canonical(request):
            return None
        out = json.loads(resp.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return out if isinstance(out, dict) and not out.get("fatal") else None


def start_prefetch(hf_home: Path, repos: list[str], reserve_gb: float, token: str | None = None):
    """Background thread: download upcoming models in stage order, while disk space allows.

    Uses the shared Hugging Face cache (the stages find the files there; concurrent access is
    coordinated by huggingface_hub's lock files). Stops at the first model that would not fit.
    """
    import shutil
    import threading

    def worker() -> None:
        os.environ.setdefault("HF_HOME", str(hf_home))
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # keep the run log readable
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
                cached_gb = hf_cached_gb(Path(hf_home), repo)
                free_gb = shutil.disk_usage(hf_home).free / 1e9
                if cached_gb >= size_gb * 0.99:
                    log.info("prefetch: %s already cached (%.0f GB)", repo, cached_gb)
                    continue
                if size_gb - cached_gb > free_gb - reserve_gb:
                    log.info("prefetch paused before %s (%.0f GB needed, %.0f GB free)", repo, size_gb - cached_gb, free_gb)
                    return
                log.info("prefetching %s (%.0f GB still to download) in the background", repo, size_gb - cached_gb)
                snapshot_download(repo, max_workers=16, token=token)
            except Exception as exc:  # noqa: BLE001 - prefetch is best effort
                log.info("prefetch of %s skipped: %s", repo, str(exc)[:200])

    thread = threading.Thread(target=worker, name="weight-prefetch", daemon=True)
    thread.start()
    return thread


def _repo_folder(hf_home: Path, repo_id: str) -> Path:
    return Path(hf_home) / "hub" / ("models--" + repo_id.replace("/", "--"))


def _real_files(folder: Path) -> set[Path]:
    """Real files behind a cached repo. Follows symlinks: newer huggingface_hub versions keep the
    data in a shared store (hub/blobs/xx/<hash>) and the repo folder holds only links to it."""
    found: set[Path] = set()
    if not folder.exists():
        return found
    for f in folder.rglob("*"):
        if f.name.endswith(".incomplete"):
            continue
        try:
            real = f.resolve(strict=True)
        except (OSError, RuntimeError):
            continue  # dangling link
        if real.is_file():
            found.add(real)
    return found


def hf_cached_gb(hf_home: Path, repo_id: str) -> float:
    """Size of a model's finished files in the Hugging Face cache (GB), counting each file once."""
    return sum(f.stat().st_size for f in _real_files(_repo_folder(hf_home, repo_id))) / 1e9


def hf_repo_gb(repo_id: str, token: str | None = None) -> float | None:
    """Total download size of a model repo (GB), from the Hub API; None if unknown."""
    try:
        from huggingface_hub import HfApi

        info = HfApi(token=token).model_info(repo_id, files_metadata=True)
        return sum((s.size or 0) for s in (info.siblings or [])) / 1e9
    except Exception:  # noqa: BLE001
        return None


def disk_free_gb(path: Path) -> float:
    import shutil

    return shutil.disk_usage(path).free / 1e9


def prune_hf_model(hf_home: Path, repo_id: str) -> float:
    """Delete a model from the Hugging Face cache; returns GB freed (it re-downloads if needed again)."""
    import shutil

    folder = _repo_folder(hf_home, repo_id)
    if not folder.exists():
        return 0.0
    mine = _real_files(folder)
    shared: set[Path] = set()  # files another cached repo also links to stay
    for other in (Path(hf_home) / "hub").glob("models--*"):
        if other != folder:
            shared |= _real_files(other)
    root = Path(hf_home).resolve()
    freed = 0
    for real in mine - shared:
        if root in real.parents or real.is_relative_to(folder.resolve()):
            try:
                freed += real.stat().st_size
                real.unlink()
            except OSError:
                pass
    shutil.rmtree(folder, ignore_errors=True)
    return freed / 1e9


def fill_template(template: list[str], values: dict[str, Any]) -> list[str]:
    """Substitute {placeholders} in a command template."""
    try:
        return [part.format(**values) for part in template]
    except KeyError as exc:
        raise ConfigError(f"command template uses unknown placeholder {exc}; available: {sorted(values)}") from exc
