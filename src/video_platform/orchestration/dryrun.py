"""Dry run: execute every pending stage with simulated models, in seconds, before the real run.

Real cached artifacts are read; everything the dry run writes (artifacts, index records, work dirs,
logs) goes to projects/<id>/state/dryrun/ and is thrown away, so the real cache is never touched.
Model workers are replaced by fakes that return answers of the right shape (and exercise retry /
rewrite / rejection paths), so code and data-shape errors surface before any model is loaded.

Not simulated: the final production stages (TTS, Wan, lip-sync, ffmpeg assembly).
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import wave
from pathlib import Path
from typing import Any

from ..core.errors import StageFailure
from ..core.log import get_logger

log = get_logger(__name__)

DRY_TARGETS = ["sample_selection", "character_references", "voice_design"]


# ---- storage overlays ------------------------------------------------------------------------

class OverlayIndex:
    """Writes go to a scratch index; reads fall back to the real one."""

    def __init__(self, real: Any, scratch: Any) -> None:
        self.real, self.scratch = real, scratch

    def put(self, record):
        self.scratch.put(record)

    def get(self, artifact_id):
        return self.scratch.get(artifact_id) or self.real.get(artifact_id)

    def find_valid(self, artifact_id):
        return self.scratch.find_valid(artifact_id) or self.real.find_valid(artifact_id)

    def __getattr__(self, name):
        return getattr(self.real, name)


class OverlayStore:
    def __init__(self, real: Any, scratch: Any) -> None:
        self.real, self.scratch = real, scratch

    def _pick(self, key):
        return self.scratch if self.scratch.exists(key) else self.real

    def put_bytes(self, data, key):
        return self.scratch.put_bytes(data, key)

    def put_file(self, source, key, *, link=False):
        return self.scratch.put_file(source, key, link=False)

    def read_bytes(self, key):
        return self._pick(key).read_bytes(key)

    def exists(self, key):
        return self.scratch.exists(key) or self.real.exists(key)

    def materialize(self, key, cache_dir):
        return self._pick(key).materialize(key, cache_dir)

    def fetch(self, key, destination):
        return self._pick(key).fetch(key, destination)

    def __getattr__(self, name):
        return getattr(self.real, name)


class DryLoaded:
    """The project's LoadedConfig with state redirected to the scratch directory."""

    def __init__(self, real: Any, state_dir: Path) -> None:
        self._real, self._state = real, state_dir

    @property
    def state_dir(self) -> Path:
        return self._state

    def __getattr__(self, name):
        return getattr(self._real, name)


# ---- fake model workers ----------------------------------------------------------------------

def _vec(seed: str, dim: int = 512) -> list[float]:
    import numpy as np

    rng = np.random.default_rng(int(hashlib.sha256(seed.encode()).hexdigest()[:12], 16))
    v = rng.normal(size=dim).astype("float32")
    return (v / np.linalg.norm(v)).tolist()


def _rows_after(text: str, header: str) -> list[str]:
    """First column (id) of the 'a | b | c' rows that follow a header line."""
    if header not in text:
        return []
    ids = []
    for line in text.split(header, 1)[1].splitlines()[1:]:
        if not line.strip():
            if ids:
                break
            continue
        m = re.match(r"^\s*([A-Za-z0-9_\-]+) \|", line)
        if m:
            ids.append(m.group(1))
        elif ids:
            break
    return ids


class FakeLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.long_once = True

    def answer(self, prompt: dict) -> Any:
        pid, user = prompt["id"], prompt.get("user", "")
        if pid == "global":
            chars = sorted(set(re.findall(r"\bSRC_CHAR_\d+\b", user)))
            return {"synopsis": "Dry-run synopsis.", "characters": [
                {"character_id": c, "role": "protagonist" if i == 0 else "supporting", "description": "dry run",
                 "relationships": [{"with": chars[0], "relation": "elder brother"}] if i == 1 else []}
                for i, c in enumerate(chars)]}
        if pid.startswith("chunk_"):
            return {"summary": f"summary of {pid}", "events": ["event"], "turning_point": pid.endswith("1"),
                    "spoiler_level": 0, "dominant_emotion": "neutral", "characters": {}}
        if pid.startswith("SRC_"):
            return {"name": "Ravi", "gender": "male", "age": 30, "appearance": {"face": "oval"}, "wardrobe": "shirt",
                    "image_prompt": "Portrait of a fictional man", "voice_description": "A male speaker."}
        if pid == "identity":
            ids = re.findall(r"^- (\S+): gender=", user, re.M)
            names = ["Arjun", "Meera", "Kabir", "Zoya", "Vikram", "Sana", "Rohan", "Kavya", "Ayaan", "Nisha", "Sameer", "Ananya"]
            return {"characters": [{
                "character_id": c, "given_name": names[i % len(names)] + ("" if i < len(names) else str(i)),
                "family_name": "Reddy" if i % 2 == 0 else "Mirza", "display_name": names[i % len(names)],
                "devanagari": "अर्जुन", "pronunciation": "AR-jun", "family_group": "F1", "profession": "engineer",
                "background": "grew up in Hyderabad", "personality": "calm",
                "forms_of_address": [{"by": "others", "form": f"{names[i % len(names)]} bhai"}],
                "language_profile": {"accent": "hyderabadi" if i % 2 == 0 else "neutral_hindi",
                                     "dialect_intensity": 0.8 if i % 2 == 0 else 0.2, "english_mixing": 0.4,
                                     "urdu_influence": 0.6, "speaking_style": "casual", "speech_habits": ["says arre"]},
            } for i, c in enumerate(ids)]}
        if "Scene lines" in user:
            ids = _rows_after(user, "Scene lines")
            moods = ["comedy", "tragedy", "confrontation", "romance", "neutral"]
            mood = moods[int(hashlib.md5(pid.encode()).hexdigest(), 16) % len(moods)]
            return {"summary": f"dry scene {pid}", "mood": mood, "intensity": 0.5, "present": [],
                    "relationships": [{"a": "x", "b": "y", "relation": "friends", "dynamic": "teasing"}],
                    "lines": [{"id": x, "speaker": None, "listener": None, "intention": "ask", "emotion": "curious",
                               "intensity": 0.5, "subtext": ""} for x in ids],
                    "enhancement": {"comedy": mood == "comedy", "notes": "light teasing"},
                    "location_plan": {"setting": "an office in HITEC City", "real_place": "HITEC City",
                                      "destinations": ["Old City", "Made Up Mall"], "fictional_places": ["Sunrise Tea Stall"],
                                      "place_references_allowed": True},
                    "names_mentioned": [{"source": "林晓", "romanized": "Lin Xiao", "character_id": None}]}
        if "Lines to write" in user:
            ids = _rows_after(user, "Lines to write")
            out = []
            for x in ids:
                text = "अरे, तुम यहाँ क्या कर रहे हो?"
                if self.long_once:  # one over-long line exercises the timing check and rewrite round
                    text = " ".join(["यह बहुत लंबी पंक्ति है जो समय में नहीं आएगी"] * 6)
                    self.long_once = False
                out.append({"id": x, "text": text, "roman": "Arre, tum yahan kya kar rahe ho?", "short": "क्या कर रहे हो?",
                            "meaning_en": "What are you doing here?",
                            "direction": {"emotion": "angry but controlled", "intensity": 0.7, "delivery": "controlled_aggressive",
                                          "speaking_rate": 1.05, "volume": "strong", "pause_before": 0.3, "voice_break": False}})
            return {"lines": out}
        if '"natural": 0-100' in user:
            ids = _rows_after(user, "Lines (id |")
            return {"lines": [{"id": x, "natural": 60 if i == 0 else 85, "regional": 80, "flags": ["literal_translation"] if i == 0 else [],
                               "improved": "अरे, यहाँ कैसे?" if i == 0 else None, "improved_roman": "Arre, yahan kaise?"}
                              for i, x in enumerate(ids)]}
        if "Score 0-100" in user:
            ids = _rows_after(user, "Lines (id |")
            return [{"scores": {"naturalness": 75, "emotional_impact": 70}, "summary": "ok", "issues": [
                {"id": ids[0], "flag": "weak_emotion", "detail": "flat", "rewrite": "तुम... यहाँ?", "rewrite_roman": "Tum... yahan?"}]
                if ids else []}][0]
        return {}

    def run(self, request: dict) -> dict:
        self.calls += 1
        results = {}
        for p in request["prompts"]:
            results[p["id"]] = self.answer(p)
        if request.get("answers_path"):
            Path(request["answers_path"]).write_text(json.dumps(results, ensure_ascii=False)[:20000], encoding="utf-8")
        return {"results": results, "raw": {}, "stats": {"prompts": len(results), "truncated": 0, "split": 0,
                                                          "salvaged": 0, "failed": 0, "empty": 0}}


def _png(path: str) -> None:
    import numpy as np

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2

        seed = int(hashlib.sha256(path.encode()).hexdigest()[:8], 16)  # unique pixels per image, like real ones
        cv2.imwrite(path, np.random.default_rng(seed).integers(0, 255, (96, 64, 3), dtype=np.uint8))
    except ImportError:
        Path(path).write_bytes(b"\x89PNG\r\n\x1a\n")


def _wav(path: str, seconds: float = 1.0, rate: int = 16000) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))


class FakeRuntime:
    """Stands in for RuntimeRunner: same interface, simulated workers, no subprocesses."""

    def __init__(self, real: Any, log_dir: Path) -> None:
        self.real = real
        self.log_dir = log_dir
        self.llm = FakeLLM()
        self.calls: dict[str, int] = {}
        self.emb: dict[str, list[float]] = {}      # image path → simulated face embedding
        self.derived: dict[str, tuple[str, float]] = {}  # output path → (reference path, similarity)
        self.cast_ref: dict[str, list[float]] = {}   # cast member → simulated identity
        self.photo_owner: dict[str, str] = {}           # photo key → cast member

    gpu_memory_gb = 96.0

    def __getattr__(self, name):
        return getattr(self.real, name)

    @staticmethod
    def _key(path: str) -> str:
        """Work-dir and artifact-store copies of an item share their last path parts."""
        return "/".join(Path(path).parts[-3:])

    def _mix(self, base_vec, seed: str, t: float) -> list[float]:
        import numpy as np

        base = np.asarray(base_vec, dtype="float32")
        noise = np.asarray(_vec(seed), dtype="float32")
        noise -= base * float(np.dot(noise, base))
        noise /= np.linalg.norm(noise) + 1e-9
        return (base * t + noise * (1 - t * t) ** 0.5).tolist()

    def embedding(self, path: str, owner: str | None = None) -> list[float]:
        import numpy as np

        path = self._key(path)
        if path in self.emb:
            return self.emb[path]
        if owner is not None:  # every photo of one cast member is the same person (small variation)
            self.photo_owner[path] = owner
            identity = self.cast_ref.setdefault(owner, _vec("cast:" + owner))
            self.emb[path] = self._mix(identity, "photo" + path, 0.97)
            return self.emb[path]
        if path in self.derived:
            ref, t = self.derived[path]
            base = np.asarray(self.cast_ref.get(ref[6:], _vec(ref)) if ref.startswith("@cast:") else self.embedding(ref),
                              dtype="float32")
            noise = np.asarray(_vec("noise" + path), dtype="float32")
            noise -= base * float(np.dot(noise, base))
            noise /= np.linalg.norm(noise) + 1e-9
            emb = (base * t + noise * (1 - t * t) ** 0.5).tolist()
        else:
            emb = _vec(path)
        self.emb[path] = emb
        return emb

    def run_command(self, runtime, argv, **_kw):
        raise StageFailure(f"dry run: external command for runtime '{runtime}' is not simulated")

    def call_worker(self, runtime: str, worker: str, request: dict, *, workdir: Path, tag: str | None = None, **_kw) -> dict:
        self.calls[worker] = self.calls.get(worker, 0) + 1
        workdir.mkdir(parents=True, exist_ok=True)
        json.dumps(request, ensure_ascii=False)  # the request must be serialisable, as for a real worker
        if worker == "llm":
            return self.llm.run(request)
        if worker == "faces" and request.get("mode") == "identity":
            out = {"embeddings": {}, "face_ratio": {}, "det_score": {}, "face_count": {}, "all_faces": {}}
            for item in request["images"]:
                owner = item["id"].split("|")[1] if item["id"].startswith("cast|") else None
                emb = self.embedding(item["path"], owner=owner)
                out["embeddings"][item["id"]] = emb
                out["face_ratio"][item["id"]], out["det_score"][item["id"]], out["face_count"][item["id"]] = 0.3, 0.9, 1
                if item.get("face") == "auto":
                    out["all_faces"][item["id"]] = [{"embedding": emb, "det_score": 0.9, "face_ratio": 0.3}]
            return out
        if worker == "images":
            results = {}
            if request.get("mode", "candidates") == "candidates":
                for j in request["jobs"]:
                    if not j.get("skip_candidate"):
                        _png(j["out"])
                    r = {"candidate": j["out"], "blend": None, "errors": []}
                    if j.get("blend"):
                        _png(j["blend"]["out"])
                        r["blend"] = j["blend"]["out"]
                        owner = self.photo_owner.get(self._key(j["blend"]["photos"][0]), "")
                        self.derived[self._key(j["blend"]["out"])] = ("@cast:" + owner, 0.42)  # resembles, not identical
                    results[j["id"]] = r
            else:
                for j in request["jobs"]:
                    for v in j["variants"].values():
                        _png(v["out"])
                        self.derived[self._key(v["out"])] = (j["master"], 0.8)  # same person, new pose
                    results[j["id"]] = {"variants": {k: v["out"] for k, v in j["variants"].items()}, "errors": []}
            return {"results": results}
        if worker == "voice_design":
            voices = {}
            for item in request["items"]:
                _wav(item["out"])
                voices[item["id"]] = {"path": item["out"], "duration_s": 1.0}
            return {"voices": voices}
        raise StageFailure(f"dry run: worker '{worker}' ({runtime}) is not simulated — is an analysis stage pending?")


# ---- entry point --------------------------------------------------------------------------------

def dry_run(ctx: Any, targets: list[str] | None = None) -> dict[str, Any]:
    """Run pending stages up to the trailer with fake models. Returns {executed, warnings, scratch}."""
    from ..orchestration.planner import Planner
    from ..orchestration.runner import Runner
    from ..storage.index import ArtifactIndex
    from ..storage.store import LocalArtifactStore

    scratch = ctx.loaded.state_dir / "dryrun"
    shutil.rmtree(scratch, ignore_errors=True)
    (scratch / "state" / "logs").mkdir(parents=True)
    index = OverlayIndex(ctx.index, ArtifactIndex(scratch / "index.sqlite"))
    store = OverlayStore(ctx.store, LocalArtifactStore(scratch / "artifacts"))
    loaded = DryLoaded(ctx.loaded, scratch / "state")
    planner = Planner(ctx.graph, ctx.loaded, ctx.registry, index)
    dry_ctx = type(ctx)(loaded, ctx.graph, ctx.registry, index, store, planner)
    runtime = FakeRuntime(None, scratch / "state" / "logs")
    warnings: list[str] = []

    class Collect:
        def __init__(self):
            import logging
            self.handler = logging.Handler()
            self.handler.emit = lambda rec: warnings.append(rec.getMessage()) if rec.levelno >= logging.WARNING else None

        def __enter__(self):
            import logging
            logging.getLogger().addHandler(self.handler)

        def __exit__(self, *exc):
            import logging
            logging.getLogger().removeHandler(self.handler)

    wanted = [t for t in (targets or DRY_TARGETS) if t in ctx.graph.order]
    import logging

    class Tag(logging.Filter):  # dry-run lines are labelled so they are never mistaken for the real run
        def filter(self, record):
            if not str(record.msg).startswith("[dry run] "):
                record.msg = "[dry run] " + str(record.msg)
            return True
    tag = Tag()
    handlers = logging.getLogger().handlers
    for h in handlers:
        h.addFilter(tag)
    try:
        with Collect():
            executed = Runner(dry_ctx, runtime, max_parallel=1).run(wanted)
    finally:
        for h in handlers:
            h.removeFilter(tag)
    return {"executed": executed, "warnings": warnings, "worker_calls": runtime.calls, "scratch": str(scratch)}
