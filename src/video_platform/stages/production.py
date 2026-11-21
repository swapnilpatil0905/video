"""Sample-trailer selection and production (shared by the trailer and full-length production).

Production is item-based: every TTS line, generated shot and lip-sync pass is an item
with a content-derived id. Re-running resumes from finished items; full production
reuses every item already produced for the approved sample; regenerating one shot or
line only redoes that item.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ..core.errors import StageFailure
from ..core.hashing import sha256_file
from ..core.log import get_logger
from ..config.schema import parse_resolution
from ..orchestration.runner import StageContext, executor
from ..runtime.launcher import fill_template
from . import media
from .assets import identity_embeddings

log = get_logger(__name__)
TIER_ORDER = {"main": 0, "supporting": 1, "recurring_secondary": 2, "minor": 3, "background": 4}
IDENTITY_RETRY_BELOW = 0.40


# ---------------------------------------------------------------------------
# Trailer selection (pure function, unit-tested)
# ---------------------------------------------------------------------------


def select_trailer(
    *,
    lines: list[dict],
    profiles: list[dict],
    asd: dict[str, dict],
    chunks: list[dict],
    shots: dict[str, dict],
    segment_shot: dict[str, str],
    video_duration: float,
    target_seconds: float,
    sel: dict,
    avoid_spoilers: bool = True,
) -> tuple[list[dict], list[dict]]:
    tiers = {p["character_id"]: p for p in profiles}
    chunk_of = {sid: c for c in chunks for sid in c["segment_ids"]}
    tail_start = video_duration * (1 - sel["spoiler_tail_fraction"])

    candidates = []
    for line in lines:
        sid, char = line["segment_id"], line.get("character_id")
        dur = line["end_s"] - line["start_s"]
        if line.get("status") != "ok" or not (sel["min_segment_seconds"] <= dur <= sel["max_segment_seconds"]):
            continue
        shot = shots.get(segment_shot.get(sid, ""))
        if shot is None:
            continue
        chunk = chunk_of.get(sid, {})
        if avoid_spoilers and (line["start_s"] >= tail_start or chunk.get("spoiler_level", 0) >= 2):
            continue
        decision = asd.get(sid, {})
        speaking_shot = bool(char) and decision.get("speaking_character") == char
        profile = tiers.get(char or "", {})
        score = profile.get("score", 0.0) + (0.5 if chunk.get("turning_point") else 0.0)
        score += 0.3 if line.get("emotion") not in (None, "neutral", "unknown") else 0.0
        score += 0.4 if speaking_shot else 0.0
        pad = sel["clip_padding_seconds"]
        candidates.append({
            "segment_id": sid, "character_id": char, "shot_id": shot["shot_id"], "start_s": line["start_s"],
            "clip_start": max(shot["start_s"], line["start_s"] - pad), "clip_end": min(shot["end_s"], line["end_s"] + pad),
            "emotion": line.get("emotion"), "speaking_shot": speaking_shot, "score": round(score, 4),
            "third": min(2, int(3 * line["start_s"] / max(video_duration, 1e-6))),
        })

    chosen: list[dict] = []

    def far_enough(c: dict) -> bool:
        return all(abs(c["start_s"] - x["start_s"]) >= sel["min_gap_seconds"] for x in chosen)

    def total() -> float:
        return sum(x["clip_end"] - x["clip_start"] for x in chosen)

    important = [p for p in profiles if p["tier"] in ("main", "supporting", "recurring_secondary")]
    important.sort(key=lambda p: (TIER_ORDER[p["tier"]], -p["score"]))
    coverage_reason: dict[str, str] = {}
    for p in important:
        own = sorted((c for c in candidates if c["character_id"] == p["character_id"]),
                     key=lambda c: (not c["speaking_shot"], -c["score"]))
        pick = next((c for c in own if far_enough(c)), None)
        if pick and total() + (pick["clip_end"] - pick["clip_start"]) <= target_seconds * 1.15:
            chosen.append({**pick, "reason": f"coverage: {p['tier']} character"})
        else:
            coverage_reason[p["character_id"]] = (
                "no speaking line of usable length outside the spoiler zone" if not own else "duration budget exhausted")

    taken = {c["segment_id"] for c in chosen}
    remaining = sorted((c for c in candidates if c["segment_id"] not in taken), key=lambda c: -c["score"])
    while total() < target_seconds and remaining:
        per_third = [sum(x["clip_end"] - x["clip_start"] for x in chosen if x["third"] == t) for t in range(3)]
        weakest = int(np.argmin(per_third))
        pick = next((c for c in remaining if c["third"] == weakest and far_enough(c)), None) or \
            next((c for c in remaining if far_enough(c)), None)
        if pick is None:
            break
        remaining.remove(pick)
        chosen.append({**pick, "reason": f"narrative/emotional coverage (third {pick['third'] + 1})"})

    chosen.sort(key=lambda c: c["start_s"])  # chronological: never invents or reorders events
    selected_chars = {c["character_id"] for c in chosen if c["character_id"]}
    character_selection = []
    for p in profiles:
        cid = p["character_id"]
        character_selection.append({
            "character_id": cid, "importance": p["tier"], "importance_score": p["score"], "screen_faces": p["face_count"],
            "speaking_seconds": p["speech_s"], "shots": p["shot_count"], "selected": cid in selected_chars,
            "has_speaking_shot": any(c["character_id"] == cid and c["speaking_shot"] for c in chosen),
            "reason": "selected" if cid in selected_chars else coverage_reason.get(cid, "not important enough / budget"),
        })
    return chosen, character_selection


@executor("sample_selection")
def sample_selection(s: StageContext) -> None:
    sel = s.config.production.selection.model_dump()
    lines = s.dep_json("localization", "localized_lines.json")
    shots = {sh["shot_id"]: sh for sh in s.dep_json("shot_detection", "shots.json")}
    asd = s.dep_json("active_speaker", "active_speaker.json")
    duration = max((sh["end_s"] for sh in shots.values()), default=0.0)
    segment_shot = {}
    for line in lines:  # primary shot = shot containing the line midpoint
        mid = (line["start_s"] + line["end_s"]) / 2
        hit = next((sid for sid, sh in shots.items() if sh["start_s"] <= mid < sh["end_s"]), None)
        if hit:
            segment_shot[line["segment_id"]] = hit
    chosen, characters = select_trailer(
        lines=lines, profiles=s.dep_json("character_profile", "character_profiles.json"), asd=asd,
        chunks=s.dep_json("narrative", "narrative.json")["chunks"], shots=shots, segment_shot=segment_shot,
        video_duration=duration, target_seconds=s.config.sample.duration_seconds, sel=sel,
        avoid_spoilers=s.config.sample.spoiler_policy == "avoid_major_twists",
    )
    s.put_json("dialogue_selection.json", chosen)
    s.put_json("character_selection.json", characters)
    s.metric("clips", len(chosen))
    s.metric("seconds", round(sum(c["clip_end"] - c["clip_start"] for c in chosen), 2), unit="s")
    missing = [c for c in characters if c["importance"] in ("main", "supporting") and not c["has_speaking_shot"]]
    for c in missing:
        s.warn(f"{c['character_id']} ({c['importance']}) has no speaking shot in the trailer: {c['reason']}")


# ---------------------------------------------------------------------------
# Production engine
# ---------------------------------------------------------------------------


@dataclass
class Clip:
    clip_id: str
    shot_id: str
    src_start: float
    src_end: float
    out_start: float = 0.0

    @property
    def seconds(self) -> float:
        return self.src_end - self.src_start


def full_edl(shots: list[dict], max_seconds: float) -> list[Clip]:
    clips, t = [], 0.0
    for sh in shots:
        start, end = sh["start_s"], sh["end_s"]
        n = max(1, int(np.ceil((end - start) / max_seconds)))
        step = (end - start) / n
        for i in range(n):
            a, b = start + i * step, start + (i + 1) * step if i < n - 1 else end
            clips.append(Clip(f"{sh['shot_id']}_{i:02d}", sh["shot_id"], round(a, 3), round(b, 3), round(t, 3)))
            t += b - a
    return clips


def sample_edl(selection: list[dict]) -> list[Clip]:
    clips, t = [], 0.0
    for i, c in enumerate(selection):
        clips.append(Clip(f"T{i:03d}_{c['shot_id']}", c["shot_id"], round(c["clip_start"], 3), round(c["clip_end"], 3), round(t, 3)))
        t += c["clip_end"] - c["clip_start"]
    return clips


def place_lines(lines: list[dict], clips: list[Clip], *, confine: bool) -> list[dict]:
    """Assign each line to the clip containing its start; compute output time and available slot."""
    placed = []
    starts = sorted(lines, key=lambda x: x["start_s"])
    for clip in clips:
        inside = [ln for ln in starts if clip.src_start <= ln["start_s"] < clip.src_end]
        for j, ln in enumerate(inside):
            nxt = inside[j + 1]["start_s"] if j + 1 < len(inside) else None
            limit = clip.src_end if confine else (nxt if nxt is not None else ln["end_s"] + 1.0)
            if nxt is not None:
                limit = min(limit, nxt)
            placed.append({**ln, "clip_id": clip.clip_id, "out_start": round(clip.out_start + ln["start_s"] - clip.src_start, 3),
                           "available_s": round(max(0.3, limit - ln["start_s"]), 3),
                           "out_clip_end": round(clip.out_start + clip.seconds, 3) if confine else None})
    return placed


def fit_tempo(duration: float, available: float, max_speedup: float) -> tuple[float, bool]:
    """Speed-up factor (≥1) to fit a line into its slot, and whether it still overflows."""
    if duration <= available:
        return 1.0, False
    tempo = duration / available
    return (min(tempo, max_speedup), tempo > max_speedup)


def char_error_rate(reference: str, hypothesis: str) -> float:
    norm = lambda x: re.sub(r"[\s.,!?;:।॥'\"“”‘’…()\[\]{}\-–—]+", "", x.lower())  # noqa: E731  keep matras
    r, h = norm(reference), norm(hypothesis)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hc in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc))
        prev = cur
    return prev[-1] / len(r)


def work_resolution(gen_area: list[int], width: int, height: int) -> tuple[int, int]:
    """Generation-size frame with the output's aspect ratio and the profile's pixel area (multiples of 16)."""
    area = gen_area[0] * gen_area[1]
    w = max(16, int(round((area * width / height) ** 0.5 / 16)) * 16)
    h = max(16, int(round(area / w / 16)) * 16)
    if w * h >= width * height:  # never larger than the output
        return width, height
    return w, h


@dataclass
class VideoJob:
    clip: Clip
    item: Any
    strategy: str
    gid: str | None
    ref: dict | None
    ref_path: Path | None
    workdir: Path
    src: Path | None = None
    wan_input: Path | None = None
    preprocessed: bool = False
    attempts: list = field(default_factory=list)
    last_out: Path | None = None
    best: tuple[Path, float | None] | None = None
    seconds: float = 0.0

    def needs_generation(self) -> bool:
        if self.best is None:
            return True
        return self.best[1] is not None and self.best[1] < IDENTITY_RETRY_BELOW


class Producer:
    """Phased, GPU-efficient production.

    Phases (each model is loaded once per phase, independent work runs in parallel):
      TTS (N parallel workers) → fit (CPU pool) → dialogue stem / M&E bed →
      cut clips (CPU pool) → Wan preprocess (K parallel) → Wan generate (one persistent process) →
      identity check (one batch) → retries → conform (CPU pool, NVENC) → LatentSync (M parallel) → assembly.
    """

    def __init__(self, s: StageContext, scope: str, profile: str) -> None:
        self.s = s
        self.scope = scope
        cfg = s.config
        self.prod = cfg.production
        self.vspec = self.prod.video
        self.sample = scope == "sample"
        self.profile_name = profile
        self.profile = self.prod.profiles[profile]
        self.prefix = f"{profile}/" if self.sample else ""
        res = cfg.sample.resolution if self.sample else cfg.video.target_resolution
        self.width, self.height = parse_resolution(res)
        self.fps = cfg.sample.fps if self.sample else cfg.video.target_fps
        self.work_w, self.work_h = work_resolution(self.profile.gen_area, self.width, self.height)
        self.steps = self.profile.wan_steps
        self.lora_active = bool(self.profile.wan_lora)
        self.source = s.loaded.source_video_path
        self.source_sha = s.planner.source_hash()
        self.lines = s.dep_json("localization", "localized_lines.json")
        self.voices = s.dep_json("voice_design", "voices.json")
        self.refs = s.dep_json("character_references", "character_references.json")
        self.ref_by_source = {r["source_character_id"]: (gid, r) for gid, r in self.refs.items()}
        self.tier_rank = {r["source_character_id"]: i for i, r in enumerate(self.refs.values())}
        self.characters = s.dep_json("character_database", "source_characters.json")
        self.chars_in_shot: dict[str, list[str]] = {}
        for c in self.characters:
            for sh in c["shot_ids"]:
                self.chars_in_shot.setdefault(sh, []).append(c["character_id"])
        self.export_dir = s.loaded.project_dir / ("sample" if self.sample else "output")

        par = self.prod.parallel
        gpu = s.runtime.gpu_memory_gb
        self.cpu_workers = par.cpu_workers or max(2, min(8, s.runtime.cpu_count // 4))
        self.tts_workers = par.tts_workers or (4 if gpu >= 80 else 2 if gpu >= 40 else 1)
        self.pre_workers = par.preprocess_workers or (3 if gpu >= 80 else 1)
        self.ls_workers = par.lipsync_workers or (2 if gpu >= 80 else 1)
        self.offload = {"on": "True", "off": "False"}.get(self.vspec.offload) or ("False" if gpu >= 80 else "True")
        self.persistent_failed = not self.vspec.persistent_worker
        self.gpus = par.gpus or s.runtime.gpu_ids or [0]
        self.wan_workers = par.wan_workers_per_gpu * len(self.gpus)
        if len(self.gpus) > 1:
            self.pre_workers = max(self.pre_workers, len(self.gpus))
            self.ls_workers = max(self.ls_workers, len(self.gpus))
        os.environ["VP_USE_NVENC"] = "1" if par.use_nvenc else "0"
        log.info("%s/%s: %d GPU(s) %.0f GB | work %dx%d → %dx%d | steps=%d lora=%s attn=%s upscale=%s | cpu=%d tts=%d "
                 "preprocess=%d wan=%d lipsync=%d | offload=%s nvenc=%s", scope, profile, len(self.gpus), gpu,
                 self.work_w, self.work_h, self.width, self.height, self.steps, self.profile.wan_lora,
                 self.profile.attention, self.profile.upscale, self.cpu_workers, self.tts_workers, self.pre_workers,
                 self.wan_workers, self.ls_workers, self.offload, media.nvenc_available())
        self.notes: list[str] = []
        self.phase_seconds: dict[str, float] = {}
        self.produced_seconds = 0.0

    # ---- helpers -------------------------------------------------------------

    def local(self, uri: str) -> Path:
        return self.s.store.materialize(self.s.store.key_from_uri(uri), self.s.cache_dir)

    def gpu_name(self) -> str:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True)
            return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else "unknown"
        except OSError:
            return "none"

    def gpu_env(self, index: int) -> dict[str, str]:
        """Pin a parallel job to one GPU (round-robin)."""
        return {"CUDA_VISIBLE_DEVICES": str(self.gpus[index % len(self.gpus)])}

    @staticmethod
    def balance(jobs: list, n: int, weight: Callable[[Any], float]) -> list[list]:
        """Longest-first greedy split into n shards of similar total work."""
        shards: list[list] = [[] for _ in range(max(1, n))]
        loads = [0.0] * len(shards)
        for job in sorted(jobs, key=weight, reverse=True):
            i = loads.index(min(loads))
            shards[i].append(job)
            loads[i] += weight(job)
        return [s for s in shards if s]

    def lora_request(self) -> dict | None:
        if not (self.lora_active and self.profile.wan_lora):
            return None
        record = self.s.ctx.registry.get(self.profile.wan_lora)
        return {"repo": record.weights.hf_repo, "file": record.weights.files[0], "strength": self.profile.wan_lora_strength}

    def finalize_all(self, clips: list[Clip], videos: dict[str, tuple[Path, dict]],
                     synced: dict[str, tuple[Path, dict]]) -> dict[str, tuple[Path, dict]]:
        """Bring every clip to the output resolution: source clips re-cut at full quality, generated clips upscaled."""
        started = time.time()
        out: dict[str, tuple[Path, dict]] = {}
        jobs = []
        for clip in clips:
            video = synced[clip.clip_id][0]
            status = videos[clip.clip_id][1].get("status", "")
            generated = status.startswith("generated")
            needs_upscale = generated and (self.work_w, self.work_h) != (self.width, self.height)
            method = self.profile.upscale if needs_upscale else "lanczos"
            payload = {"input": sha256_file(video) if generated else None, "source": None if generated else self.source_sha,
                       "range": [clip.src_start, clip.src_end], "out": [self.width, self.height, self.fps], "method": method}
            item = self.s.item("final", payload, f"clip:{clip.clip_id}")
            if item.done:
                out[clip.clip_id] = (item.output("final.mp4"), item.data)
            else:
                jobs.append((clip, item, video, generated, method))

        esr = [j for j in jobs if j[4] == "realesrgan"]
        upscaled: dict[str, Path] = {}
        if esr:
            weights = self.s.runtime.weights_dir / "RealESRGAN_x2plus.pth"
            try:
                if not weights.exists():
                    raise FileNotFoundError(weights)
                result = self.s.runtime.call_worker("image", "upscale", {
                    "weights": str(weights),
                    "jobs": [{"id": c.clip_id, "input": str(v), "output": str(it.path("upscaled.mp4"))} for c, it, v, _, _ in esr],
                }, workdir=self.s.workdir / "upscale", tag=self.profile_name)["results"]
                upscaled = {c.clip_id: it.path("upscaled.mp4") for c, it, _, _, _ in esr
                            if result.get(c.clip_id, {}).get("ok")}
            except Exception as exc:
                self.s.warn(f"Real-ESRGAN unavailable, using Lanczos upscaling: {exc}")
            if len(upscaled) < len(esr):
                self.notes.append(f"Real-ESRGAN failed on {len(esr) - len(upscaled)} clip(s); Lanczos used instead")

        def finish(job: tuple) -> tuple:
            clip, item, video, generated, method = job
            if generated:
                src = upscaled.get(clip.clip_id, video)
                used = "realesrgan" if clip.clip_id in upscaled else "lanczos"
            else:  # full-quality re-cut from the original source
                src = media.cut_video(self.source, clip.src_start, clip.src_end, item.path("src_full.mp4"), fps=self.fps)
                used = "source"
            final = media.conform_video(src, item.path("final.mp4"), seconds=clip.seconds, fps=self.fps,
                                        width=self.width, height=self.height)
            return clip, item, final, {"upscale": used}

        for clip, item, final, data in media.pmap(finish, jobs, self.cpu_workers):
            item.finish({"final.mp4": final}, data)
            out[clip.clip_id] = (item.output("final.mp4"), item.data)
        self.timed("upscale_finalize", started)
        return out

    def timed(self, name: str, started: float) -> None:
        self.phase_seconds[name] = round(self.phase_seconds.get(name, 0.0) + time.time() - started, 2)
        log.info("phase %s: %.1fs", name, time.time() - started)

    # ---- TTS -----------------------------------------------------------------------

    def synthesize(self, placed: list[dict]) -> dict[str, dict]:
        started = time.time()
        tts = self.prod.tts
        fallback_voice = next(iter(self.voices.values()), None)
        if fallback_voice is None:
            raise StageFailure("no designed voices available")
        jobs: dict[str, tuple[Any, dict, str]] = {}
        ref_sha: dict[str, str] = {}
        for ln in placed:
            if ln["status"] != "ok":
                continue
            voice = self.voices.get(ln.get("speaker_id") or "")
            if voice is None:
                voice = fallback_voice
                self.s.warn(f"{ln['segment_id']}: no voice for speaker {ln.get('speaker_id')}; using {voice['voice_id']}")
            ref = self.local(voice["uri"])
            ref_sha.setdefault(str(ref), sha256_file(ref))
            for variant in ("text", "short"):
                text = ln["target_text"] if variant == "text" else ln.get("short_text") or ""
                if not text or (variant == "short" and text == ln["target_text"]):
                    continue
                exag = tts.exaggeration.get(ln.get("emotion") or "unknown", tts.exaggeration["unknown"])
                payload = {"text": text, "voice": voice["voice_id"], "ref": ref_sha[str(ref)], "model": tts.model,
                           "lang": self.s.config.languages.target, "exag": exag, "cfg": tts.cfg_weight}
                item = self.s.item("tts", payload, f"line:{ln['segment_id']}:{variant}")
                jobs[f"{ln['segment_id']}|{variant}"] = (item, {"text": text, "ref": str(ref), "exaggeration": exag,
                                                                "cfg_weight": tts.cfg_weight}, voice["voice_id"])
        todo = [(k, v) for k, v in jobs.items() if not v[0].done]
        if todo:
            shards = [todo[i :: self.tts_workers] for i in range(min(self.tts_workers, len(todo)))]
            log.info("TTS: %d lines in %d parallel workers (%d cached)", len(todo), len(shards), len(jobs) - len(todo))

            def run_shard(index_and_shard: tuple[int, list]) -> dict:
                index, shard = index_and_shard
                return self.s.runtime.call_worker("tts", "tts", {
                    "language": self.s.config.languages.target,
                    "items": [{"id": k, "out": str(item.path("line.wav")), **req} for k, (item, req, _) in shard],
                }, workdir=self.s.workdir / "tts", tag=f"shard{index}")

            results = media.pmap(run_shard, list(enumerate(shards)), len(shards))
            per_line = (time.time() - started) * len(shards) / max(1, len(todo))
            merged_lines = {k: v for r in results for k, v in r["lines"].items()}
            merged_errors = {k: v for r in results for k, v in r["errors"].items()}
            for key, (item, _, _) in todo:
                r = merged_lines.get(key)
                if r:
                    item.finish({"line.wav": Path(r["path"])}, {"duration_s": r["duration_s"], "seconds": per_line})
                else:
                    self.s.warn(f"TTS failed for {key}: {merged_errors.get(key)}")
        audio = {}
        for key, (item, _, voice_id) in jobs.items():
            if item.done:
                audio[key] = {"path": item.output("line.wav"), "duration_s": item.data["duration_s"], "voice_id": voice_id}
        self.timed("tts", started)
        return audio

    def fit_lines(self, placed: list[dict], audio: dict[str, dict]) -> list[dict]:
        started = time.time()
        out_dir = self.s.workdir / f"{self.scope}_fitted"
        plans = []
        for ln in placed:
            full, short = audio.get(f"{ln['segment_id']}|text"), audio.get(f"{ln['segment_id']}|short")
            if not full:
                plans.append(({**ln, "placed": False, "reason": "no audio (missing translation or TTS failure)"}, None))
                continue
            choice, variant = full, "text"
            tempo, overflow = fit_tempo(full["duration_s"], ln["available_s"], self.prod.tts.max_speedup)
            if overflow and short:
                tempo2, overflow2 = fit_tempo(short["duration_s"], ln["available_s"], self.prod.tts.max_speedup)
                choice, variant, tempo, overflow = short, "short", tempo2, overflow2
            entry = {**ln, "placed": True, "variant": variant, "tempo": round(tempo, 3), "overflow": overflow,
                     "voice_id": choice["voice_id"], "out_duration_s": round(choice["duration_s"] / tempo, 3),
                     "spoken_text": ln["target_text"] if variant == "text" else ln["short_text"],
                     "audio_path": str(out_dir / f"{ln['segment_id']}.wav")}
            plans.append((entry, (choice["path"], tempo)))

        def retime(plan: tuple[dict, Any]) -> None:
            entry, job = plan
            if job:
                media.retime_audio(job[0], Path(entry["audio_path"]), tempo=job[1], rate=self.prod.tts.sample_rate)

        media.pmap(retime, plans, self.cpu_workers)
        self.timed("fit", started)
        return [entry for entry, _ in plans]

    def build_stem(self, fitted: list[dict], total_seconds: float) -> Path:
        import soundfile as sf

        rate = self.prod.tts.sample_rate
        buffer = np.zeros(int(np.ceil(total_seconds * rate)) + rate, dtype="float32")
        for ln in fitted:
            if not ln.get("placed"):
                continue
            data, _ = sf.read(ln["audio_path"], dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            start = int(ln["out_start"] * rate)
            end = start + len(data)
            if ln.get("out_clip_end") is not None:  # trailer: never bleed into the next clip
                end = min(end, int(ln["out_clip_end"] * rate))
            end = min(end, len(buffer))
            if end > start:
                buffer[start:end] += data[: end - start]
        np.clip(buffer, -1.0, 1.0, out=buffer)
        path = self.s.workdir / f"{self.scope}_dialogue.wav"
        sf.write(str(path), buffer[: int(total_seconds * rate)], rate)
        return path

    def build_bed(self, clips: list[Clip]) -> Path | None:
        import soundfile as sf

        if not self.s.config.rights.reuse_source_music_and_effects:
            self.s.warn("rights.reuse_source_music_and_effects is false: no music/effects bed (dialogue only)")
            return None
        source = self.local(self.s.dep_ref("source_separation", "music_effects.wav").uri)
        out = self.s.workdir / f"{self.scope}_bed.wav"
        with sf.SoundFile(str(source)) as src:
            rate = src.samplerate
            with sf.SoundFile(str(out), "w", samplerate=rate, channels=src.channels) as dst:
                for clip in clips:
                    src.seek(int(clip.src_start * rate))
                    dst.write(src.read(int(round(clip.seconds * rate)), dtype="float32"))
        return out

    # ---- video planning -------------------------------------------------------------

    def primary_character(self, clip: Clip, placed: list[dict]) -> tuple[str, dict] | None:
        visible = set(self.chars_in_shot.get(clip.shot_id, []))
        speaking = [ln.get("character_id") for ln in placed if ln["clip_id"] == clip.clip_id and ln.get("character_id")]
        for char in speaking:
            if char in visible and char in self.ref_by_source:
                return self.ref_by_source[char]
        ranked = sorted((c for c in visible if c in self.ref_by_source), key=lambda c: self.tier_rank.get(c, 99))
        return self.ref_by_source[ranked[0]] if ranked else None

    def plan_videos(self, clips: list[Clip], placed: list[dict]) -> tuple[dict[str, tuple[Path, dict]], list[VideoJob]]:
        done: dict[str, tuple[Path, dict]] = {}
        jobs: list[VideoJob] = []
        ref_sha: dict[str, str] = {}
        for clip in clips:
            target = self.primary_character(clip, placed) if self.vspec.enabled else None
            strategy = "animate_replace" if target else "keep_source"
            gid, ref = target if target else (None, None)
            if strategy == "keep_source" and not self.s.config.rights.reuse_source_footage:
                self.s.warn(f"{clip.clip_id}: no replacement character visible; source footage retained "
                            "(rights.reuse_source_footage is false — policy exception recorded)")
            ref_path = self.local(ref["master"]) if ref else None
            if ref_path and str(ref_path) not in ref_sha:
                ref_sha[str(ref_path)] = sha256_file(ref_path)
            payload = {
                "source": self.source_sha, "range": [clip.src_start, clip.src_end], "strategy": strategy,
                "character": gid, "ref": ref_sha.get(str(ref_path)) if ref_path else None,
                "video": self.vspec.model_dump(include={"model", "wan_task", "frame_num", "work_fps",
                                                        "preprocess_command", "generate_command"}),
                "profile": self.profile.model_dump(include={"gen_area", "wan_steps", "wan_lora", "wan_lora_strength",
                                                            "fallback_steps", "attention"}),
                "out": [self.work_w, self.work_h, self.fps],
            }
            item = self.s.item("video", payload, f"clip:{clip.clip_id}")
            if item.done:
                done[clip.clip_id] = (item.output("clip.mp4"), item.data)
            else:
                jobs.append(VideoJob(clip, item, strategy, gid, ref, ref_path, item.path("x").parent))
        return done, jobs

    # ---- video phases ------------------------------------------------------------------

    def produce_videos(self, jobs: list[VideoJob]) -> dict[str, tuple[Path, dict]]:
        if not jobs:
            return {}
        rt = self.s.runtime
        started = time.time()
        self.produced_seconds += sum(j.clip.seconds for j in jobs)

        def cut(job: VideoJob) -> None:
            job.src = media.cut_video(self.source, job.clip.src_start, job.clip.src_end, job.workdir / "source.mp4",
                                      fps=self.fps)
            if job.strategy == "animate_replace":
                job.wan_input = media.cut_video(self.source, job.clip.src_start, job.clip.src_end,
                                                job.workdir / "wan_input.mp4", fps=self.vspec.work_fps)

        media.pmap(cut, jobs, self.cpu_workers)
        self.timed("cut", started)

        animate = [j for j in jobs if j.strategy == "animate_replace"]
        if animate:
            started = time.time()
            repo = rt.repo("wan")

            index_of = {j.clip.clip_id: i for i, j in enumerate(animate)}

            def preprocess(job: VideoJob) -> None:
                values = self.template_values(job, seed=0)
                try:
                    rt.run_command("wan", fill_template(self.vspec.preprocess_command, values), cwd=repo,
                                   log_name=f"wan_preprocess_{job.clip.clip_id}", timeout_minutes=self.vspec.timeout_minutes,
                                   env_extra=self.gpu_env(index_of[job.clip.clip_id]))
                    job.preprocessed = True
                except Exception as exc:
                    job.attempts.append({"phase": "preprocess", "error": f"{type(exc).__name__}: {exc}"[:500]})

            media.pmap(preprocess, animate, self.pre_workers)
            self.timed("wan_preprocess", started)

            ready = [j for j in animate if j.preprocessed]
            for attempt in range(self.vspec.retries + 1):
                todo = [j for j in ready if j.needs_generation()]
                if not todo:
                    break
                started = time.time()
                self.generate_batch(todo, attempt)
                self.timed("wan_generate", started)
                started = time.time()
                self.measure_identity([j for j in todo if j.last_out is not None])
                self.timed("identity_check", started)

        started = time.time()
        results = media.pmap(self.finalize_video, jobs, self.cpu_workers)
        for job, final, data in results:
            job.item.finish({"clip.mp4": final}, data)
        self.timed("conform", started)
        return {job.clip.clip_id: (job.item.output("clip.mp4"), job.item.data) for job, _, _ in results}

    def template_values(self, job: VideoJob, *, seed: int, out: Path | None = None) -> dict[str, Any]:
        rt = self.s.runtime
        return {"python": rt.python("wan"), "weights": str(rt.weights_dir), "video": str(job.wan_input),
                "ref": str(job.ref_path), "workdir": str(job.workdir), "out": str(out or job.workdir / "wan.mp4"),
                "seed": seed, "offload": self.offload, "area_w": self.profile.gen_area[0],
                "area_h": self.profile.gen_area[1], "steps": self.steps if not self.lora_active else self.profile.fallback_steps}

    def generate_batch(self, jobs: list[VideoJob], attempt: int) -> None:
        """Render clips: one persistent Wan process per GPU slot, each with a balanced share of clips.

        Every clip of a character uses the same master reference image, settings and seed, so the
        result does not depend on which process renders it (consistency is preserved).
        """
        seed = 4242 + attempt
        outs = {j.clip.clip_id: j.workdir / f"wan_{attempt}.mp4" for j in jobs}
        for j in jobs:
            j.last_out = None
        rt = self.s.runtime
        attempted: set[str] = set()
        if not self.persistent_failed:
            shards = self.balance(jobs, self.wan_workers, lambda j: j.clip.seconds)
            log.info("Wan: %d clips across %d persistent worker(s)", len(jobs), len(shards))

            def run_shard(index_and_shard: tuple[int, list[VideoJob]]) -> dict:
                index, shard = index_and_shard
                try:
                    return rt.call_worker("wan", "wan_animate", {
                        "repo": str(rt.repo("wan")), "ckpt_dir": str(rt.weights_dir / "Wan2.2-Animate-14B"),
                        "task": self.vspec.wan_task, "frame_num": self.vspec.frame_num,
                        "offload_model": self.offload == "True", "use_relighting_lora": True,
                        "sampling_steps": self.steps, "attention": self.profile.attention, "lora": self.lora_request(),
                        "jobs": [{"id": j.clip.clip_id, "src_root": str(j.workdir / "process"),
                                  "out": str(outs[j.clip.clip_id]), "seed": seed} for j in shard],
                    }, workdir=self.s.workdir / "wan", tag=f"attempt{attempt}_w{index}",
                        timeout_minutes=self.vspec.timeout_minutes * max(1, len(shard)), env_extra=self.gpu_env(index))
                except Exception as exc:
                    return {"fatal": str(exc), "results": {}}

            responses = media.pmap(run_shard, list(enumerate(shards)), len(shards))
            if self.lora_active and any(r.get("lora_failed") for r in responses):
                self.lora_active = False
                self.steps = self.profile.fallback_steps
                note = (f"distillation LoRA '{self.profile.wan_lora}' could not be applied "
                        f"({next(r['fatal'] for r in responses if r.get('lora_failed'))[:200]}); "
                        f"profile '{self.profile_name}' continues with {self.steps} steps (slower, same quality)")
                self.s.warn(note)
                self.notes.append(note)
                return self.generate_batch(jobs, attempt)
            for r in responses:
                for key, value in (r.get("notes") or {}).items():
                    note = f"wan {key}: {value}"
                    if note not in self.notes:
                        self.notes.append(note)
            fatal = [r["fatal"] for r in responses if r.get("fatal")]
            if fatal:
                self.persistent_failed = True
                self.s.warn("persistent Wan worker unavailable; falling back to one CLI run per clip "
                            f"(slower): {fatal[0][-300:]}")
            results = {k: v for r in responses for k, v in r.get("results", {}).items()}
            for j in jobs:
                r = results.get(j.clip.clip_id)
                if r is None:
                    continue
                attempted.add(j.clip.clip_id)
                j.attempts.append({"seed": seed, "mode": "persistent", **{k: v for k, v in r.items() if k != "ok"}})
                j.seconds += r.get("seconds", 0.0)
                if r.get("ok") and outs[j.clip.clip_id].exists():
                    j.last_out = outs[j.clip.clip_id]
        pending = [j for j in jobs if j.last_out is None and (self.persistent_failed and j.clip.clip_id not in attempted
                                                              or not self.vspec.persistent_worker)]
        if not pending:
            return
        if self.lora_active:
            self.notes.append("CLI fallback cannot load the distillation LoRA; those clips used fallback_steps")

        def run_cli(index_and_shard: tuple[int, list[VideoJob]]) -> None:  # one process per GPU slot at a time
            index, shard = index_and_shard
            for j in shard:
                out = outs[j.clip.clip_id]
                t0 = time.time()
                try:
                    rt.run_command("wan", fill_template(self.vspec.generate_command,
                                                        self.template_values(j, seed=seed, out=out)),
                                   cwd=rt.repo("wan"), log_name=f"wan_generate_{j.clip.clip_id}",
                                   timeout_minutes=self.vspec.timeout_minutes, env_extra=self.gpu_env(index))
                    if not out.exists():
                        found = [f for f in sorted(j.workdir.glob("**/*.mp4"), key=os.path.getmtime)
                                 if f.name not in ("source.mp4", "wan_input.mp4") and not f.name.startswith("wan_")]
                        if found:
                            shutil.copy2(found[-1], out)
                    if out.exists():
                        j.last_out = out
                    j.attempts.append({"seed": seed, "mode": "cli", "seconds": round(time.time() - t0, 2)})
                except Exception as exc:
                    j.attempts.append({"seed": seed, "mode": "cli", "error": f"{type(exc).__name__}: {exc}"[:500]})
                j.seconds += time.time() - t0

        shards = self.balance(pending, self.wan_workers, lambda j: j.clip.seconds)
        media.pmap(run_cli, list(enumerate(shards)), len(shards))

    def measure_identity(self, jobs: list[VideoJob]) -> None:
        if not jobs:
            return

        def frame(job: VideoJob) -> Path:
            out = job.last_out
            return media.still(out, max(0.0, media.duration(out) / 2), out.with_suffix(".mid.jpg"))

        frames = dict(zip((j.clip.clip_id for j in jobs), media.pmap(frame, jobs, self.cpu_workers)))
        embeddings, ratios = identity_embeddings(self.s, frames, with_ratio=True)
        for job in jobs:
            job.attempts[-1]["face_ratio"] = round(ratios.get(job.clip.clip_id, 0.0), 3)
            emb = embeddings.get(job.clip.clip_id)
            master = (job.ref or {}).get("master_embedding")
            similarity = round(float(np.dot(emb, master)), 4) if emb is not None and master else None
            job.attempts[-1]["identity_cos"] = similarity
            if job.best is None or (similarity if similarity is not None else -1) > (job.best[1] if job.best[1] is not None else -1):
                job.best = (job.last_out, similarity)

    def finalize_video(self, job: VideoJob) -> tuple[VideoJob, Path, dict]:
        data: dict[str, Any] = {"strategy": job.strategy, "character": job.gid, "attempts": job.attempts,
                                "seconds": round(job.seconds, 2)}
        produced = None
        if job.strategy == "animate_replace":
            if job.best:
                produced, data["identity_cos"] = job.best
                data["face_ratio"] = max((a.get("face_ratio") or 0.0) for a in job.attempts) if job.attempts else 0.0
                sim = job.best[1]
                data["status"] = ("generated_unverified" if sim is None else
                                  "generated" if sim >= IDENTITY_RETRY_BELOW else "generated_low_identity")
            else:
                data["status"] = "fallback_source"
                self.s.warn(f"{job.clip.clip_id}: generation failed ({len(job.attempts)} attempts); using source footage")
        else:
            data["status"] = "source"
        final = media.conform_video(produced or job.src, job.workdir / "clip.mp4", seconds=job.clip.seconds,
                                    fps=self.fps, width=self.work_w, height=self.work_h)
        return job, final, data

    # ---- lip-sync ---------------------------------------------------------------------

    def lipsync_all(self, clips: list[Clip], videos: dict[str, tuple[Path, dict]], stem: Path,
                    speech_clips: set[str]) -> dict[str, tuple[Path, dict]]:
        spec = self.prod.lipsync
        out: dict[str, tuple[Path, dict]] = {}
        min_ratio = self.profile.lipsync_min_face_ratio
        eligible = [c for c in clips if spec.enabled and c.clip_id in speech_clips
                    and videos[c.clip_id][1].get("status", "").startswith("generated")
                    and (videos[c.clip_id][1].get("face_ratio") or 0.0) >= min_ratio]
        skipped = [c for c in clips if c.clip_id in speech_clips and c not in eligible
                   and videos[c.clip_id][1].get("status", "").startswith("generated")]
        if skipped and min_ratio > 0:
            self.notes.append(f"lip-sync skipped on {len(skipped)} wide shots (face < {min_ratio:.0%} of frame height)")
        for clip in clips:
            if clip not in eligible:
                out[clip.clip_id] = (videos[clip.clip_id][0], {"lipsync": "not_applied"})
        if not eligible:
            return out
        started = time.time()
        audio_dir = self.s.workdir / "lipsync_audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        def cut_audio(clip: Clip) -> Path:
            path = audio_dir / f"{clip.clip_id}.wav"
            media.ffmpeg(["-ss", f"{clip.out_start:.3f}", "-t", f"{clip.seconds:.3f}", "-i", str(stem),
                          "-ar", "16000", "-ac", "1", str(path)])
            return path

        audios = dict(zip((c.clip_id for c in eligible), media.pmap(cut_audio, eligible, self.cpu_workers)))
        jobs = []
        for clip in eligible:
            video = videos[clip.clip_id][0]
            payload = {"video": sha256_file(video), "audio": sha256_file(audios[clip.clip_id]),
                       "spec": spec.model_dump(exclude={"timeout_minutes"}), "steps": self.profile.lipsync_steps}
            item = self.s.item("lipsync", payload, f"clip:{clip.clip_id}")
            if item.done:
                out[clip.clip_id] = (item.output("clip_ls.mp4"), item.data)
            else:
                jobs.append((clip, item, video))
        rt = self.s.runtime

        slot = {id(job[1]): i for i, job in enumerate(jobs)}

        def run(job: tuple[Clip, Any, Path]) -> tuple[Clip, Any, Path, dict]:
            clip, item, video = job
            t0 = time.time()
            raw = item.path("latentsync.mp4")
            values = {"python": rt.python("latentsync"), "video": str(video), "audio": str(audios[clip.clip_id]),
                      "out": str(raw), "weights": str(rt.weights_dir), "ls_steps": self.profile.lipsync_steps}
            try:
                rt.run_command("latentsync", fill_template(spec.command, values), cwd=rt.repo("latentsync"),
                               log_name=f"latentsync_{clip.clip_id}", timeout_minutes=spec.timeout_minutes,
                               env_extra=self.gpu_env(slot[id(item)]))
                final = media.conform_video(raw, item.path("clip_ls.mp4"), seconds=clip.seconds, fps=self.fps,
                                            width=self.work_w, height=self.work_h)
                return clip, item, final, {"lipsync": "applied", "seconds": round(time.time() - t0, 2)}
            except Exception as exc:
                return clip, item, video, {"lipsync": "failed", "error": str(exc)[:500], "seconds": round(time.time() - t0, 2)}

        for clip, item, final, data in media.pmap(run, jobs, self.ls_workers):
            if data["lipsync"] == "failed":
                self.s.warn(f"{clip.clip_id}: lip-sync failed, keeping un-synced shot: {data['error'][:200]}")
            item.finish({"clip_ls.mp4": final}, data)
            out[clip.clip_id] = (item.output("clip_ls.mp4"), item.data)
        self.timed("lipsync", started)
        return out

    # ---- QA ----------------------------------------------------------------------

    def back_transcribe(self, fitted: list[dict], limit: int = 25) -> dict[str, Any]:
        sample = [ln for ln in fitted if ln.get("placed")][:limit]
        if not sample:
            return {}
        try:
            result = self.s.runtime.call_worker("core", "asr", {
                "mode": "check", "model": self.s.ctx.registry.get("faster-whisper-large-v3").weights.hf_repo,
                "language": self.s.config.languages.target,
                "files": [{"id": ln["segment_id"], "path": ln["audio_path"]} for ln in sample],
            }, workdir=self.s.workdir / "qa_asr")
        except Exception as exc:
            self.s.warn(f"back-transcription QA failed: {exc}")
            return {"error": str(exc)[:300]}
        cers = {ln["segment_id"]: round(char_error_rate(ln["spoken_text"], result["texts"].get(ln["segment_id"], "")), 3)
                for ln in sample}
        return {"lines_checked": len(cers), "mean_cer": round(float(np.mean(list(cers.values()))), 3), "per_line": cers}

    # ---- main ----------------------------------------------------------------------

    def run(self) -> None:
        s = self.s
        shots = s.dep_json("shot_detection", "shots.json")
        if self.sample:
            selection = s.dep_json("sample_selection", "dialogue_selection.json")
            clips = sample_edl(selection)
            in_scope = {c["segment_id"] for c in selection}
            lines = [ln for ln in self.lines if ln["segment_id"] in in_scope or any(
                c.src_start <= ln["start_s"] < c.src_end for c in clips)]
        else:
            clips = full_edl(shots, self.vspec.max_clip_seconds)
            lines = self.lines
        if not clips:
            raise StageFailure(f"{self.scope}: nothing to produce (empty edit list)")
        total = sum(c.seconds for c in clips)
        placed = place_lines(lines, clips, confine=self.sample)
        log.info("%s: %d clips, %.1fs, %d lines", self.scope, len(clips), total, len(placed))

        fitted = self.fit_lines(placed, self.synthesize(placed))
        started = time.time()
        stem = self.build_stem(fitted, total)
        bed = self.build_bed(clips)
        self.timed("audio_beds", started)

        done, jobs = self.plan_videos(clips, placed)
        log.info("video: %d clips cached, %d to produce (%d character replacements)", len(done), len(jobs),
                 sum(j.strategy == "animate_replace" for j in jobs))
        videos = {**done, **self.produce_videos(jobs)}
        speech_clips = {ln["clip_id"] for ln in fitted if ln.get("placed")}
        synced = self.lipsync_all(clips, videos, stem, speech_clips)
        finals = self.finalize_all(clips, videos, synced)

        timeline = []
        for clip in clips:
            info, ls_info = videos[clip.clip_id][1], synced[clip.clip_id][1]
            video, final_info = finals[clip.clip_id]
            clip_lines = [ln for ln in fitted if ln["clip_id"] == clip.clip_id]
            timeline.append({
                **asdict(clip), "out_end": round(clip.out_start + clip.seconds, 3), **info, **ls_info, **final_info,
                "video_file": str(video), "source_characters": self.chars_in_shot.get(clip.shot_id, []),
                "lines": [{k: ln.get(k) for k in ("segment_id", "speaker_id", "character_id", "source_text",
                                                  "spoken_text", "voice_id", "emotion", "tempo", "overflow")}
                          for ln in clip_lines],
                "video_model": self.vspec.model if info.get("strategy") == "animate_replace" else None,
                "lipsync_model": self.prod.lipsync.model if ls_info.get("lipsync") == "applied" else None,
            })

        started = time.time()
        name = "trailer" if self.sample else "final"
        tag = f"{self.profile_name}_{name}"
        video_only = media.concat_videos([Path(t["video_file"]) for t in timeline], s.workdir / f"{tag}_video.mp4",
                                         fps=self.fps, final_quality=self.sample)
        mix = media.mix_dialogue_and_bed(stem, bed, s.workdir / f"{tag}_mix.m4a",
                                         dialogue_db=self.prod.mix.dialogue_gain_db, bed_db=self.prod.mix.bed_gain_db,
                                         lufs=s.config.audio.loudness_lufs, true_peak=s.config.audio.true_peak_dbtp)
        final = media.mux(video_only, mix, s.workdir / f"{tag}.mp4")
        srt = media.write_srt(((ln["out_start"], ln["out_start"] + ln["out_duration_s"], ln["spoken_text"])
                               for ln in sorted(fitted, key=lambda x: x["out_start"]) if ln.get("placed")),
                              s.workdir / f"{tag}.srt")
        s.put_file(final, f"{self.prefix}{name}.mp4")
        s.put_file(srt, f"{self.prefix}{name}.srt")
        if self.sample and s.config.sample.generate_subtitles:
            s.put_file(media.burn_subtitles(final, srt, s.workdir / f"{tag}_with_subtitles.mp4"),
                       f"{self.prefix}trailer_with_subtitles.mp4")
        s.put_file(media.still(final, total / 3, s.workdir / f"{tag}_thumbnail.jpg", width=640), f"{self.prefix}thumbnail.jpg")
        self.timed("assembly", started)
        for t in timeline:
            t.pop("video_file", None)
        self.reports(final, total, clips, fitted, timeline)

    def reports(self, final: Path, total: float, clips: list[Clip], fitted: list[dict], timeline: list[dict]) -> None:
        s = self.s
        checks = media.video_checks(final)
        actual = media.duration(final)
        statuses: dict[str, int] = {}
        for t in timeline:
            statuses[t.get("status", "?")] = statuses.get(t.get("status", "?"), 0) + 1
        ids = [t["identity_cos"] for t in timeline if t.get("identity_cos") is not None]
        speech_clips = [t for t in timeline if t["lines"]]
        synced = [t for t in speech_clips if t.get("lipsync") == "applied"]
        placed = [ln for ln in fitted if ln.get("placed")]
        quality = {
            "scope": self.scope, "profile": self.profile_name, "profile_settings": self.profile.model_dump(),
            "optimization_notes": self.notes,
            "note": "objective = computed metrics; no subjective model ratings are included. Human review is the gate.",
            "objective": {
                "duration_expected_s": round(total, 3), "duration_actual_s": round(actual, 3),
                "duration_ok": abs(actual - total) < 0.5,
                "resolution": f"{self.width}x{self.height}", "fps": self.fps, **checks,
                "clips": len(clips), "clip_status": statuses,
                "identity_cos_mean": round(float(np.mean(ids)), 4) if ids else None,
                "identity_cos_min": round(float(np.min(ids)), 4) if ids else None,
                "identity_low_clips": [t["clip_id"] for t in timeline if (t.get("identity_cos") or 1) < IDENTITY_RETRY_BELOW],
                "lines_in_scope": len(fitted), "lines_placed": len(placed),
                "lines_missing": [ln["segment_id"] for ln in fitted if not ln.get("placed")],
                "lines_overflowing": [ln["segment_id"] for ln in placed if ln.get("overflow")],
                "lipsync_coverage": round(len(synced) / len(speech_clips), 3) if speech_clips else None,
                "tts_back_transcription": self.back_transcribe(placed),
            },
            "failed_checks": [],
            "recommendations": [],
        }
        o = quality["objective"]
        if not o["duration_ok"]:
            quality["failed_checks"].append("duration mismatch")
        if o["black_seconds"] > 1.0:
            quality["failed_checks"].append("black frames")
        if o["identity_low_clips"]:
            quality["recommendations"].append(f"regenerate low-identity clips: {o['identity_low_clips'][:10]}")
        if o["lines_overflowing"]:
            quality["recommendations"].append("shorten overflowing lines or allow more speed-up")
        if statuses.get("fallback_source"):
            quality["recommendations"].append("some shots fell back to source footage; check worker_wan logs")
        s.put_json(f"{self.prefix}quality_report.json", quality)
        s.put_json(f"{self.prefix}{'trailer_timeline.json' if self.sample else 'production_timeline.json'}", timeline)

        gpu = self.gpu_name()
        price = float(os.environ.get("VP_GPU_PRICE_PER_HOUR", "0") or 0)
        gpu_phases = ("tts", "wan_preprocess", "wan_generate", "identity_check", "lipsync", "upscale_finalize")
        gen_seconds = sum(self.phase_seconds.get(p, 0.0) for p in gpu_phases)
        produced = self.produced_seconds or total
        per_output_second = gen_seconds / produced if produced else 0.0
        meta = s.dep_json("ingest", "video_metadata.json")
        cost = {
            "gpu": gpu, "gpu_price_per_hour_usd": price or None,
            "this_run_generation_hours": round(gen_seconds / 3600, 3),
            "gpu_seconds_per_output_second": round(per_output_second, 2),
            "estimate_full_production_gpu_hours": round(per_output_second * meta["duration_s"] / 3600, 1) if self.sample else None,
            "estimate_full_production_cost_usd": round(per_output_second * meta["duration_s"] / 3600 * price, 2)
            if self.sample and price else None,
            "basis": "wall-clock of GPU phases in this run ÷ seconds of video newly produced (cached items excluded)",
            "phase_seconds": self.phase_seconds,
            "parallelism": {"cpu_workers": self.cpu_workers, "tts_workers": self.tts_workers,
                            "preprocess_workers": self.pre_workers, "lipsync_workers": self.ls_workers,
                            "wan_mode": "cli_per_clip" if self.persistent_failed else "persistent",
                            "offload_model": self.offload, "nvenc": media.nvenc_available()},
        }
        s.put_json(f"{self.prefix}cost_report.json", cost)

        if self.sample:
            characters = s.dep_json("sample_selection", "character_selection.json")
            coverage = {
                "important": [c["character_id"] for c in characters if c["importance"] in ("main", "supporting", "recurring_secondary")],
                "selected": [c["character_id"] for c in characters if c["selected"]],
                "generated": sorted({self.refs[t["character"]]["source_character_id"] for t in timeline
                                     if t.get("character") and t.get("status", "").startswith("generated")}),
                "with_speaking_shots": [c["character_id"] for c in characters if c["has_speaking_shot"]],
                "with_voice_previews": sorted({v["character_id"] for v in self.voices.values() if v.get("character_id")}),
                "with_lipsync": sorted({self.refs[t["character"]]["source_character_id"] for t in timeline
                                        if t.get("character") and t.get("lipsync") == "applied"}),
                "missing": [{"character_id": c["character_id"], "reason": c["reason"]} for c in characters
                            if c["importance"] in ("main", "supporting") and not c["selected"]],
            }
            s.put_json(f"{self.prefix}character_coverage_report.json", coverage)
        summary = {
            "project": s.config.project.name, "scope": self.scope, "profile": self.profile_name,
            "optimization_notes": self.notes, "video": f"{'trailer' if self.sample else 'final'}.mp4",
            "duration_s": round(actual, 2), "languages": [s.config.languages.source, s.config.languages.target],
            "characters_generated": len(self.refs), "voices": len(self.voices),
            "quality_summary": "pass" if not quality["failed_checks"] else "issues: " + ", ".join(quality["failed_checks"]),
            "known_issues": quality["recommendations"], "cost": cost,
            "next_step": f"bash start.sh approve {self.profile_name}   (or: bash start.sh reject)" if self.sample else "done",
        }
        s.put_json(f"{self.prefix}production_summary.json", summary)
        self.export()

    def export(self) -> None:
        """Copy deliverables into projects/<id>/sample[/<profile>] or projects/<id>/output for easy download."""
        root = self.export_dir
        out = root / self.profile_name if self.sample else root
        out.mkdir(parents=True, exist_ok=True)
        for ref in self.s.outputs:
            if self.sample and f"/{self.prefix}" not in ref.uri:
                continue
            path = self.local(ref.uri)
            shutil.copy2(path, out / path.name)
        if self.sample:
            shutil.copy2(out / "trailer.mp4", root / f"trailer_{self.profile_name}.mp4")
            out = root
            for name in ("dialogue_selection.json", "character_selection.json"):
                shutil.copy2(self.s.dep_path("sample_selection", name), out / name)
            voices = out / "voice_preview"
            voices.mkdir(exist_ok=True)
            for v in self.voices.values():
                shutil.copy2(self.local(v["uri"]), voices / f"{v['voice_id']}.wav")
            refs = out / "character_references"
            refs.mkdir(exist_ok=True)
            for gid, r in self.refs.items():
                shutil.copy2(self.local(r["master"]), refs / f"{gid}_master.png")
                for name, uri in r["variants"].items():
                    shutil.copy2(self.local(uri), refs / f"{gid}_{name}.png")
        (out / "README.txt").write_text(
            "Generated by the AI video platform (personal learning project, private use).\n", encoding="utf-8")
        log.info("deliverables exported to %s", out)


@executor("sample_production")
def sample_production(s: StageContext) -> None:
    """One trailer per profile (e.g. high vs balanced). Shared items (TTS, references, voices) are made once."""
    comparison = {}
    for profile in s.config.sample.compare_profiles:
        producer = Producer(s, "sample", profile)
        producer.run()
        cost = json.loads(s.store.read_bytes(s.store.key_from_uri(s.own_uri(f"{profile}/cost_report.json"))))
        quality = json.loads(s.store.read_bytes(s.store.key_from_uri(s.own_uri(f"{profile}/quality_report.json"))))
        comparison[profile] = {
            "trailer": f"sample/trailer_{profile}.mp4",
            "settings": s.config.production.profiles[profile].model_dump(),
            "measured_gpu_seconds_per_output_second": cost["gpu_seconds_per_output_second"],
            "estimate_full_production_gpu_hours": cost["estimate_full_production_gpu_hours"],
            "estimate_full_production_cost_usd": cost["estimate_full_production_cost_usd"],
            "identity_cos_mean": quality["objective"]["identity_cos_mean"],
            "lipsync_coverage": quality["objective"]["lipsync_coverage"],
            "tts_mean_cer": (quality["objective"]["tts_back_transcription"] or {}).get("mean_cer"),
            "black_seconds": quality["objective"]["black_seconds"],
            "notes": producer.notes,
            "approve_with": f"bash start.sh approve {profile}",
        }
    s.put_json("comparison.json", comparison)
    shutil.copy2(s.store.materialize(s.store.key_from_uri(s.outputs[-1].uri), s.cache_dir),
                 s.loaded.project_dir / "sample" / "comparison.json")


@executor("full_production")
def full_production(s: StageContext) -> None:
    profile = s.config.production.profile
    if not profile:
        raise StageFailure("no production profile chosen; approve a trailer with: bash start.sh approve <profile>")
    Producer(s, "full", profile).run()
