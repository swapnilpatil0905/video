"""Character reference images and designed voices (identity assets reused everywhere)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..orchestration.runner import StageContext, executor
from .analysis import character_centroids

NEGATIVE = (
    "cartoon, anime, illustration, painting, 3d render, cgi, plastic skin, doll, airbrushed, deformed face, "
    "asymmetric eyes, extra fingers, blurry, low quality, watermark, text, celebrity"
)
# Cosine similarity above this to the original actor's face centroid → too similar; regenerate.
MAX_SOURCE_SIMILARITY = 0.45
MAX_ATTEMPTS = 3


def identity_embeddings(s: StageContext, images: dict[str, Path], *, with_ratio: bool = False):
    """Face embedding (and face-height ratio) of the largest face per image, in one worker call."""
    if not images:
        return ({}, {}) if with_ratio else {}
    result = s.runtime.call_worker("core", "faces", {
        "mode": "identity", "images": [{"id": k, "path": str(v)} for k, v in images.items()],
    }, workdir=s.workdir / "identity")
    return (result["embeddings"], result.get("face_ratio", {})) if with_ratio else result["embeddings"]


@executor("character_references")
def character_references(s: StageContext) -> None:
    p = s.params
    designs = s.dep_json("character_design", "character_designs.json")
    centroids = character_centroids(s)
    images = s.config.production.character_images
    registry = s.ctx.registry
    pending = {d["gen_character_id"]: d for d in designs}
    final: dict[str, dict] = {}
    attempt_seed = {gid: p["seed"] + 1000 * i for i, gid in enumerate(pending)}

    for attempt in range(MAX_ATTEMPTS):
        if not pending:
            break
        chars = []
        for gid, d in pending.items():
            out_dir = s.workdir / gid / f"attempt{attempt}"
            out_dir.mkdir(parents=True, exist_ok=True)
            chars.append({
                "id": gid, "seed": attempt_seed[gid] + attempt * 7919, "out_dir": str(out_dir),
                "prompt": f"{d['image_prompt']}. {images.style_suffix}", "negative": NEGATIVE,
                "variants": {k: f"{v} Keep the exact same face, identity, hairstyle and outfit. Photorealistic."
                             for k, v in images.variants.items()},
            })
        result = s.runtime.call_worker("image", "images", {
            "model_repo": registry.get(images.model).weights.hf_repo,
            "edit_repo": registry.get(images.edit_model).weights.hf_repo,
            "width": images.width, "height": images.height, "steps": images.steps, "edit_steps": images.edit_steps,
            "true_cfg_scale": images.true_cfg_scale, "characters": chars,
        }, workdir=s.workdir / f"attempt{attempt}")["characters"]

        masters = {gid: Path(r["master"]) for gid, r in result.items() if r["master"]}
        embeddings = identity_embeddings(s, {**masters, **{
            f"{gid}|{name}": Path(path) for gid, r in result.items() for name, path in r["variants"].items()}})
        retry = {}
        for gid, d in pending.items():
            r = result.get(gid, {})
            master_emb = embeddings.get(gid)
            if not r.get("master") or master_emb is None:
                s.warn(f"{gid}: no usable face in master image (attempt {attempt + 1}): {r.get('errors')}")
                retry[gid] = d
                continue
            src = centroids.get(d["source_character_id"])
            src_sim = float(np.dot(master_emb, src)) if src else None
            if src_sim is not None and src_sim > MAX_SOURCE_SIMILARITY:
                s.warn(f"{gid}: master too similar to the original actor (cos={src_sim:.2f}); regenerating")
                retry[gid] = d
                continue
            consistency = {}
            for name in r["variants"]:
                emb = embeddings.get(f"{gid}|{name}")
                consistency[name] = round(float(np.dot(master_emb, emb)), 4) if emb is not None else None
            final[gid] = {"design": d, "result": r, "source_similarity": src_sim, "variant_identity": consistency,
                          "master_embedding": master_emb}
        pending = retry

    references = {}
    for gid, info in final.items():
        r = info["result"]
        master_ref = s.put_file(Path(r["master"]), f"{gid}/master.png")
        variants = {}
        for name, path in r["variants"].items():
            sim = info["variant_identity"].get(name)
            if sim is not None and sim < 0.5:
                s.warn(f"{gid}/{name}: variant identity drift (cos={sim}); excluded from the sheet")
                continue
            variants[name] = s.put_file(Path(path), f"{gid}/{name}.png").uri
        references[gid] = {
            "source_character_id": info["design"]["source_character_id"], "name": info["design"]["name"],
            "master": master_ref.uri, "variants": variants, "master_embedding": info["master_embedding"],
            "source_centroid": centroids.get(info["design"]["source_character_id"]),
            "source_similarity": info["source_similarity"], "variant_identity": info["variant_identity"],
        }
    for gid in pending:
        s.warn(f"{gid}: no acceptable reference after {MAX_ATTEMPTS} attempts; its shots will fall back")
    s.put_json("character_references.json", references)
    s.metric("characters_with_references", len(references))
    s.metric("characters_failed", len(pending))
    if references:
        sims = [v for r in references.values() for v in r["variant_identity"].values() if v is not None]
        if sims:
            s.metric("sheet_identity_mean_cos", round(float(np.mean(sims)), 4), kind="objective")


@executor("voice_design")
def voice_design(s: StageContext) -> None:
    spec = s.config.production.voice_design
    designs = {d["source_character_id"]: d for d in s.dep_json("character_design", "character_designs.json")}
    profiles = {p["character_id"]: p for p in s.dep_json("character_profile", "character_profiles.json")}
    mappings = s.dep_json("speaker_mapping", "speaker_mapping.json")
    items, meta = [], {}
    for n, m in enumerate(sorted(mappings, key=lambda m: m["speaker_id"])):
        if m["supporting_speech_s"] <= 0:
            continue
        char = m.get("character_id")
        design = designs.get(char or "")
        gender = (design or {}).get("gender") or profiles.get(char or "", {}).get("gender")
        description = (design or {}).get("voice_description") or spec.default_description.format(
            gender=gender if gender in ("male", "female") else ("male" if n % 2 else "female"))
        voice_id = f"VOICE_{m['speaker_id'].removeprefix('SPK_')}"
        out = s.workdir / f"{voice_id}.wav"
        items.append({"id": voice_id, "description": description, "text": spec.reference_text, "out": str(out),
                      "seed": 1000 + n})
        meta[voice_id] = {"speaker_id": m["speaker_id"], "character_id": char,
                          "gen_character_id": (design or {}).get("gen_character_id"), "description": description,
                          "mapping_status": m["status"]}
    result = s.runtime.call_worker("parler", "voice_design", {
        "hf_repo": s.ctx.registry.get(spec.model).weights.hf_repo, "items": items,
    }, workdir=s.workdir)["voices"]
    voices = {}
    for voice_id, info in meta.items():
        r = result.get(voice_id)
        if not r:
            s.warn(f"{voice_id}: voice generation failed")
            continue
        ref = s.put_file(Path(r["path"]), f"{voice_id}.wav")
        voices[info["speaker_id"]] = {**info, "voice_id": voice_id, "uri": ref.uri, "duration_s": r["duration_s"],
                                      "origin": "generated"}
    s.put_json("voices.json", voices)
    s.metric("voices", len(voices))
