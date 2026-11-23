"""Character reference images and designed voices (identity assets reused everywhere)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.errors import StageFailure
from ..orchestration.runner import StageContext, executor
from ..storytelling.gender import mentions_other_gender, norm_gender
from .analysis import character_centroids

MAX_ATTEMPTS = 3


def identity_embeddings(s: StageContext, images: dict[str, Path], *, with_ratio: bool = False,
                        faces: dict[str, str] | None = None, full: bool = False):
    """Face embedding (and face-height ratio / detection score) per image, in one worker call."""
    if not images:
        return {} if not (with_ratio or full) else ({}, {}) if with_ratio else {}
    result = s.runtime.call_worker("core", "faces", {
        "mode": "identity",
        "images": [{"id": k, "path": str(v), "face": (faces or {}).get(k, "largest")} for k, v in images.items()],
    }, workdir=s.workdir / "identity")
    if full:
        return {k: {"embedding": result["embeddings"].get(k), "face_ratio": result.get("face_ratio", {}).get(k, 0.0),
                    "gender": (result.get("gender") or {}).get(k),
                    "det_score": result.get("det_score", {}).get(k, 1.0), "face_count": result.get("face_count", {}).get(k),
                    "all_faces": result.get("all_faces", {}).get(k)}
                for k in images}
    return (result["embeddings"], result.get("face_ratio", {})) if with_ratio else result["embeddings"]


@executor("character_references")
def character_references(s: StageContext) -> None:
    """Realistic, original faces: N candidates per character (identity-aware prompts), consenting-cast
    blends, measured selection, pose sheet, contact sheet. See storytelling/casting.py."""
    from ..storytelling import casting
    from ..core.hashing import sha256_file

    spec = s.config.production.character_images
    designs = s.dep_json("character_design", "character_designs.json")
    identity_doc = s.dep_json("character_identity", "character_identity.json")
    identity = {c["character_id"]: c for c in identity_doc.get("characters", [])}
    region = identity_doc.get("region") or s.config.languages.target_region or "India"
    centroids = character_centroids(s)
    registry = s.ctx.registry
    model_repo = registry.get(spec.model).weights.hf_repo
    edit_repo = registry.get(spec.edit_model).weights.hf_repo
    base_req = {"width": spec.width, "height": spec.height, "steps": spec.steps, "edit_steps": spec.edit_steps,
                "true_cfg_scale": spec.true_cfg_scale}

    # ---- consenting cast: photos → reference embeddings -------------------------------------
    cast = [m.model_dump() for m in spec.cast]
    targets = casting.resolve_cast_targets(cast, designs)
    for member in cast:
        if member["name"] not in {m["name"] for m in targets.values()}:
            s.warn(f"cast member {member['name']}: no character matches '{member['character']}'")
    photo_paths: dict[str, Path] = {}
    photo_faces: dict[str, str] = {}
    for member in cast:
        for i, photo in enumerate(member["photos"]):
            path = Path(photo["path"])
            path = path if path.is_absolute() else s.loaded.project_dir / path
            if not path.exists():
                raise StageFailure(f"cast photo for {member['name']} not found: {path}")
            photo_paths[f"cast|{member['name']}|{i}"] = path
            photo_faces[f"cast|{member['name']}|{i}"] = photo["face"]
    photo_metrics = identity_embeddings(s, photo_paths, faces=photo_faces, full=True) if photo_paths else {}
    cast_emb: dict[str, list[float] | None] = {}
    for member in cast:
        keys = [k for k in photo_paths if k.split("|")[1] == member["name"]]
        auto = {k: photo_metrics[k].get("all_faces") or [] for k in keys if photo_faces[k] == "auto"}
        if auto:  # group photos: keep the face that matches this person's other photos
            picked = casting.resolve_auto_faces(auto)
            for k, face in picked.items():
                photo_metrics[k]["embedding"] = face["embedding"] if face else None
            multi = [photo_paths[k].name for k in auto if len(auto[k]) > 1]
            if multi:
                s.provenance.setdefault("cast_group_photos", {})[member["name"]] = multi
        bad = [photo_paths[k].name for k in keys if photo_metrics[k]["embedding"] is None]
        if bad:
            s.warn(f"cast {member['name']}: no face found in {bad}")
        cast_emb[member["name"]] = casting.mean_embedding([photo_metrics[k]["embedding"] for k in keys])
        if cast_emb[member["name"]] is None:
            raise StageFailure(f"cast {member['name']}: no usable face in any photo")
    photo_sha = {k: sha256_file(v) for k, v in photo_paths.items()}
    s.provenance["cast"] = [{"name": m["name"], "character": m["character"], "mode": m["mode"], "consent": m["consent"],
                             "photos": len(m["photos"]), "note": "photos are not stored in artifacts; only derived faces"}
                            for m in cast]

    # ---- candidates (and blends) per character -------------------------------------------------
    pending = {d["gen_character_id"]: d for d in designs}
    strength = {gid: "clear" for gid in targets}
    scored_all: dict[str, list[dict]] = {gid: [] for gid in pending}
    chosen: dict[str, dict] = {}
    for attempt in range(MAX_ATTEMPTS):
        if not pending:
            break
        jobs, items = [], {}
        for n, (gid, d) in enumerate(pending.items()):
            member = targets.get(gid)
            appearance = spec.appearance.model_dump() if spec.appearance.apply_to == "all" or (
                spec.appearance.apply_to == "leads" and gid in targets) else None
            prompt = casting.face_prompt(d, identity.get(d["source_character_id"]), region, spec.style_suffix, appearance)
            for k in range(spec.candidates):
                seed = spec.seed + 1000 * n + 7919 * attempt + 104729 * k
                cand = s.item("face_candidate", {"prompt": prompt, "negative": spec.negative, "seed": seed,
                                                 "model": spec.model, **base_req}, f"{gid}:cand:{attempt}:{k}")
                job = {"id": f"{gid}#{attempt}.{k}", "prompt": prompt, "negative": spec.negative, "seed": seed,
                       "out": str(cand.output("candidate.png") if cand.done else cand.path("candidate.png")),
                       "skip_candidate": cand.done}
                blend_item = None
                if member is not None:
                    keys = [x for x in photo_paths if x.split("|")[1] == member["name"]]
                    best_keys = casting.best_photos(keys, photo_metrics, 2)
                    bprompt = casting.blend_prompt(len(best_keys), member["mode"], strength[gid], appearance=appearance)
                    blend_item = s.item("face_blend", {"candidate": {"prompt": prompt, "seed": seed, "model": spec.model},
                                                       "photos": [photo_sha[x] for x in best_keys], "prompt": bprompt,
                                                       "seed": seed + 1, "edit_model": spec.edit_model, **base_req},
                                        f"{gid}:blend:{attempt}:{k}")
                    if not blend_item.done:
                        job["blend"] = {"photos": [str(photo_paths[x]) for x in best_keys], "prompt": bprompt,
                                        "seed": seed + 1, "out": str(blend_item.path("blend.png"))}
                if not (cand.done and (blend_item is None or blend_item.done)):
                    jobs.append(job)
                items[job["id"]] = (gid, cand, blend_item)
        if jobs:
            result = s.runtime.call_worker("image", "images", {"mode": "candidates", "model_repo": model_repo,
                                                               "edit_repo": edit_repo, "jobs": jobs, **base_req},
                                           workdir=s.workdir / f"attempt{attempt}", tag=f"cand{attempt}")["results"]
            for job in jobs:
                gid, cand, blend_item = items[job["id"]]
                r = result.get(job["id"], {})
                if r.get("candidate") and not cand.done:
                    cand.finish({"candidate.png": Path(r["candidate"])}, {"seed": job["seed"]})
                if blend_item is not None and r.get("blend") and not blend_item.done:
                    blend_item.finish({"blend.png": Path(r["blend"])}, {"errors": r.get("errors", [])})
                for err in r.get("errors", []):
                    s.warn(f"{job['id']}: {err}")
        # measure every image of this attempt
        images = {}
        for jid, (gid, cand, blend_item) in items.items():
            if blend_item is not None:
                if blend_item.done:
                    images[jid] = blend_item.output("blend.png")
            elif cand.done:
                images[jid] = cand.output("candidate.png")
        measured = identity_embeddings(s, images, full=True)
        retry = {}
        for gid, d in pending.items():
            member = targets.get(gid)
            others = {name: emb for name, emb in cast_emb.items() if not member or name != member["name"]}
            batch = []
            for jid in sorted(j for j in items if items[j][0] == gid):
                if jid not in measured:
                    batch.append({"id": jid, "path": None, "valid": False, "score": -9.0, "reasons": ["generation failed"],
                                  "metrics": {}})
                    continue
                sc = casting.score_candidate(measured[jid], source=centroids.get(d["source_character_id"]),
                                             cast_emb=cast_emb.get(member["name"]) if member else None,
                                             cast_member=member, other_cast=others)
                batch.append({"id": jid, "path": str(images[jid]), **sc, "embedding": measured[jid]["embedding"]})
            scored_all[gid] += batch
            pick = spec.pick.get(gid)
            index = casting.choose(scored_all[gid], pick)
            if index is not None and scored_all[gid][index].get("path") and (scored_all[gid][index]["valid"] or pick is not None):
                chosen[gid] = scored_all[gid][index]
            else:
                if member is not None and member["mode"] == "blend":
                    strength[gid] = casting.next_strength(batch, member, strength[gid])
                reasons = sorted({r for b in batch for r in b["reasons"]})
                s.warn(f"{gid} attempt {attempt + 1}: no acceptable face ({'; '.join(reasons)[:200]})")
                retry[gid] = d
        pending = retry

    # ---- pose / expression sheet from the winners ---------------------------------------------
    var_jobs, var_items = [], {}
    for gid, win in chosen.items():
        master = Path(win["path"])
        look = spec.appearance
        body = f" Same {look.build}." if look.apply_to == "all" or (look.apply_to == "leads" and gid in targets) else ""
        prompts = {k: f"{v} Keep the exact same face, identity, hairstyle and outfit.{body} Photorealistic, natural skin."
                   for k, v in spec.variants.items()}
        item = s.item("face_variants", {"master": sha256_file(master), "variants": prompts,
                                        "edit_model": spec.edit_model, **base_req}, f"{gid}:variants")
        var_items[gid] = (item, prompts)
        if not item.done:
            var_jobs.append({"id": gid, "master": str(master), "seed": spec.seed,
                             "variants": {k: {"prompt": p, "out": str(item.path(f"{k}.png"))} for k, p in prompts.items()}})
    if var_jobs:
        res = s.runtime.call_worker("image", "images", {"mode": "variants", "edit_repo": edit_repo, "jobs": var_jobs,
                                                        **base_req}, workdir=s.workdir / "variants", tag="variants")["results"]
        for job in var_jobs:
            r = res.get(job["id"], {})
            if r.get("variants"):
                var_items[job["id"]][0].finish({f"{k}.png": Path(v) for k, v in r["variants"].items()})
            for err in r.get("errors", []):
                s.warn(f"{job['id']}: {err}")
    variant_paths = {}
    for gid, (item, prompts) in var_items.items():
        variant_paths[gid] = {k: item.output(f"{k}.png") for k in prompts if item.done and _has(item, f"{k}.png")}
    sheet_emb = identity_embeddings(s, {f"{gid}|{k}": p for gid, v in variant_paths.items() for k, p in v.items()})

    references = {}
    for gid, win in chosen.items():
        d = next(x for x in designs if x["gen_character_id"] == gid)
        who = identity.get(d["source_character_id"], {})
        master_ref = s.put_file(Path(win["path"]), f"{gid}/master.png")
        variants, consistency = {}, {}
        for k, path in variant_paths.get(gid, {}).items():
            emb = sheet_emb.get(f"{gid}|{k}")
            sim = round(float(np.dot(win["embedding"], emb)), 4) if emb is not None else None
            consistency[k] = sim
            if sim is not None and sim < 0.5:
                s.warn(f"{gid}/{k}: variant identity drift (cos={sim}); excluded from the sheet")
                continue
            variants[k] = s.put_file(path, f"{gid}/{k}.png").uri
        member = targets.get(gid)
        references[gid] = {
            "source_character_id": d["source_character_id"], "name": who.get("full_name") or d["name"],
            "master": master_ref.uri, "variants": variants, "master_embedding": win["embedding"],
            "source_centroid": centroids.get(d["source_character_id"]),
            "source_similarity": win["metrics"].get("source_similarity"), "variant_identity": consistency,
            "cast": ({"name": member["name"], "mode": member["mode"], "similarity": win["metrics"].get("cast_similarity")}
                     if member else None),
            "chosen": win["id"], "chosen_by": "manual" if gid in spec.pick else "automatic",
            "candidates": [{k: v for k, v in c.items() if k not in ("embedding", "path")} for c in scored_all[gid]],
        }
    rows = [{"id": gid, "candidates": scored_all[gid],
             "chosen": next((i for i, c in enumerate(scored_all[gid]) if gid in chosen and c["id"] == chosen[gid]["id"]), None)}
            for gid in scored_all]
    sheet = casting.contact_sheet(rows, str(s.workdir / "contact_sheet.jpg"))
    if sheet:
        s.put_file(Path(sheet), "contact_sheet.jpg")
    for gid in pending:
        s.warn(f"{gid}: no acceptable face after {MAX_ATTEMPTS} attempts; its shots will fall back")
    s.put_json("character_references.json", references)
    s.metric("characters_with_references", len(references))
    s.metric("characters_failed", len(pending))
    s.metric("cast_characters", sorted(gid for gid in references if references[gid]["cast"]))
    s.metric("candidates_generated", sum(len(v) for v in scored_all.values()))
    sims = [v for r in references.values() for v in r["variant_identity"].values() if v is not None]
    if sims:
        s.metric("sheet_identity_mean_cos", round(float(np.mean(sims)), 4), kind="objective")


def _has(item, name: str) -> bool:
    return any(ref.uri and ref.uri.rsplit("/", 1)[-1] == name for ref in (item.record.outputs if item.record else []))


@executor("voice_design")
def voice_design(s: StageContext) -> None:
    spec = s.config.production.voice_design
    designs = {d["source_character_id"]: d for d in s.dep_json("character_design", "character_designs.json")}
    profiles = {p["character_id"]: p for p in s.dep_json("character_profile", "character_profiles.json")}
    mappings = s.dep_json("speaker_mapping", "speaker_mapping.json")
    identity = {c["character_id"]: c for c in (s.dep_json("character_identity", "character_identity.json", required=False)
                                               or {}).get("characters", [])}
    accent_on = s.config.localization.audio.regional_pronunciation
    items, meta = [], {}
    for n, m in enumerate(sorted(mappings, key=lambda m: m["speaker_id"])):
        if m["supporting_speech_s"] <= 0:
            continue
        char = m.get("character_id")
        design = designs.get(char or "")
        gender = norm_gender((identity.get(char or "") or {}).get("gender") or (design or {}).get("gender")
                             or profiles.get(char or "", {}).get("gender"))
        description = (design or {}).get("voice_description")
        if not description or mentions_other_gender(description, gender):  # the voice must match the character
            description = spec.default_description.format(gender=gender or "person")
        who = identity.get(char or "")
        if who and accent_on:  # regional accent + speaking style in the designed voice (Indic Parler-TTS)
            style = who["language_profile"].get("speaking_style", "")
            description = f"{description.rstrip('. ')}. The speaker {who['voice_accent']}" + (f", {style}." if style else ".")
        voice_id = f"VOICE_{m['speaker_id'].removeprefix('SPK_')}"
        out = s.workdir / f"{voice_id}.wav"
        items.append({"id": voice_id, "description": description, "text": spec.reference_text, "out": str(out),
                      "seed": 1000 + n})
        meta[voice_id] = {"speaker_id": m["speaker_id"], "character_id": char,
                          "gen_character_id": (design or {}).get("gen_character_id"), "description": description,
                          "mapping_status": m["status"], "character_name": (who or {}).get("full_name"),
                          "accent": (who or {}).get("language_profile", {}).get("accent")}
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
