"""Story understanding and character design (LLM stages). Screenplay adaptation lives in adaptation.py."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..orchestration.runner import StageContext, executor
from ..storytelling.gender import mentions_other_gender, norm_gender
from .analysis import load_segments

LANGUAGE_NAMES = {
    "hi": "Hindi", "zh": "Chinese", "en": "English", "te": "Telugu", "ta": "Tamil", "bn": "Bengali",
    "mr": "Marathi", "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam", "pa": "Punjabi", "ja": "Japanese",
    "ko": "Korean", "es": "Spanish", "fr": "French",
}


def lang_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code.split("-")[0], code)


def call_llm(s: StageContext, prompts: list[dict[str, str]], *, model_id: str | None = None,
             tag: str | None = None, max_tokens: int | None = None) -> dict[str, Any]:
    llm = s.config.production.llm
    record = s.ctx.registry.get(model_id or llm.model)
    response = s.runtime.call_worker("llm", "llm", {
        "hf_repo": record.weights.hf_repo, "prompts": prompts, "backend": llm.backend,
        "max_model_len": llm.max_model_len, "max_tokens": max_tokens or llm.max_tokens, "temperature": llm.temperature,
        "tensor_parallel_size": llm.tensor_parallel_size, "gpu_memory_utilization": llm.gpu_memory_utilization,
        "retries": 2, "answers_path": str(s.workdir / f"llm{'.' + tag if tag else ''}.answers.jsonl"),
    }, workdir=s.workdir, tag=tag)
    stats = response.get("stats") or {}
    if stats:
        prefix = f"llm_{tag}_" if tag else "llm_"
        for key in ("prompts", "truncated", "split", "salvaged", "failed", "empty"):
            s.metric(prefix + key, stats.get(key, 0))
    failed = [k for k, v in response["results"].items() if v is None]
    if failed:
        s.warn(f"LLM returned unusable output for {len(failed)} prompt(s): {failed[:10]}")
    return response["results"]


def speaker_labels(mappings: list[dict]) -> dict[str, str]:
    return {m["speaker_id"]: (m["character_id"] or m["speaker_id"]) for m in mappings}


def chunk_segments(segments: list[dict], size: int) -> list[list[dict]]:
    return [segments[i : i + size] for i in range(0, len(segments), size)]


def _fmt_line(seg: dict, labels: dict[str, str]) -> str:
    who = labels.get(seg.get("speaker_id") or "", seg.get("speaker_id") or "UNKNOWN")
    return f"[{seg['segment_id']} @{seg['start_s']:.1f}s {who}] {seg['text']}"


# ---------------------------------------------------------------------------


@executor("narrative")
def narrative(s: StageContext) -> None:
    segments = load_segments(s)
    labels = speaker_labels(s.dep_json("speaker_mapping", "speaker_mapping.json"))
    profiles = s.dep_json("character_profile", "character_profiles.json")
    chunks = chunk_segments(segments, s.config.production.llm.chunk_segments)
    src = lang_name(s.config.languages.source)
    system = (
        f"You are a professional script analyst. You read {src} dialogue transcripts of a drama and summarise "
        "them faithfully in English. Never invent events. Character ids like SRC_CHAR_0003 are on-screen "
        "people; SPK_xx are unidentified voices. Answer in JSON only."
    )
    prompts = []
    for i, chunk in enumerate(chunks):
        user = (
            "Transcript part (segment id, time, speaker, text):\n" + "\n".join(_fmt_line(x, labels) for x in chunk) +
            '\n\nReturn JSON: {"summary": str (3-5 sentences), "events": [str], "turning_point": bool, '
            '"spoiler_level": 0|1|2 (2 = reveals a major twist or the ending), "dominant_emotion": str, '
            '"characters": {"<id>": "what they do / their role here"}}'
        )
        prompts.append({"id": f"chunk_{i:04d}", "system": system, "user": user})
    results = call_llm(s, prompts)

    out_chunks = []
    for i, chunk in enumerate(chunks):
        r = as_dict(results.get(f"chunk_{i:04d}"))
        out_chunks.append({
            "chunk_id": f"chunk_{i:04d}", "start_s": chunk[0]["start_s"], "end_s": chunk[-1]["end_s"],
            "segment_ids": [x["segment_id"] for x in chunk], "summary": r.get("summary", ""),
            "events": r.get("events", []), "turning_point": bool(r.get("turning_point", False)),
            "spoiler_level": int(r.get("spoiler_level", 0) or 0), "dominant_emotion": r.get("dominant_emotion"),
            "characters": r.get("characters", {}), "status": "ok" if r else "llm_failed",
        })

    known = [p["character_id"] for p in profiles if p["tier"] != "background"]
    summaries = "\n".join(f"{c['chunk_id']} ({c['start_s']:.0f}s): {c['summary']}" for c in out_chunks if c["summary"])
    profile_text = "\n".join(
        f"{p['character_id']}: tier={p['tier']}, speech={p['speech_s']}s, faces={p['face_count']}, "
        f"gender≈{p.get('gender')}, age≈{p.get('age_estimate')}" for p in profiles if p["tier"] != "background"
    )
    overall = as_dict(call_llm(s, [{
        "id": "global", "system": system,
        "user": (
            f"Chronological part summaries:\n{summaries}\n\nCharacters observed:\n{profile_text}\n\n"
            'Return JSON: {"synopsis": str, "main_conflict": str, "characters": [{"character_id": str, '
            '"role": "protagonist"|"antagonist"|"supporting"|"minor", "description": str, '
            '"relationships": [{"with": str, "relation": str}]}]} '
            f"Only use these character ids: {known}."
        ),
    }]).get("global"))
    s.put_json("narrative.json", {"chunks": out_chunks, "global": overall})
    s.metric("chunks", len(out_chunks))
    s.metric("chunks_failed", sum(c["status"] != "ok" for c in out_chunks))
    s.metric("turning_points", sum(c["turning_point"] for c in out_chunks))
    s.provenance["note"] = "LLM summaries are subjective model output, grounded in transcript segment ids"


@executor("character_design")
def character_design(s: StageContext) -> None:
    profiles = s.dep_json("character_profile", "character_profiles.json")
    story = s.dep_json("narrative", "narrative.json").get("global", {})
    roles = {c.get("character_id"): c for c in story.get("characters", []) if isinstance(c, dict)}
    llm = s.config.production.llm
    to_design = [p for p in profiles if p["tier"] in ("main", "supporting", "recurring_secondary")][: llm.max_characters_to_design]
    region = s.config.languages.target_region or ""
    target = lang_name(s.config.languages.target)
    system = (
        "You are a casting director creating NEW, fictional, photorealistic characters for a "
        f"{target}-language remake for an audience in {region or 'the target region'}. "
        "Never describe or resemble a real, identifiable person or the original actors. Answer in JSON only."
    )
    prompts = []
    for p in to_design:
        role = roles.get(p["character_id"], {})
        prompts.append({"id": p["character_id"], "system": system, "user": (
            f"Source character {p['character_id']}: estimated gender={p.get('gender')}, age≈{p.get('age_estimate')}, "
            f"importance={p['tier']}, role={role.get('role')}, description={role.get('description')}, "
            f"relationships={role.get('relationships')}. Story: {story.get('synopsis', '')[:1500]}\n\n"
            "Design the replacement. Keep the same gender and approximate age. Return JSON: "
            '{"name": str (culturally appropriate first name), "gender": str, "age": int, '
            '"appearance": {"face": str, "skin_tone": str, "hair": str, "build": str}, "wardrobe": str, '
            '"image_prompt": str (one detailed English paragraph for a photorealistic portrait: face, hair, skin, '
            "clothing, expression, lighting; no celebrity names), "
            '"voice_description": str (English, e.g. "A female speaker with a warm, slightly low-pitched voice '
            'speaks at a moderate pace. The recording is very clear and close-sounding.")}'
        )})
    results = call_llm(s, prompts) if prompts else {}
    designs = []
    for n, p in enumerate(to_design, 1):
        d = as_dict(results.get(p["character_id"]))
        source_gender = norm_gender(p.get("gender"))  # measured on the source faces: it wins over the design answer
        gender = source_gender or norm_gender(d.get("gender")) or d.get("gender") or "person"
        if not d:
            s.warn(f"design for {p['character_id']} failed; using a neutral fallback description")
        elif source_gender and norm_gender(d.get("gender")) not in (None, source_gender):
            s.warn(f"design for {p['character_id']} changed the gender ({d.get('gender')}); kept {source_gender}")
            d = {k: v for k, v in d.items() if k not in ("image_prompt", "voice_description")}
        for key in ("image_prompt", "voice_description"):
            if mentions_other_gender(str(d.get(key) or ""), gender):
                s.warn(f"design for {p['character_id']}: {key} describes the other gender; replaced")
                d.pop(key, None)
        designs.append({
            "gen_character_id": "GEN_" + p["character_id"].removeprefix("SRC_"),
            "source_character_id": p["character_id"], "tier": p["tier"],
            "name": d.get("name") or f"Character {n}", "gender": gender, "age": d.get("age") or p.get("age_estimate"),
            "appearance": d.get("appearance", {}), "wardrobe": d.get("wardrobe", ""),
            "image_prompt": d.get("image_prompt") or (
                f"Portrait of a fictional {p.get('age_estimate') or 30}-year-old {gender} from {region or 'India'}, "
                "natural look, everyday clothing."),
            "voice_description": d.get("voice_description") or
                s.config.production.voice_design.default_description.format(gender=gender),
            "status": "ok" if d else "fallback",
        })
    s.put_json("character_designs.json", designs)
    s.metric("designed", len(designs))
    s.metric("fallbacks", sum(d["status"] != "ok" for d in designs))


def as_dict(result: Any) -> dict:
    """An LLM answer as a dict ({} for null, a bare list or other shapes)."""
    if isinstance(result, list) and len(result) == 1 and isinstance(result[0], dict):
        return result[0]
    return result if isinstance(result, dict) else {}


def answer_items(result: Any) -> list[dict]:
    """Line objects from an LLM answer: {"lines": [...]}, a bare list, or {"<any key>": [...]}."""
    if isinstance(result, dict):
        items = result.get("lines")
        if items is None:
            items = next((v for v in result.values() if isinstance(v, list)), [])
            if not items and "id" in result:
                items = [result]
    elif isinstance(result, list):
        items = result
    else:
        items = []
    return [x for x in items if isinstance(x, dict) and x.get("id")]


def batched(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), max(1, size))]


def group_by(items: list[dict], key: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        out[item.get(key)].append(item)
    return out
