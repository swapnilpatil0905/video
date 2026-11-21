"""Story understanding, character design and localization (LLM stages)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..orchestration.runner import StageContext, executor
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
        gender = d.get("gender") or p.get("gender") or "person"
        if not d:
            s.warn(f"design for {p['character_id']} failed; using a neutral fallback description")
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


@executor("localization")
def localization(s: StageContext) -> None:
    segments = load_segments(s)
    story = s.dep_json("narrative", "narrative.json")
    designs = s.dep_json("character_design", "character_designs.json")
    emotions = s.dep_json("emotion", "emotions.json")
    mappings = s.dep_json("speaker_mapping", "speaker_mapping.json")
    speaker_char = {m["speaker_id"]: m["character_id"] for m in mappings}
    names = {d["source_character_id"]: d["name"] for d in designs}
    glossary = "\n".join(f"{src} → {name}" for src, name in names.items())
    loc = s.config.localization
    src, tgt = lang_name(s.config.languages.source), lang_name(s.config.languages.target)
    region = s.config.languages.target_region or ""
    system = (
        f"You are an expert film dialogue writer localizing a {src} drama into natural, {loc.speech_register} {tgt} "
        f"for audiences in {region or 'the target region'}. Preserve meaning, intent, emotion and each "
        f"character's way of speaking. Adapt idioms and cultural references naturally. Avoid stiff, textbook "
        f"{tgt}. Use the given replacement names instead of original names. Write {tgt} in its native script. "
        "Never drop a line. Answer in JSON only."
    )
    chunks = story["chunks"]
    by_id = {seg["segment_id"]: seg for seg in segments}
    batch = s.config.localization.batch_lines

    def make_prompt(i: int, chunk: dict, part: list[str], pid: str) -> dict:
        lines = []
        for sid in part:
            seg = by_id[sid]
            char = speaker_char.get(seg.get("speaker_id") or "")
            who = names.get(char or "", seg.get("speaker_id") or "UNKNOWN")
            emo = emotions.get(sid, {}).get("emotion", "unknown")
            lines.append(f"{sid} | {who} | {seg['end_s'] - seg['start_s']:.1f}s | {emo} | {seg['text']}")
        previous = chunks[i - 1]["summary"] if i else ""
        return {"id": pid, "system": system, "user": (
            f"Story synopsis: {story.get('global', {}).get('synopsis', '')[:1200]}\n"
            f"Previous part: {previous}\nThis part: {chunk['summary']}\nName glossary:\n{glossary}\n\n"
            f"Lines (id | speaker | available seconds | detected emotion | {src} text):\n" + "\n".join(lines) +
            f'\n\nReturn JSON: {{"lines": [{{"id": str, "text": str ({tgt}, natural), "short": str (shorter {tgt} '
            'version that fits the available seconds, same meaning), "emotion": str, "delivery": str (how to say it)}]}. '
            "Include every id exactly once."
        )}

    prompts = []
    small = max(3, batch // 3)
    for i, chunk in enumerate(chunks):
        for j, part in enumerate(batched(chunk["segment_ids"], batch)):
            pid = f"{chunk['chunk_id']}_p{j}"
            prompt = make_prompt(i, chunk, part, pid)
            if len(part) > small:  # if the answer is cut off, the worker re-asks in smaller pieces
                prompt["split"] = [make_prompt(i, chunk, sub, f"{pid}_s{k}") for k, sub in enumerate(batched(part, small))]
            prompts.append(prompt)
    results = call_llm(s, prompts, max_tokens=s.config.localization.max_tokens)
    got = {}
    for r in results.values():
        for x in answer_items(r):
            got.setdefault(str(x["id"]).strip(), x)

    localized, missing = [], []
    for chunk in chunks:
        for sid in chunk["segment_ids"]:
            seg = by_id[sid]
            x = got.get(sid)
            if not x or not str(x.get("text", "")).strip():
                missing.append(sid)
            emotion = (x or {}).get("emotion") or emotions.get(sid, {}).get("emotion", "unknown")
            localized.append({
                "segment_id": sid, "start_s": seg["start_s"], "end_s": seg["end_s"],
                "target_duration_s": round(seg["end_s"] - seg["start_s"], 3),
                "speaker_id": seg.get("speaker_id"), "character_id": speaker_char.get(seg.get("speaker_id") or ""),
                "source_text": seg["text"], "target_text": (x or {}).get("text", "").strip(),
                "short_text": (x or {}).get("short", "").strip(), "emotion": str(emotion).lower(),
                "delivery": (x or {}).get("delivery", ""), "target_language": s.config.languages.target,
                "status": "ok" if x and str(x.get("text", "")).strip() else "missing",
            })
    polish_model = s.config.production.llm.polish_model
    if polish_model:
        localized, polished = polish_lines(s, localized, chunks, names, polish_model, tgt, region)
        s.metric("polished_lines", polished)
        s.provenance["polish_model"] = polish_model
    if missing:
        s.warn(f"{len(missing)} lines have no translation (status=missing, kept silent and reported): {missing[:10]}")
    s.put_json("localized_lines.json", localized)
    s.metric("lines", len(localized))
    s.metric("missing", len(missing))


def polish_lines(s: StageContext, lines: list[dict], chunks: list[dict], names: dict[str, str], model_id: str,
                 tgt: str, region: str) -> tuple[list[dict], int]:
    """Second pass with an Indic-tuned model: make drafts sound like natural spoken dialogue (same meaning)."""
    system = (
        f"You are a native {tgt} screenwriter{' from ' + region if region else ''}. Rewrite draft film dialogue so it "
        f"sounds like natural, everyday spoken {tgt}: idiomatic, emotional, not literal or textbook. Keep the exact "
        "meaning, the speaker's intent and the character names. Keep lines short enough to speak in the available "
        f"time. Write {tgt} in its native script. Answer in JSON only."
    )
    by_chunk: dict[str, list[dict]] = defaultdict(list)
    chunk_of = {sid: c["chunk_id"] for c in chunks for sid in c["segment_ids"]}
    for line in lines:
        if line["status"] == "ok":
            by_chunk[chunk_of.get(line["segment_id"], "chunk_x")].append(line)
    prompts = []
    batch = s.config.localization.batch_lines
    for chunk_id, all_items in by_chunk.items():
        for j, items in enumerate(batched(all_items, batch)):
            rows = "\n".join(f"{x['segment_id']} | {x['target_duration_s']:.1f}s | {x['emotion']} | source: {x['source_text']} | "
                             f"draft: {x['target_text']} | short draft: {x['short_text']}" for x in items)
            prompts.append({"id": f"{chunk_id}_p{j}", "system": system, "user": (
                f"Character names: {', '.join(names.values())}\n\nLines (id | seconds | emotion | source | draft | short draft):\n"
                f"{rows}\n\nReturn JSON: {{\"lines\": [{{\"id\": str, \"text\": str, \"short\": str}}]}} with every id exactly once."
            )})
    try:
        results = call_llm(s, prompts, model_id=model_id, tag="polish",
                           max_tokens=s.config.localization.max_tokens)
    except Exception as exc:  # noqa: BLE001 - keep the drafts
        s.warn(f"dialogue polish with {model_id} failed; keeping the first-pass translation: {str(exc)[:200]}")
        return lines, 0
    improved = {}
    for result in results.values():
        for item in answer_items(result):
            if str(item.get("text", "")).strip():
                improved[str(item["id"]).strip()] = item
    count = 0
    for line in lines:
        item = improved.get(line["segment_id"])
        if item:
            line["draft_text"] = line["target_text"]
            line["target_text"] = str(item["text"]).strip()
            line["short_text"] = str(item.get("short") or line["short_text"]).strip()
            line["polished_by"] = model_id
            count += 1
    return lines, count


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
