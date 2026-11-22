"""Indian story adaptation stages: character identity → scene analysis → screenplay (localization).

Order matters: the model first understands each scene and the people in it, then writes the
Indian screenplay for that scene with an audio direction per line, then the result is evaluated
(timing, names, slang, geography, originality, naturalness, screenplay quality) and weak lines are
rewritten. Translation-then-emotion is deliberately avoided.
"""

from __future__ import annotations

from typing import Any

from ..core.log import get_logger
from ..orchestration.runner import StageContext, executor
from ..storytelling import (
    CreativeReferenceManager, DialogueContextAnalyzer, DialogueDurationOptimizer, EmotionAndDeliveryPlanner,
    IndianDialogueRewriter, IndianNameGenerator, LocationDatabase, ModelSelector, NaturalConversationGenerator,
    RegionalEnvironmentAdapter, RegionalLanguageProfileManager, ScreenplayEnhancementEngine,
    ScreenplayQualityEvaluator,
)
from ..storytelling.name_map import SourceNameMap
from ..storytelling.names import name_distance
from .analysis import load_segments
from .story import answer_items, as_dict, batched, call_llm, lang_name

log = get_logger(__name__)


# ---- shared setup ---------------------------------------------------------------------------

def profile_manager(s: StageContext) -> RegionalLanguageProfileManager:
    loc = s.config.localization
    return RegionalLanguageProfileManager.load(loc.regional_profile, loc.dialect.model_dump(), loc.writing_style)


def selector(s: StageContext) -> ModelSelector:
    llm = s.config.production.llm
    defaults = {"analyzer": llm.model, "writer": llm.model, "screenplay_reviewer": llm.model,
                "naturalness_reviewer": llm.polish_model or llm.model}
    try:
        gpu = s.runtime.gpu_memory_gb
    except Exception:  # noqa: BLE001
        gpu = None
    return ModelSelector(s.config.localization.models.model_dump(), defaults, s.ctx.registry.get, gpu)


def region_name(s: StageContext, pm: RegionalLanguageProfileManager) -> str:
    return s.config.localization.character_names.region or pm.city or s.config.languages.target_region or "India"


def speaker_characters(s: StageContext) -> dict[str, str | None]:
    return {m["speaker_id"]: m.get("character_id") for m in s.dep_json("speaker_mapping", "speaker_mapping.json")}


def identity_index(identity: dict) -> dict[str, dict]:
    return {c["character_id"]: c for c in identity.get("characters", [])}


def character_lines(ids: list[str], registry: dict[str, dict], pm: RegionalLanguageProfileManager) -> str:
    rows = []
    for cid in ids:
        c = registry.get(cid)
        if not c:
            continue
        address = "; ".join(f"{f.get('by')}: {f.get('form')}" for f in c.get("forms_of_address", [])[:4])
        rows.append(f"- {cid} = {c['full_name']} ({c.get('devanagari') or ''}), {c.get('gender') or ''} "
                    f"{c.get('age') or ''}, {c.get('profession') or ''}; personality: {c.get('personality')}; speaks: "
                    f"{pm.describe_for_prompt(c['language_profile'])}" + (f"; addressed as: {address}" if address else ""))
    return "\n".join(rows) or "(unknown speakers)"


# ---- 1. character identity -------------------------------------------------------------------

@executor("character_identity")
def character_identity(s: StageContext) -> None:
    loc = s.config.localization
    pm = profile_manager(s)
    region = region_name(s, pm)
    designs = s.dep_json("character_design", "character_designs.json")
    story = s.dep_json("narrative", "narrative.json").get("global", {}) or {}
    roles = {c.get("character_id"): c for c in story.get("characters", []) if isinstance(c, dict)}
    relationships = []
    for cid, role in roles.items():
        for rel in role.get("relationships") or []:
            if isinstance(rel, dict) and rel.get("with"):
                relationships.append({"a": cid, "b": rel["with"], "relation": rel.get("relation", "")})
    characters = [{"character_id": d["source_character_id"], "gender": d.get("gender"), "age": d.get("age"),
                   "tier": d.get("tier"), "role": roles.get(d["source_character_id"], {}).get("role"),
                   "description": roles.get(d["source_character_id"], {}).get("description"),
                   "design_name": d.get("name")} for d in designs]
    gen = IndianNameGenerator(pm.naming(), region, loc.character_names.preserve_relationships,
                              loc.character_names.min_name_distance)
    rel_text = "; ".join(f"{r['a']}–{r['b']}: {r['relation']}" for r in relationships)
    profile_text = f"{pm.dialect_info().get('description', '')} (project default intensity {loc.dialect.intensity})"

    registry: list[dict] = []
    issues: list[str] = []
    if loc.character_names.mode == "culturally_adaptive" and characters:
        model = selector(s).pick("writer")
        avoid: list[str] = []
        for attempt in range(2):
            prompt = gen.prompt(characters, story.get("synopsis", ""), rel_text, profile_text, loc.writing_style, avoid)
            result = call_llm(s, [prompt], model_id=model, tag=f"identity{attempt}", max_tokens=loc.max_tokens)
            body = result.get("identity")
            items = body.get("characters", []) if isinstance(body, dict) else (body if isinstance(body, list) else [])
            raw = {str(x.get("character_id")).strip(): x for x in items if isinstance(x, dict) and x.get("character_id")}
            registry = [{**gen.normalize(raw.get(c["character_id"]), c["character_id"], c["design_name"] or c["character_id"]),
                         "gender": c["gender"], "age": c["age"], "tier": c["tier"]} for c in characters]
            for entry in registry:
                entry["language_profile"] = pm.normalize_character_profile(raw.get(entry["character_id"], {}).get("language_profile"))
            issues = gen.validate(registry, relationships)
            if not issues:
                break
            avoid = sorted({r["given_name"] for r in registry if any(r["given_name"] in i for i in issues)})
            s.warn(f"identity attempt {attempt + 1}: {issues[:4]}")
    else:
        registry = [{**gen.normalize({"given_name": c["design_name"]}, c["character_id"], c["design_name"] or c["character_id"]),
                     "gender": c["gender"], "age": c["age"], "tier": c["tier"],
                     "language_profile": pm.default_character_profile()} for c in characters]
    for entry in registry:
        entry.pop("language_profile_raw", None)
        entry["voice_accent"] = pm.voice_accent(entry["language_profile"])
    accents = {(r["language_profile"]["accent"], round(r["language_profile"]["dialect_intensity"], 1)) for r in registry}
    if len(registry) > 1 and len(accents) < 2:
        s.warn("all characters share one speaking profile; the trailer cannot show two different accents")
    for issue in issues:
        s.warn(f"identity: {issue}")
    s.put_json("character_identity.json", {"region": region, "regional_profile": pm.id, "writing_style": loc.writing_style,
                                           "characters": registry, "relationships": relationships, "issues": issues})
    s.metric("characters", len(registry))
    s.metric("distinct_profiles", len(accents))
    s.metric("name_issues", len(issues))


# ---- 2. scene analysis ---------------------------------------------------------------------------

@executor("scene_analysis")
def scene_analysis(s: StageContext) -> None:
    loc = s.config.localization
    pm = profile_manager(s)
    region = region_name(s, pm)
    segments = load_segments(s)
    story = s.dep_json("narrative", "narrative.json")
    emotions = s.dep_json("emotion", "emotions.json")
    identity = s.dep_json("character_identity", "character_identity.json")
    registry = identity_index(identity)
    spk_char = speaker_characters(s)
    chunk_of = {sid: c["chunk_id"] for c in story["chunks"] for sid in c["segment_ids"]}
    chunk_summary = {c["chunk_id"]: c.get("summary", "") for c in story["chunks"]}
    scenes = group_scenes_for(segments, chunk_of, loc)
    db = LocationDatabase(pm.profile)
    env = RegionalEnvironmentAdapter(db, enabled=loc.cultural_adaptation)
    analyzer = DialogueContextAnalyzer(lang_name(s.config.languages.source), env.plan_schema(),
                                       db.for_prompt() if env.enabled else "", region)
    by_id = {seg["segment_id"]: seg for seg in segments}
    prompts, previous = [], ""
    for scene in scenes:
        chars = sorted({spk_char.get(by_id[sid].get("speaker_id") or "") for sid in scene["segment_ids"]} - {None, ""})
        rows = []
        for sid in scene["segment_ids"]:
            seg = by_id[sid]
            cid = spk_char.get(seg.get("speaker_id") or "")
            who = f"{cid} ({registry[cid]['display_name']})" if cid in registry else (seg.get("speaker_id") or "UNKNOWN")
            rows.append(f"{sid} | {who} | {seg['end_s'] - seg['start_s']:.1f}s | "
                        f"{emotions.get(sid, {}).get('emotion', 'unknown')} | {seg['text']}")
        prompt = analyzer.prompt(scene, rows, character_lines(chars, registry, pm), story.get("global", {}).get("synopsis", ""),
                                 chunk_summary.get(scene["chunk_id"], ""), previous)
        prompt["user"] += (' Also list names the original lines use: "names_mentioned": [{"source": str, '
                           '"romanized": str, "character_id": id|null}].')
        prompts.append(prompt)
        previous = chunk_summary.get(scene["chunk_id"], "")[:200]
    results = call_llm(s, prompts, model_id=selector(s).pick("analyzer"), tag="scenes", max_tokens=loc.max_tokens) \
        if prompts else {}
    engine = ScreenplayEnhancementEngine(loc.storytelling.model_dump())
    out, failed = [], 0
    for scene in scenes:
        raw = as_dict(results.get(scene["scene_id"]))
        analysed = DialogueContextAnalyzer.normalize(raw, scene)
        analysed["location_plan"] = env.validate_plan(analysed.pop("location_plan_raw"))
        analysed["enhancement_plan"] = engine.plan(analysed)
        analysed["names_mentioned"] = [n for n in raw.get("names_mentioned") or [] if isinstance(n, dict)][:10]
        failed += analysed["status"] != "ok"
        out.append(analysed)
    if failed:
        s.warn(f"{failed}/{len(scenes)} scenes could not be analysed (written with less context)")
    s.put_json("scenes.json", {"region": region, "scenes": out})
    s.metric("scenes", len(out))
    s.metric("scenes_failed", failed)
    moods: dict[str, int] = {}
    for x in out:
        moods[x["mood"]] = moods.get(x["mood"], 0) + 1
    s.metric("moods", moods)
    s.metric("scenes_with_real_places", sum(bool(env.allowed_places(x["location_plan"])) for x in out))


def group_scenes_for(segments: list[dict], chunk_of: dict[str, str], loc) -> list[dict]:
    from ..storytelling.context import group_scenes

    return group_scenes(segments, chunk_of, loc.storytelling.scene_gap_seconds, loc.storytelling.scene_max_lines)


# ---- 3. screenplay adaptation + evaluation ------------------------------------------------------

@executor("localization")
def localization(s: StageContext) -> None:
    loc = s.config.localization
    q = loc.quality
    pm = profile_manager(s)
    region = region_name(s, pm)
    segments = {seg["segment_id"]: seg for seg in load_segments(s)}
    emotions = s.dep_json("emotion", "emotions.json")
    identity = s.dep_json("character_identity", "character_identity.json")
    registry = identity_index(identity)
    scenes = s.dep_json("scene_analysis", "scenes.json")["scenes"]
    spk_char = speaker_characters(s)
    designs = s.dep_json("character_design", "character_designs.json")
    tgt = lang_name(s.config.languages.target)
    sel = selector(s)
    models = {r: sel.pick(r) for r in ("writer", "naturalness_reviewer", "screenplay_reviewer")}

    db = LocationDatabase(pm.profile)
    env = RegionalEnvironmentAdapter(db, enabled=loc.cultural_adaptation)
    refs = CreativeReferenceManager(loc.creative_references.model_dump(), s.loaded.project_dir)
    duration = DialogueDurationOptimizer(q.syllables_per_second, q.max_speedup, q.min_fill)
    planner = EmotionAndDeliveryPlanner(loc.audio.model_dump(), style=loc.writing_style)
    conversation = NaturalConversationGenerator()
    writer = IndianDialogueRewriter(target=tgt, region=region, style=loc.writing_style, register=loc.speech_register,
                                    conversation=conversation, direction_schema=planner.schema(),
                                    originality_rule=refs.rule())
    name_map = SourceNameMap(scenes, registry)
    examples = {}
    for scene in scenes:
        for sid in scene["segment_ids"]:
            for e in name_map.unresolved():
                key = e["source"] or e["romanized"]
                if key not in examples and key in segments[sid]["text"]:
                    examples[key] = segments[sid]["text"]
    names_prompt = name_map.prompt(region, pm.naming(), examples)
    if names_prompt:
        answer = call_llm(s, [names_prompt], model_id=models["writer"], tag="names", max_tokens=loc.max_tokens)
        for problem in name_map.accept(answer.get("source_names"), loc.character_names.min_name_distance):
            s.warn(f"source names: {problem}")
    s.put_json("name_map.json", name_map.report())
    s.metric("source_names", len(name_map.entries))
    s.metric("source_names_unmapped", len(name_map.unresolved()))
    old_names = [d.get("name") for d in designs] + name_map.old_names()
    registry_names = [c["given_name"] for c in registry.values()] + [c.get("family_name") or "" for c in registry.values()]
    old_names = [n for n in old_names if n and all(name_distance(n, r) > 0.05 for r in registry_names if r)]
    evaluator = ScreenplayQualityEvaluator(min_score=q.min_score, duration=duration, environment=env, references=refs,
                                           profiles=pm, conversation=conversation, old_names=old_names)
    technique = refs.technique_guide()

    lines: dict[str, dict] = {}
    scene_of: dict[str, dict] = {}
    versions: dict[str, list[dict]] = {}

    def speaker_of(sid: str) -> str | None:
        return spk_char.get(segments[sid].get("speaker_id") or "")

    def name_of(cid: str | None, sid: str) -> str:
        return registry[cid]["display_name"] if cid in registry else (segments[sid].get("speaker_id") or "someone")

    def with_names(characters: str, sids: list[str]) -> str:
        note = name_map.note([segments[x]["text"] for x in sids])
        return f"{characters}\n{note}" if note else characters

    def writer_prompts(targets: dict[str, list[str]], feedback: dict[str, str], tag: str) -> list[dict]:
        prompts = []
        small = max(3, loc.batch_lines // 3)
        for scene in scenes:
            sids = targets.get(scene["scene_id"])
            if not sids:
                continue
            chars = sorted({speaker_of(x) for x in scene["segment_ids"]} - {None})
            intensity = max([registry[c]["language_profile"]["dialect_intensity"] for c in chars if c in registry] or [0.3])
            plan = scene["location_plan"]
            context = scene.get("line_context", {})

            def row(sid: str) -> str:
                seg, ctx = segments[sid], context.get(sid, {})
                cid = speaker_of(sid)
                listener = ctx.get("listener")
                listener_name = registry[listener]["display_name"] if listener in registry else (listener or "?")
                avail = seg["end_s"] - seg["start_s"]
                b = duration.budget(avail)
                return (f"{sid} | {name_of(cid, sid)} → {listener_name} | {avail:.1f}s | {b['target']}/{b['max']} | "
                        f"{ctx.get('intention') or ''} | {ctx.get('emotion') or emotions.get(sid, {}).get('emotion', '')} | "
                        f"{name_map.apply(seg['text'])}")

            def build(part: list[str], pid: str) -> dict:
                first = scene["segment_ids"].index(part[0])
                before = [f"{name_of(speaker_of(x), x)}: {lines[x]['target_text']}" if x in lines and lines[x].get("target_text")
                          else f"{name_of(speaker_of(x), x)}: ({name_map.apply(segments[x]['text'])})"
                          for x in scene["segment_ids"][max(0, first - 4):first]]
                fb = "\n".join(f"{x}: {feedback[x]}" for x in part if x in feedback)
                return writer.prompt(pid, scene=scene, enhancement=scene["enhancement_plan"]["instructions"] +
                                     ("\n" + technique if technique else ""),
                                     characters=with_names(character_lines(chars, registry, pm), part),
                                     expressions=pm.expressions_for_prompt(intensity),
                                     places=env.allowed_places(plan), fictional=plan.get("fictional_places", []),
                                     lines=[row(x) for x in part], previous_lines=before, feedback=fb)

            for j, part in enumerate(batched(sids, loc.batch_lines)):
                pid = f"{scene['scene_id']}_{tag}{j}"
                prompt = build(part, pid)
                if len(part) > small:
                    prompt["split"] = [build(sub, f"{pid}_s{k}") for k, sub in enumerate(batched(part, small))]
                prompts.append(prompt)
        return prompts

    def accept(sid: str, x: dict, source: str, note: str = "") -> None:
        seg = segments[sid]
        cid = speaker_of(sid)
        prof = registry.get(cid or "", {}).get("language_profile")
        old = lines.get(sid, {})
        direction = planner.normalize(x.get("direction") or old.get("direction"), character_id=cid,
                                      accent=(prof or {}).get("accent"),
                                      fallback_emotion=emotions.get(sid, {}).get("emotion", "neutral"))
        line = {
            "segment_id": sid, "scene_id": scene_of[sid]["scene_id"], "start_s": seg["start_s"], "end_s": seg["end_s"],
            "target_duration_s": round(seg["end_s"] - seg["start_s"], 3), "speaker_id": seg.get("speaker_id"),
            "character_id": cid, "character_name": name_of(cid, sid), "source_text": seg["text"],
            "target_text": writer.clean(x.get("text")), "roman_text": writer.clean(x.get("roman") or old.get("roman_text")),
            "short_text": writer.clean(x.get("short") or old.get("short_text")),
            "meaning_en": writer.clean(x.get("meaning_en") or old.get("meaning_en")),
            "emotion": direction["emotion"], "delivery": direction["delivery"], "direction": direction,
            "target_language": s.config.languages.target,
        }
        line["status"] = "ok" if line["target_text"] else "missing"
        line["issues"] = evaluator.line_issues(line, prof, scene_of[sid]["location_plan"], registry_names)
        line["review"] = old.get("review")
        lines[sid] = line
        versions.setdefault(sid, []).append({"version": len(versions.get(sid, [])) + 1, "source": source,
                                             "text": line["target_text"], "roman": line["roman_text"],
                                             "issues": [i["flag"] for i in line["issues"]], "note": note[:200]})

    def better(sid: str, candidate: dict, source: str, note: str) -> bool:
        """Accept a rewrite only if it has no more hard issues than the current line."""
        current = lines.get(sid)
        backup = dict(current) if current else None
        backup_versions = list(versions.get(sid, []))
        accept(sid, {**(current or {}), "text": candidate.get("text"), "roman": candidate.get("roman"),
                     "short": candidate.get("short") or (current or {}).get("short_text"),
                     "direction": candidate.get("direction") or (current or {}).get("direction"),
                     "meaning_en": (current or {}).get("meaning_en")}, source, note)
        new_hard = sum(i["flag"] != "too_short" for i in lines[sid]["issues"])
        old_hard = sum(i["flag"] != "too_short" for i in backup["issues"]) if backup else 99
        if backup and new_hard > old_hard:
            lines[sid] = backup
            versions[sid] = backup_versions + [{**versions[sid][-1], "rejected": True}]
            return False
        return True

    # -- pass 1: the writer writes each analysed scene ------------------------------------------
    for scene in scenes:
        for sid in scene["segment_ids"]:
            scene_of[sid] = scene
    targets = {sc["scene_id"]: list(sc["segment_ids"]) for sc in scenes}
    results = call_llm(s, writer_prompts(targets, {}, "w"), model_id=models["writer"], max_tokens=loc.max_tokens)
    got = {str(x["id"]).strip(): x for r in results.values() for x in answer_items(r)}
    for sid in scene_of:
        accept(sid, got.get(sid, {}), "writer")

    # -- pass 2: naturalness review (Indic model) ---------------------------------------------------
    def naturalness(sids: list[str], tag: str) -> int:
        if not models["naturalness_reviewer"] or not sids:
            return 0
        system = evaluator.naturalness_system(tgt, region)
        prompts = []
        for scene in scenes:
            mine = [x for x in scene["segment_ids"] if x in set(sids) and lines[x]["status"] == "ok"]
            for j, part in enumerate(batched(mine, loc.batch_lines)):
                rows = []
                for x in part:
                    ln = lines[x]
                    prof = registry.get(ln["character_id"] or "", {}).get("language_profile") or pm.default_character_profile()
                    rows.append(f"{x} | {ln['character_name']}: {pm.describe_for_prompt(prof)} | {ln['target_duration_s']:.1f}s | "
                                f"{scene['mood']} | {name_map.apply(ln['source_text'])} | {ln['target_text']}")
                p = evaluator.naturalness_prompt(f"{scene['scene_id']}_{tag}{j}", rows,
                                                 with_names(f"Scene: {scene.get('summary', '')}", part))
                p["system"] = system
                prompts.append(p)
        if not prompts:
            return 0
        try:
            res = call_llm(s, prompts, model_id=models["naturalness_reviewer"], tag=tag, max_tokens=loc.max_tokens)
        except Exception as exc:  # noqa: BLE001 - keep the writer's lines
            s.warn(f"naturalness review with {models['naturalness_reviewer']} failed: {str(exc)[:200]}")
            return 0
        reviews: dict[str, dict] = {}
        for r in res.values():
            reviews.update(ScreenplayQualityEvaluator.parse_naturalness(r))
        changed = 0
        for x, rv in reviews.items():
            if x not in lines:
                continue
            lines[x]["review"] = rv
            for flag in rv["flags"]:
                if flag not in {i["flag"] for i in lines[x]["issues"]}:
                    lines[x]["issues"].append({"flag": flag, "detail": "naturalness review"})
            weak = (rv["natural"] is not None and rv["natural"] < q.min_score) or rv["flags"]
            if weak and rv["improved"] and rv["improved"] != lines[x]["target_text"]:
                ok = better(x, {"text": rv["improved"], "roman": rv.get("improved_roman")}, "naturalness",
                            f"natural={rv['natural']} flags={rv['flags']}")
                changed += ok
                lines[x]["review"] = {**rv, "applied": ok}
        return changed

    polished = naturalness(list(lines), "nat")

    # -- pass 3: screenplay review per scene (scores + line fixes) ---------------------------------
    def screenplay_review(tag: str) -> dict[str, dict]:
        if not models["screenplay_reviewer"]:
            return {}
        system = evaluator.screenplay_system(tgt)
        prompts = []
        for scene in scenes:
            rows = [f"{x} | {lines[x]['character_name']} | {lines[x]['target_duration_s']:.1f}s | "
                    f"{name_map.apply(lines[x].get('meaning_en') or segments[x]['text'])} | {lines[x]['target_text']} | "
                    f"{lines[x]['direction']['emotion']}/{lines[x]['direction']['delivery']}" for x in scene["segment_ids"]]
            chars = sorted({speaker_of(x) for x in scene["segment_ids"]} - {None})
            p = evaluator.screenplay_prompt(scene, rows, with_names(character_lines(chars, registry, pm), scene["segment_ids"]), "")
            p["id"] = f"{scene['scene_id']}_{tag}"
            p["system"] = system
            prompts.append(p)
        try:
            res = call_llm(s, prompts, model_id=models["screenplay_reviewer"], tag=tag, max_tokens=loc.max_tokens)
        except Exception as exc:  # noqa: BLE001
            s.warn(f"screenplay review failed: {str(exc)[:200]}")
            return {}
        return {sc["scene_id"]: ScreenplayQualityEvaluator.parse_screenplay(res.get(f"{sc['scene_id']}_{tag}"))
                for sc in scenes}

    reviews = screenplay_review("review")
    fixed = 0
    for scene in scenes:
        for issue in (reviews.get(scene["scene_id"]) or {}).get("issues", []):
            x = str(issue.get("id")).strip()
            if x in lines:
                lines[x]["issues"].append({"flag": issue.get("flag") or "review", "detail": str(issue.get("detail"))[:200]})
                if issue.get("rewrite"):
                    fixed += better(x, {"text": issue["rewrite"], "roman": issue.get("rewrite_roman")}, "screenplay_review",
                                    str(issue.get("detail"))[:200])

    # -- pass 4: bounded rewrite rounds for lines that still fail -----------------------------------
    rewritten = 0
    for round_no in range(q.max_rewrite_rounds):
        todo: dict[str, list[str]] = {}
        feedback: dict[str, str] = {}
        for scene in scenes:
            for x in scene["segment_ids"]:
                if evaluator.line_needs_rewrite(lines[x]):
                    todo.setdefault(scene["scene_id"], []).append(x)
                    feedback[x] = "; ".join(f"{i['flag']}: {i['detail']}" for i in lines[x]["issues"])[:300]
        if not todo:
            break
        log.info("screenplay rewrite round %d: %d lines", round_no + 1, sum(len(v) for v in todo.values()))
        res = call_llm(s, writer_prompts(todo, feedback, f"r{round_no}"), model_id=models["writer"],
                       tag=f"rewrite{round_no}", max_tokens=loc.max_tokens)
        new = {str(x["id"]).strip(): x for r in res.values() for x in answer_items(r)}
        changed = [x for x, cand in new.items() if x in lines and better(x, cand, f"rewrite_{round_no + 1}", feedback.get(x, ""))]
        rewritten += len(changed)
        if q.final_naturalness_check and changed:
            naturalness(changed, f"nat_r{round_no}")

    # -- final scores, flags and outputs ----------------------------------------------------------
    scene_reports = []
    for scene in scenes:
        mine = [lines[x] for x in scene["segment_ids"]]
        for issue in evaluator.scene_issues(mine):
            mine[0].setdefault("scene_issues", []).append(issue)
        combined = evaluator.combine((reviews.get(scene["scene_id"]) or {}).get("scores", {}), evaluator.deterministic_scores(mine))
        scene_reports.append({"scene_id": scene["scene_id"], "mood": scene["mood"], "summary": scene.get("summary"),
                              "start_s": scene["start_s"], "end_s": scene["end_s"], **combined,
                              "review_summary": (reviews.get(scene["scene_id"]) or {}).get("summary", ""),
                              "flags": sorted({i["flag"] for ln in mine for i in ln["issues"]})})
    ordered = [lines[x] for x in sorted(lines, key=lambda k: lines[k]["start_s"])]
    for ln in ordered:
        ln["quality"] = "below_threshold" if evaluator.line_needs_rewrite(ln) else "ok"
        ln["versions"] = len(versions.get(ln["segment_id"], []))
    missing = [ln["segment_id"] for ln in ordered if ln["status"] != "ok"]
    below = [ln["segment_id"] for ln in ordered if ln["quality"] == "below_threshold"]
    overall = [r["overall"] for r in scene_reports if r["overall"] is not None]
    s.put_json("localized_lines.json", ordered)
    s.put_json("screenplay_quality.json", {
        "overall": int(sum(overall) / len(overall)) if overall else None, "min_score": q.min_score,
        "models": models, "scenes": scene_reports, "lines_below_threshold": below, "lines_missing": missing,
        "note": "model scores are subjective estimates; timing, names, geography, slang and originality are measured. "
                "A native-speaker review of the trailer is the real gate."})
    s.put_json("screenplay_versions.json", versions)
    s.put_json("creative_references.json", {"sources": refs.provenance(), "rules": refs.rule()})
    s.provenance["models"] = models
    s.metric("lines", len(ordered))
    s.metric("missing", len(missing))
    s.metric("below_threshold", len(below))
    s.metric("naturalness_fixed", polished)
    s.metric("review_fixed", fixed)
    s.metric("rewritten", rewritten)
    s.metric("screenplay_score", int(sum(overall) / len(overall)) if overall else None)
    if missing:
        s.warn(f"{len(missing)} lines have no adaptation (kept silent and reported): {missing[:10]}")
    if below:
        s.warn(f"{len(below)} lines still below quality threshold after rewrites (flagged for human review): {below[:10]}")
