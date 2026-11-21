"""Human-readable previews of what each stage produced (debugging aid).

After every stage the runner logs a few "↳" lines into run.log (counts and a couple of sample
items) and writes a fuller report to state/logs/stages/<stage>.txt: metrics, warnings, every
output file with its size, and the first items of every JSON output.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Callable

MAX_TEXT = 160


def _short(value: Any, limit: int = MAX_TEXT) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def shape(payload: Any) -> str:
    """'list[2203]' / 'dict{chunks: list[55], global: dict{6}}'."""
    if isinstance(payload, list):
        return f"list[{len(payload)}]"
    if isinstance(payload, dict):
        parts = []
        for key, value in list(payload.items())[:8]:
            if isinstance(value, list):
                parts.append(f"{key}: list[{len(value)}]")
            elif isinstance(value, dict):
                parts.append(f"{key}: dict{{{len(value)}}}")
            else:
                parts.append(f"{key}: {_short(value, 40)}")
        more = ", …" if len(payload) > 8 else ""
        return "dict{" + ", ".join(parts) + more + "}"
    return type(payload).__name__


def _items(payload: Any) -> list:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, list) and value:
                return value
    return []


# ---- stage-specific previews (a few telling lines each) ------------------------------

def _localized(payload: list, n: int) -> list[str]:
    status = Counter(x.get("status") for x in payload)
    revised = sum(1 for x in payload if x.get("polished_by") or (x.get("versions") or 1) > 1)
    below = sum(1 for x in payload if x.get("quality") == "below_threshold")
    lines = [f"{len(payload)} lines: {dict(status)}, revised {revised}, below threshold {below}"]
    ok = [x for x in payload if x.get("status") == "ok"]
    step = max(1, len(ok) // max(1, n))
    for x in ok[::step][:n]:
        draft = f" (draft: {_short(x['draft_text'], 60)})" if x.get("draft_text") else ""
        d = x.get("direction") or {}
        lines.append(f"{x['segment_id']} {x.get('character_name') or ''} [{d.get('emotion', x.get('emotion'))} "
                     f"{d.get('intensity', '')} {d.get('delivery', '')}] {_short(x.get('source_text'), 40)} → "
                     f"{_short(x.get('target_text'), 70)}" + (f" ({_short(x['roman_text'], 60)})" if x.get("roman_text") else "")
                     + draft + (f" ⚑{[i.get('flag') for i in x.get('issues', [])]}" if x.get("issues") else ""))
    missing = [x["segment_id"] for x in payload if x.get("status") != "ok"][:10]
    if missing:
        lines.append(f"missing: {missing}")
    return lines


def _narrative(payload: dict, n: int) -> list[str]:
    chunks = payload.get("chunks", [])
    failed = sum(1 for c in chunks if c.get("status") != "ok")
    synopsis = (payload.get("global") or {}).get("synopsis", "")
    lines = [f"{len(chunks)} chunks ({failed} failed); synopsis: {_short(synopsis, 200)}"]
    for c in chunks[:n]:
        lines.append(f"{c.get('chunk_id')} {c.get('start_s', 0):.0f}s: {_short(c.get('summary'), 120)}")
    return lines


def _designs(payload: list, n: int) -> list[str]:
    return [f"{len(payload)} characters"] + [
        f"{d.get('source_character_id')} → {d.get('name')}: {_short(d.get('appearance') or d.get('description'), 100)}"
        for d in payload[:n]]


def _speaker_mapping(payload: list, n: int) -> list[str]:
    mapped = [m for m in payload if m.get("character_id")]
    return [f"{len(mapped)}/{len(payload)} speakers mapped"] + [
        f"{m.get('speaker_id')} → {m.get('character_id')} (score {m.get('score', m.get('confidence', '?'))})"
        for m in payload[: n * 2]]


def _transcript(payload: Any, n: int) -> list[str]:
    items = _items(payload)
    return [f"{len(items)} segments"] + [
        f"{x.get('segment_id', x.get('id'))} {float(x.get('start_s', x.get('start', 0))):.1f}s "
        f"{x.get('speaker_id', '')}: {_short(x.get('text'), 100)}" for x in items[:n]]


def _identity(payload: dict, n: int) -> list[str]:
    chars = payload.get("characters", [])
    out = [f"{len(chars)} characters, region {payload.get('region')}, profile {payload.get('regional_profile')}"]
    for c in chars[: n * 2]:
        lp = c.get("language_profile") or {}
        out.append(f"{c.get('character_id')} → {c.get('full_name')} ({c.get('devanagari')}) | {lp.get('accent')} "
                   f"{lp.get('dialect_intensity')} | eng {lp.get('english_mixing')} | {_short(lp.get('speaking_style'), 40)}")
    return out + [f"issue: {i}" for i in payload.get("issues", [])[:3]]


def _scenes(payload: dict, n: int) -> list[str]:
    scenes = payload.get("scenes", [])
    moods = Counter(x.get("mood") for x in scenes)
    places = [x for x in scenes if (x.get("location_plan") or {}).get("place_references_allowed")]
    out = [f"{len(scenes)} scenes; moods {dict(moods)}; {len(places)} with real-place references"]
    for x in scenes[:n]:
        plan = x.get("location_plan") or {}
        out.append(f"{x.get('scene_id')} {x.get('start_s', 0):.0f}s [{x.get('mood')} {x.get('intensity')}] "
                   f"{_short(x.get('summary'), 90)} | at: {_short(plan.get('setting'), 30)} "
                   f"{plan.get('real_place') or ''} | enhance: {(x.get('enhancement_plan') or {}).get('allowed')}")
    return out


def _screenplay_quality(payload: dict, n: int) -> list[str]:
    scenes = payload.get("scenes", [])
    weak = sorted(scenes, key=lambda x: x.get("overall") or 0)[:n]
    out = [f"overall {payload.get('overall')} (min {payload.get('min_score')}); below threshold: "
           f"{len(payload.get('lines_below_threshold', []))} lines; missing: {len(payload.get('lines_missing', []))}; "
           f"models {payload.get('models')}"]
    for x in weak:
        out.append(f"weakest {x.get('scene_id')} [{x.get('mood')}] {x.get('overall')} — low: {x.get('below_threshold')} "
                   f"flags: {x.get('flags')}")
    return out


PREVIEWS: dict[str, Callable[[Any, int], list[str]]] = {
    "character_identity.json": _identity,
    "scenes.json": _scenes,
    "screenplay_quality.json": _screenplay_quality,
    "localized_lines.json": _localized,
    "narrative.json": _narrative,
    "character_designs.json": _designs,
    "speaker_mapping.json": _speaker_mapping,
    "transcript.json": _transcript,
    "transcript_clean.json": _transcript,
}


def generic_preview(payload: Any, n: int) -> list[str]:
    lines = [shape(payload)]
    for item in _items(payload)[:n]:
        lines.append(_short(item))
    return lines


def preview(name: str, payload: Any, n: int) -> list[str]:
    try:
        fn = PREVIEWS.get(name)
        return fn(payload, n) if fn else generic_preview(payload, n)
    except Exception as exc:  # noqa: BLE001 - a preview must never fail a stage
        return [f"(preview failed: {type(exc).__name__}: {exc}) {shape(payload)}"]
