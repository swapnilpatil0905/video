"""DialogueContextAnalyzer: understand each scene BEFORE any dialogue is written.

Scenes are runs of dialogue without a long silence (and within one narrative chunk). For each
scene the analyzer model returns where it happens, who is present, the mood, relationships,
each line's intention/listener/emotion/subtext, enhancement opportunities and a location plan.
"""

from __future__ import annotations

from typing import Any

MOODS = ("comedy", "romance", "tragedy", "emotional", "suspense", "thriller", "family_drama", "action",
         "confrontation", "neutral")


def group_scenes(segments: list[dict], chunk_of: dict[str, str], gap_s: float, max_lines: int) -> list[dict]:
    """Split time-ordered dialogue into scenes: a silence > gap_s, a chunk change or the size cap starts a new one."""
    scenes: list[dict] = []
    current: list[dict] = []
    for seg in sorted(segments, key=lambda x: x["start_s"]):
        if current and (seg["start_s"] - current[-1]["end_s"] > gap_s
                        or chunk_of.get(seg["segment_id"]) != chunk_of.get(current[-1]["segment_id"])
                        or len(current) >= max_lines):
            scenes.append(_scene(len(scenes), current, chunk_of))
            current = []
        current.append(seg)
    if current:
        scenes.append(_scene(len(scenes), current, chunk_of))
    return scenes


def _scene(index: int, segs: list[dict], chunk_of: dict[str, str]) -> dict:
    return {"scene_id": f"SCN_{index:04d}", "start_s": segs[0]["start_s"], "end_s": segs[-1]["end_s"],
            "chunk_id": chunk_of.get(segs[0]["segment_id"]), "segment_ids": [x["segment_id"] for x in segs]}


class DialogueContextAnalyzer:
    def __init__(self, source_language: str, location_schema: str, places_text: str, region: str | None) -> None:
        self.source_language = source_language
        self.location_schema = location_schema
        self.places_text = places_text
        self.region = region

    def system(self) -> str:
        return (
            f"You are a script analyst. You read a scene of a {self.source_language} drama and explain what is really "
            "happening: setting, who is present, relationships, each line's intention, listener, emotion and subtext. "
            "Infer emotion from the whole scene and the characters, not from single words. Be concrete and brief. "
            "Answer in JSON only."
        )

    def prompt(self, scene: dict, lines: list[str], characters: str, synopsis: str, chunk_summary: str,
               previous_scene: str) -> dict[str, str]:
        place_block = (f"\nThe remake is set in {self.region}. Real places you may use in the location plan:\n"
                       f"{self.places_text}\nUse a real place only if the setting genuinely matches it; otherwise "
                       "describe the setting generically or mark invented places as fictional.\n") if self.places_text else ""
        return {"id": scene["scene_id"], "system": self.system(), "user": (
            f"Story synopsis: {synopsis[:1200]}\nThis part of the story: {chunk_summary[:600]}\n"
            f"Previous scene: {previous_scene[:400] or '(start)'}\nCharacters (id → remake name, profile):\n{characters}\n"
            f"{place_block}\nScene lines (id | speaker | seconds | detected audio emotion | text):\n" + "\n".join(lines) +
            "\n\nReturn JSON: {"
            '"summary": str, "mood": one of ' + "|".join(MOODS) + ', "secondary_mood": str|null, '
            '"intensity": float 0-1, "present": [character ids], '
            '"relationships": [{"a": id, "b": id, "relation": str, "dynamic": str}], '
            '"lines": [{"id": str, "speaker": id, "listener": id|"group"|null, "intention": str, '
            '"emotion": str (e.g. "angry but controlled", "sad, hiding it"), "intensity": float 0-1, "subtext": str}], '
            '"enhancement": {"comedy": bool, "romance": bool, "emotion": bool, "tragedy": bool, "suspense": bool, '
            '"confrontation": bool, "notes": str (what could be strengthened without changing the story)}, '
            f"{self.location_schema}" + "}"
        )}

    @staticmethod
    def normalize(raw: Any, scene: dict) -> dict[str, Any]:
        raw = raw if isinstance(raw, dict) else {}
        mood = str(raw.get("mood") or "neutral").lower().strip()
        mood = mood if mood in MOODS else next((m for m in MOODS if m in mood), "neutral")
        try:
            intensity = max(0.0, min(1.0, float(raw.get("intensity", 0.5))))
        except (TypeError, ValueError):
            intensity = 0.5
        lines = {}
        for x in raw.get("lines") or []:
            if isinstance(x, dict) and x.get("id"):
                lines[str(x["id"]).strip()] = {k: x.get(k) for k in ("speaker", "listener", "intention", "emotion",
                                                                    "intensity", "subtext")}
        enhancement = raw.get("enhancement") if isinstance(raw.get("enhancement"), dict) else {}
        return {**scene, "summary": str(raw.get("summary") or "")[:500], "mood": mood,
                "secondary_mood": raw.get("secondary_mood"), "intensity": intensity,
                "present": [str(x) for x in raw.get("present") or []],
                "relationships": [r for r in raw.get("relationships") or [] if isinstance(r, dict)][:12],
                "line_context": lines, "enhancement": enhancement, "location_plan_raw": raw.get("location_plan"),
                "status": "ok" if raw else "analysis_failed"}
