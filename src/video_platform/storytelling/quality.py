"""ScreenplayQualityEvaluator: deterministic checks + two model reviews, scored 0–100 per category.

Line checks (no model): missing, timing, names, slang density, geography, originality, repetition,
script. Naturalness review (Indic model): natural/regional scores, literal-translation and slang
flags, an improved line. Screenplay review (scene model): the 14 category scores and per-line
issues with a rewrite. Lines that fail are rewritten (bounded rounds) and every version is kept.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

CATEGORIES = ("naturalness", "regional_authenticity", "name_usage", "voice_consistency", "emotional_accuracy",
              "cultural_accuracy", "scene_continuity", "story_coherence", "dialogue_duration",
              "lipsync_compatibility", "humour_appropriateness", "emotional_impact", "repetition",
              "translation_completeness")
FLAGS = ("literal_translation", "unnatural_hinglish", "excessive_slang", "inconsistent_accent", "inappropriate_humour",
         "weak_emotion", "incorrect_name", "cultural_inconsistency", "too_long", "missing_translation", "geography",
         "originality", "robotic")
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LATIN = re.compile(r"[A-Za-z]")


def _score(value: Any) -> int | None:
    try:
        return max(0, min(100, int(round(float(value)))))
    except (TypeError, ValueError):
        return None


class ScreenplayQualityEvaluator:
    def __init__(self, *, min_score: int, duration, environment, references, profiles, conversation,
                 old_names: list[str] | None = None) -> None:
        self.min_score = min_score
        self.duration = duration
        self.environment = environment
        self.references = references
        self.profiles = profiles
        self.conversation = conversation
        self.old_names = [n for n in (old_names or []) if n and len(n) >= 3]

    # ---- deterministic ------------------------------------------------------------------

    def line_issues(self, line: dict[str, Any], char_profile: dict[str, Any] | None, plan: dict[str, Any] | None,
                    registry_names: list[str]) -> list[dict[str, str]]:
        text = line.get("target_text") or ""
        if not text.strip():
            return [{"flag": "missing_translation", "detail": "no line"}]
        issues = []
        d = line.get("direction") or {}
        timing = self.duration.check(text, line["target_duration_s"], d.get("speaking_rate", 1.0),
                                     d.get("pause_before", 0.0), d.get("pause_after", 0.0))
        line["timing"] = timing
        if timing["verdict"] == "too_long":
            short_ok = line.get("short_text") and self.duration.check(
                line["short_text"], line["target_duration_s"], d.get("speaking_rate", 1.0))["verdict"] != "too_long"
            if not short_ok:
                issues.append({"flag": "too_long", "detail": f"≈{timing['estimated_s']}s for {timing['usable_s']}s"})
        if len(_DEVANAGARI.findall(text)) < len(_LATIN.findall(text)):
            issues.append({"flag": "unnatural_hinglish", "detail": "mostly roman script; TTS needs Devanagari"})
        for old in self.old_names:
            if old.lower() not in [n.lower() for n in registry_names] and re.search(rf"\b{re.escape(old)}\b", text + " " + (line.get("roman_text") or ""), re.I):
                issues.append({"flag": "incorrect_name", "detail": f"uses old/source name {old}"})
        if char_profile is not None:
            density = self.profiles.slang_density(text)
            if density >= 3 or (density >= 1 and char_profile.get("dialect_intensity", 0.5) < 0.25):
                issues.append({"flag": "excessive_slang", "detail": f"{density} dialect markers for intensity "
                                                                    f"{char_profile.get('dialect_intensity')}"})
        if plan is not None:
            for detail in self.environment.check_line(text + " " + (line.get("roman_text") or ""), plan):
                issues.append({"flag": "geography", "detail": detail})
        for detail in self.references.originality_issues(text, line.get("roman_text") or ""):
            issues.append({"flag": "originality", "detail": detail})
        return issues

    def scene_issues(self, lines: list[dict[str, Any]]) -> list[dict[str, str]]:
        texts = [ln.get("target_text") or "" for ln in lines if ln.get("target_text")]
        counts = Counter(t.strip() for t in texts if len(t.strip()) > 6)
        issues = [{"flag": "repetition", "detail": f"repeated line: {t[:40]}"} for t, c in counts.items() if c > 1]
        issues += [{"flag": "robotic", "detail": d} for d in self.conversation.robotic_issues(texts)]
        return issues

    def deterministic_scores(self, lines: list[dict[str, Any]]) -> dict[str, int]:
        n = max(1, len(lines))
        flags = Counter(i["flag"] for ln in lines for i in ln.get("issues", []))
        fills = [ln["timing"]["fill"] for ln in lines if ln.get("timing", {}).get("fill")]
        lipsync = 100 - int(100 * sum(abs(1 - f) for f in fills) / max(1, len(fills))) if fills else None
        return {
            "translation_completeness": int(100 * (1 - flags["missing_translation"] / n)),
            "dialogue_duration": int(100 * (1 - flags["too_long"] / n)),
            "name_usage": int(100 * (1 - flags["incorrect_name"] / n)),
            "repetition": max(0, 100 - 25 * flags["repetition"]),
            **({"lipsync_compatibility": max(0, lipsync)} if lipsync is not None else {}),
        }

    # ---- model reviews --------------------------------------------------------------------

    def naturalness_system(self, target: str, region: str | None) -> str:
        return (f"You are a native {target} dialogue editor{' from ' + region if region else ''}. You judge whether "
                "film dialogue sounds like real people talking, and fix it when it does not. Be strict about literal "
                "translations, textbook phrasing, forced slang and unnatural Hinglish. Keep the meaning, the speaker's "
                "speaking profile, the names and the length. Answer in JSON only.")

    def naturalness_prompt(self, pid: str, rows: list[str], scene_note: str) -> dict[str, str]:
        return {"id": pid, "system": "", "user": (
            f"{scene_note}\nLines (id | speaker profile | seconds | mood | original | current line):\n" + "\n".join(rows)
            + "\n\nReturn JSON: {\"lines\": [{\"id\": str, \"natural\": 0-100, \"regional\": 0-100 (fits the speaker's "
            'regional profile), "flags": [any of ' + ", ".join(FLAGS) + '], "improved": str|null (better line in '
            'Devanagari, only if it needs fixing), "improved_roman": str|null}]} with every id exactly once.'
        )}

    def screenplay_system(self, target: str) -> str:
        return (f"You are a senior script editor reviewing an Indian {target} adaptation of a foreign drama. You score "
                "the scene honestly and point to the exact lines that need work, with a better version. "
                "Answer in JSON only.")

    def screenplay_prompt(self, scene: dict, rows: list[str], characters: str, technique: str) -> dict[str, str]:
        return {"id": scene["scene_id"], "system": "", "user": (
            f"Scene {scene['scene_id']}: {scene.get('summary', '')} (mood {scene.get('mood')}, intensity "
            f"{scene.get('intensity', 0.5):.1f})\nCharacters:\n{characters}\n{technique}\n"
            "Lines (id | speaker | seconds | original meaning | adapted line | direction):\n" + "\n".join(rows)
            + "\n\nScore 0-100: " + ", ".join(CATEGORIES) + ". Return JSON: {\"scores\": {category: int}, "
            '"issues": [{"id": line id, "flag": one of ' + ", ".join(FLAGS) + ', "detail": str, "rewrite": str '
            '(fixed line in Devanagari, same meaning, fits the seconds), "rewrite_roman": str}], "summary": str}'
        )}

    @staticmethod
    def parse_naturalness(result: Any) -> dict[str, dict]:
        from ..stages.story import answer_items

        out = {}
        for item in answer_items(result):
            out[str(item["id"]).strip()] = {
                "natural": _score(item.get("natural")), "regional": _score(item.get("regional")),
                "flags": [f for f in item.get("flags") or [] if f in FLAGS],
                "improved": (str(item.get("improved")).strip() or None) if item.get("improved") else None,
                "improved_roman": item.get("improved_roman")}
        return out

    @staticmethod
    def parse_screenplay(result: Any) -> dict[str, Any]:
        result = result if isinstance(result, dict) else {}
        scores = {c: _score((result.get("scores") or {}).get(c)) for c in CATEGORIES}
        issues = [i for i in result.get("issues") or [] if isinstance(i, dict) and i.get("id")]
        return {"scores": {k: v for k, v in scores.items() if v is not None}, "issues": issues,
                "summary": str(result.get("summary") or "")[:400]}

    def combine(self, model_scores: dict[str, int], det: dict[str, int]) -> dict[str, Any]:
        scores = {**model_scores, **det}  # measured beats estimated
        overall = int(sum(scores.values()) / len(scores)) if scores else None
        weak = [k for k, v in scores.items() if v < self.min_score]
        return {"scores": scores, "overall": overall, "below_threshold": weak}

    def line_needs_rewrite(self, line: dict[str, Any]) -> bool:
        hard = {"missing_translation", "too_long", "incorrect_name", "geography", "originality", "excessive_slang",
                "literal_translation", "unnatural_hinglish", "inappropriate_humour"}
        if any(i["flag"] in hard for i in line.get("issues", [])):
            return True
        review = line.get("review") or {}
        if review.get("applied"):
            return False  # the reviewer's own improved line was accepted
        natural = review.get("natural")
        return natural is not None and natural < self.min_score
