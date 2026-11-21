"""ScreenplayEnhancementEngine: what a scene may strengthen — decided from the scene, never at random."""

from __future__ import annotations

from typing import Any

GUIDANCE = {
    "comedy": "Situational or character humour, light sarcasm, timing, a humorous reaction or a regional "
              "expression — only where it grows out of the situation. Never a forced joke.",
    "romance": "Natural, slightly hesitant conversation, subtle teasing, unfinished sentences, a pause before "
               "the important word, chemistry through what is NOT said.",
    "emotion": "Hesitation, short broken sentences, a pause or silence where a person would struggle, regret "
               "shown in behaviour rather than announced.",
    "tragedy": "Restraint: fewer words, voice breaking on key words, silence; never melodramatic speeches.",
    "suspense": "Short lines, lowered voices, unanswered questions, a beat of silence before a reveal, urgency.",
    "confrontation": "Escalation, interruptions (a line can stop mid-word with '—'), strong short sentences, "
                     "strategic silence, threats without shouting where the character is controlled.",
}
# scene moods in which each enhancement may never be added, whatever the model suggested
FORBIDDEN_IN = {"comedy": {"tragedy", "emotional", "suspense", "thriller", "confrontation"}}
MOOD_TO_KEYS = {"comedy": ["comedy"], "romance": ["romance"], "tragedy": ["tragedy", "emotion"],
                "emotional": ["emotion"], "suspense": ["suspense"], "thriller": ["suspense"],
                "family_drama": ["emotion"], "action": ["suspense"], "confrontation": ["confrontation"], "neutral": []}
CONFIG_KEY = {"comedy": "enhance_comedy", "romance": "enhance_romance", "emotion": "enhance_emotions",
              "tragedy": "enhance_tragedy", "suspense": "enhance_suspense", "confrontation": "enhance_emotions"}


class ScreenplayEnhancementEngine:
    def __init__(self, storytelling: dict[str, Any]) -> None:
        self.cfg = storytelling

    def plan(self, scene: dict[str, Any]) -> dict[str, Any]:
        """Enhancements allowed for this scene (config ∧ scene mood ∧ analyzer opportunity) with guidance."""
        if self.cfg.get("mode") == "faithful_localization":
            return {"allowed": [], "instructions": "Faithful localization: keep the scene's content; only make it natural."}
        mood = scene.get("mood", "neutral")
        suggested = {k for k, v in (scene.get("enhancement") or {}).items() if v is True}
        candidates = set(MOOD_TO_KEYS.get(mood, [])) | suggested
        allowed = []
        for key in sorted(candidates):
            if key not in GUIDANCE or not self.cfg.get(CONFIG_KEY.get(key, ""), False):
                continue
            if mood in FORBIDDEN_IN.get(key, set()):
                continue
            if key == "comedy" and self.cfg.get("avoid_forced_humour", True) and mood != "comedy" \
                    and (scene.get("intensity", 0.5) > 0.6 or "comedy" not in suggested):
                continue  # humour only where the analyzer saw an opening in a light scene
            allowed.append(key)
        lines = [f"- {k}: {GUIDANCE[k]}" for k in allowed]
        notes = (scene.get("enhancement") or {}).get("notes")
        rules = []
        if self.cfg.get("preserve_story_logic", True):
            rules.append("Keep every plot fact, decision and cause-and-effect of the scene.")
        if self.cfg.get("preserve_character_motivations", True):
            rules.append("Keep each character's motivation and attitude.")
        return {"allowed": allowed, "instructions": "\n".join(
            (["Strengthen (only as described):"] + lines if lines else ["No enhancement: write the scene naturally."])
            + ([f"Analyst notes: {str(notes)[:300]}"] if notes and lines else []) + rules)}
