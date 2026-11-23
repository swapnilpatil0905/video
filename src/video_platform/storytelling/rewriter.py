"""IndianDialogueRewriter + NaturalConversationGenerator.

The writer receives the analysed scene (mood, intentions, relationships, location plan), the
character registry with speaking profiles, the allowed enhancements and a syllable budget per line.
It writes the scene as an Indian screenwriter would — never word by word — plus an audio
direction per line.
"""

from __future__ import annotations

import re
from typing import Any

from .gender import RULE as GENDER_RULE

STYLE_RULES = {
    "natural_hindi": "Everyday spoken Hindi as people really talk; English words only where Indians naturally use them.",
    "natural_hinglish": "Natural urban Hinglish: Hindi sentences with English words/phrases where a real person would "
                        "switch (work, tech, emotions people say in English). Never translate common English words "
                        "into pure Hindi; never write full English sentences unless the character would.",
    "formal_hindi": "Clear, polite Hindi without slang.",
    "dakhni": "Dakhni-heavy Hyderabadi speech for characters with high dialect intensity.",
}


class NaturalConversationGenerator:
    """Rules for real conversation (fed to the writer) and checks for robotic output."""

    RULES = (
        "Real people do not speak in perfect sentences. Use, where the moment calls for it: short replies "
        "('हाँ।', 'तो?'), incomplete sentences ending in '...', interruptions (end a line with '—' and let the next "
        "speaker cut in), natural fillers (अरे, हाँ तो, मतलब, यार, अच्छा) sparingly, a repeated word for emphasis, "
        "hesitation, and silence (an empty-sounding line like '...' only if the character truly says nothing). "
        "Vary sentence length between characters. Keep every piece of story information and every cause-and-effect."
    )
    FILLERS = ("अरे", "यार", "मतलब", "अच्छा", "हाँ तो", "वो", "देखो", "सुनो")

    def robotic_issues(self, lines: list[str]) -> list[str]:
        """Scene-level smells: every line long and complete, or fillers in every line."""
        issues = []
        if len(lines) >= 5:
            long = [t for t in lines if len(t.split()) >= 9 and t.rstrip().endswith(("।", ".", "?"))]
            if len(long) == len(lines):
                issues.append("every line is a long complete sentence (robotic rhythm)")
            filled = [t for t in lines if any(t.strip().startswith(f) for f in self.FILLERS)]
            if len(filled) >= 0.7 * len(lines):
                issues.append("fillers at the start of most lines (over-used)")
        return issues


class IndianDialogueRewriter:
    def __init__(self, *, target: str, region: str | None, style: str, register: str, conversation: NaturalConversationGenerator,
                 direction_schema: str, originality_rule: str) -> None:
        self.target = target
        self.region = region
        self.style = style
        self.register = register
        self.conversation = conversation
        self.direction_schema = direction_schema
        self.originality_rule = originality_rule

    def system(self) -> str:
        return (
            f"You are an Indian screenwriter. You rewrite a scene from a foreign drama as if you had originally written "
            f"it for {self.region or 'Indian'} audiences in {self.target}. You never translate word by word: you keep the "
            "meaning, intention, emotion and story information, and write what THIS character would naturally say to "
            f"THIS listener in THIS situation. Register: {self.register}. {STYLE_RULES.get(self.style, '')} "
            f"{self.conversation.RULES} Use each character's speaking profile: a high dialect intensity means "
            "regional expressions come naturally to them; a low one means almost none. Never put slang in every line. "
            f"Use the given Indian names and forms of address exactly. {GENDER_RULE} {self.originality_rule} "
            f"Write {self.target} in Devanagari (English words in Devanagari too, as Indians spell them) so a "
            "text-to-speech voice reads it correctly. Answer in JSON only."
        )

    def prompt(self, pid: str, *, scene: dict, enhancement: str, characters: str, expressions: str,
               places: list[str], fictional: list[str], lines: list[str], previous_lines: list[str],
               feedback: str = "") -> dict[str, str]:
        place_text = (f"Real places this scene may mention (only if natural): {', '.join(places)}. "
                      if places else "Do not name real places in this scene. ")
        if fictional:
            place_text += f"Invented places (fictional, may be named): {', '.join(fictional)}. "
        return {"id": pid, "system": self.system(), "user": (
            f"Scene {scene['scene_id']} — {scene.get('summary', '')}\nMood: {scene.get('mood')} (intensity "
            f"{scene.get('intensity', 0.5):.1f}); setting: {(scene.get('location_plan') or {}).get('setting', '')}\n"
            f"Relationships: {scene.get('relationships')}\n{enhancement}\n{place_text}No travel times, no invented "
            f"real-world roads/stations/businesses.\nCharacters in this scene:\n{characters}\n"
            f"Regional expressions you MAY use where they fit the character and moment:\n{expressions}\n"
            + (f"Lines just before (already written):\n" + "\n".join(previous_lines[-4:]) + "\n" if previous_lines else "")
            + "\nLines to write (id | speaker (gender) → listener (gender) | seconds available | syllables target/max | intention | "
            "emotion | original text):\n" + "\n".join(lines)
            + (f"\n\nReviewer feedback to fix (rewrite these lines accordingly):\n{feedback}" if feedback else "")
            + "\n\nEach line must fit its syllable budget (the max is a hard limit). Return JSON: {\"lines\": [{"
            '"id": str, "text": str (the line, Devanagari), "roman": str (same line in simple roman letters), '
            '"short": str (shorter version, same meaning), "meaning_en": str (what it conveys, in English), '
            f"{self.direction_schema}" + "}]} with every id exactly once."
        )}

    @staticmethod
    def clean(text: Any) -> str:
        text = re.sub(r"\s+", " ", str(text or "")).strip()
        return text.strip('"“”')
