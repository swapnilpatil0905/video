"""IndianNameGenerator: a consistent, culturally coherent character identity registry.

One model call sees ALL characters (so names are unique and families share coherent surnames),
then deterministic checks reject duplicates / confusingly similar names and missing family links.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from typing import Any

FAMILY_WORDS = ("father", "mother", "son", "daughter", "brother", "sister", "sibling", "parent", "grand",
                "uncle", "aunt", "cousin", "papa", "mama", "dad", "mom", "family")
MARRIAGE_WORDS = ("wife", "husband", "married", "spouse", "fiance", "fiancée", "fiancee", "in-law")


def _ascii(name: str) -> str:
    return unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode().lower().strip()


def name_distance(a: str, b: str) -> float:
    """0 = identical, 1 = completely different (on given names, case/diacritics-insensitive)."""
    a, b = _ascii(a).split(" ")[0], _ascii(b).split(" ")[0]
    if not a or not b:
        return 1.0
    return 1.0 - difflib.SequenceMatcher(None, a, b).ratio()


class IndianNameGenerator:
    def __init__(self, naming: dict[str, Any], region: str | None, preserve_relationships: bool = True,
                 min_distance: float = 0.34) -> None:
        self.naming = naming
        self.region = region
        self.preserve_relationships = preserve_relationships
        self.min_distance = min_distance

    def system(self) -> str:
        return (
            "You are a casting writer adapting a drama for Indian audiences. You give every character an Indian name "
            "that fits their gender, age, family, social background, profession and personality in THIS story. "
            "Family members share a coherent family name unless the story says otherwise. Do not choose a religion or "
            "community for flavour and never stereotype; only reflect a community if the story deliberately implies it. "
            "Names must be distinct and not confusable with each other. You also define how each character speaks. "
            "Answer in JSON only."
        )

    def prompt(self, characters: list[dict], synopsis: str, relationships: str, profile_text: str,
               style: str, avoid: list[str] | None = None) -> dict[str, str]:
        n = self.naming
        examples = (f"Example male names (NOT a fixed list): {', '.join(n.get('examples_male', [])[:8])}\n"
                    f"Example female names (NOT a fixed list): {', '.join(n.get('examples_female', [])[:8])}\n")
        rows = "\n".join(
            f"- {c['character_id']}: gender={c.get('gender')}, age≈{c.get('age')}, importance={c.get('tier')}, "
            f"role={c.get('role')}, description={str(c.get('description') or '')[:200]}" for c in characters)
        return {"id": "identity", "system": self.system(), "user": (
            f"Region: {self.region or 'India'}. Naming conventions: {n.get('conventions', '')}\n{examples}"
            f"Forms of address in this region: {n.get('honorifics', {})}\n"
            f"Dialogue style for the remake: {style}. Regional speaking profile: {profile_text}\n"
            f"Story synopsis: {synopsis[:1500]}\nRelationships: {relationships[:1500]}\nCharacters:\n{rows}\n"
            + (f"Do NOT use these names (taken or too similar): {', '.join(avoid)}\n" if avoid else "")
            + "\nGive the characters DISTINCT speaking profiles where their backgrounds differ (at least two different "
            "accents/intensities if there are several characters). Return JSON: {\"characters\": [{"
            '"character_id": str, "given_name": str, "family_name": str|null, "display_name": str (how others mostly '
            'call them), "devanagari": str (full name in Devanagari, for speech), "pronunciation": str (simple roman '
            'syllables, e.g. "AH-run REH-dee"), "nickname": str|null, "family_group": str|null (shared id for one '
            'family), "profession": str|null, "background": str (one line), "personality": str, '
            '"forms_of_address": [{"by": character id or "others", "form": str (e.g. "Arjun bhai", "Ammi")}], '
            '"language_profile": {"language": "Hindi", "region": str, "accent": "hyderabadi"|"dakhni"|"neutral_hindi"|'
            '"mumbai"|"delhi"|"lucknowi"|"punjabi"|"telugu"|"other", "dialect_intensity": float 0-1, '
            '"english_mixing": float 0-1, "urdu_influence": float 0-1, "speaking_style": str, '
            '"speech_habits": [str]}}]}'
        )}

    def validate(self, registry: list[dict], relationships: list[dict]) -> list[str]:
        """Problems: duplicate/similar names, missing names, family members without a shared family name."""
        issues = []
        for i, a in enumerate(registry):
            if not a.get("given_name"):
                issues.append(f"{a['character_id']}: no name")
            for b in registry[i + 1:]:
                if a.get("given_name") and b.get("given_name") and \
                        name_distance(a["given_name"], b["given_name"]) < self.min_distance:
                    issues.append(f"names too similar: {a['given_name']} ({a['character_id']}) / "
                                  f"{b['given_name']} ({b['character_id']})")
        if self.preserve_relationships:
            by_id = {r["character_id"]: r for r in registry}
            for rel in relationships:
                text = str(rel.get("relation", "")).lower()
                a, b = by_id.get(rel.get("a")), by_id.get(rel.get("b"))
                if not a or not b or not any(w in text for w in FAMILY_WORDS) or any(w in text for w in MARRIAGE_WORDS):
                    continue
                fa, fb = _ascii(a.get("family_name") or ""), _ascii(b.get("family_name") or "")
                if fa and fb and fa != fb:
                    issues.append(f"family names differ for related characters {a['character_id']} ({a.get('family_name')}) "
                                  f"and {b['character_id']} ({b.get('family_name')}): {text}")
        return issues

    @staticmethod
    def normalize(raw: Any, character_id: str, fallback_name: str) -> dict[str, Any]:
        raw = raw if isinstance(raw, dict) else {}
        given = re.sub(r"\s+", " ", str(raw.get("given_name") or "").strip())
        family = (str(raw.get("family_name")).strip() or None) if raw.get("family_name") else None
        full = " ".join(x for x in (given, family) if x) or fallback_name
        return {
            "character_id": character_id, "given_name": given or fallback_name.split(" ")[0], "family_name": family,
            "full_name": full, "display_name": str(raw.get("display_name") or given or full),
            "devanagari": str(raw.get("devanagari") or ""), "pronunciation": str(raw.get("pronunciation") or ""),
            "nickname": raw.get("nickname"), "family_group": raw.get("family_group"),
            "profession": raw.get("profession"), "background": str(raw.get("background") or "")[:200],
            "personality": str(raw.get("personality") or "")[:200],
            "forms_of_address": [f for f in raw.get("forms_of_address") or [] if isinstance(f, dict)][:8],
            "language_profile_raw": raw.get("language_profile"),
            "status": "ok" if given else "fallback",
        }
