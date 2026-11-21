"""RegionalLanguageProfileManager: regional profiles (data files) and per-character speaking profiles."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

PROFILE_DIR = Path(__file__).resolve().parents[3] / "configs" / "regional_profiles"

ACCENTS = ("hyderabadi", "dakhni", "neutral_hindi", "mumbai", "delhi", "lucknowi", "punjabi", "telugu", "other")
MIXING_LEVELS = {"low": 0.15, "medium": 0.4, "high": 0.7}


def _level(value: Any, default: float) -> float:
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    if isinstance(value, str):
        return MIXING_LEVELS.get(value.strip().lower(), default)
    return default


def _norm(text: str) -> str:
    return re.sub(r"[^\wऀ-ॿ]+", " ", text.lower()).strip()


class RegionalLanguageProfileManager:
    def __init__(self, profile: dict[str, Any], dialect: dict[str, Any], writing_style: str) -> None:
        self.profile = profile
        self.dialect = dialect
        self.writing_style = writing_style

    @classmethod
    def load(cls, name: str, dialect: dict[str, Any], writing_style: str, directory: Path | None = None
             ) -> "RegionalLanguageProfileManager":
        path = (directory or PROFILE_DIR) / f"{name}.yaml"
        if not path.exists():
            available = sorted(p.stem for p in (directory or PROFILE_DIR).glob("*.yaml"))
            raise FileNotFoundError(f"regional profile '{name}' not found; available: {available}")
        return cls(yaml.safe_load(path.read_text(encoding="utf-8")) or {}, dialect, writing_style)

    # ---- profile data -------------------------------------------------------------

    @property
    def id(self) -> str:
        return self.profile.get("id", "custom")

    @property
    def city(self) -> str | None:
        return self.profile.get("city")

    def dialect_info(self, name: str | None = None) -> dict[str, Any]:
        dialects = self.profile.get("dialects") or {}
        return dialects.get(name or self.dialect.get("name", ""), next(iter(dialects.values()), {}))

    def expressions(self, max_intensity: float = 1.0) -> list[dict]:
        return [e for e in self.profile.get("expressions") or [] if float(e.get("min_intensity", 0)) <= max_intensity]

    def expression_markers(self) -> list[str]:
        """Normalised dialect markers (both scripts) used to measure slang density."""
        out = []
        for e in self.profile.get("expressions") or []:
            if float(e.get("min_intensity", 0)) <= 0.2:
                continue  # pan-Indian words (pakka, ek dum) are not dialect markers
            for key in ("roman", "devanagari"):
                if e.get(key):
                    out.append(_norm(e[key]))
        return [m for m in out if m]

    def naming(self) -> dict[str, Any]:
        return self.profile.get("naming") or {}

    def culture(self) -> dict[str, Any]:
        return self.profile.get("culture") or {}

    # ---- character speaking profiles -------------------------------------------------

    def default_character_profile(self) -> dict[str, Any]:
        return {
            "language": "Hindi", "region": self.city or "India", "accent": "neutral_hindi",
            "dialect": self.dialect.get("name"), "dialect_intensity": float(self.dialect.get("intensity", 0.3)),
            "english_mixing": float(self.dialect.get("english_mixing", 0.3)),
            "urdu_influence": float(self.dialect.get("urdu_influence", 0.3)), "speaking_style": "natural, casual",
            "speech_habits": [],
        }

    def normalize_character_profile(self, raw: dict[str, Any] | None) -> dict[str, Any]:
        base = self.default_character_profile()
        raw = raw if isinstance(raw, dict) else {}
        accent = str(raw.get("accent") or base["accent"]).strip().lower().replace(" ", "_")
        if accent not in ACCENTS:
            accent = next((a for a in ACCENTS if a in accent), "other")
        habits = raw.get("speech_habits") or []
        return {
            "language": raw.get("language") or base["language"], "region": raw.get("region") or base["region"],
            "accent": accent, "dialect": raw.get("dialect") or base["dialect"],
            "dialect_intensity": _level(raw.get("dialect_intensity"), base["dialect_intensity"]),
            "english_mixing": _level(raw.get("english_mixing"), base["english_mixing"]),
            "urdu_influence": _level(raw.get("urdu_influence"), base["urdu_influence"]),
            "speaking_style": str(raw.get("speaking_style") or base["speaking_style"])[:120],
            "speech_habits": [str(h)[:80] for h in habits][:5] if isinstance(habits, list) else [],
        }

    def voice_accent(self, character_profile: dict[str, Any]) -> str:
        """Accent phrase for a voice-design description (e.g. Indic Parler-TTS)."""
        accent = character_profile.get("accent", "neutral_hindi")
        dialects = self.profile.get("dialects") or {}
        if accent in ("hyderabadi", "dakhni"):
            key = "dakhni_urdu" if accent == "dakhni" or character_profile.get("dialect_intensity", 0) >= 0.7 \
                else "hyderabadi_hindi"
            info = dialects.get(key) or self.dialect_info()
        else:
            info = dialects.get(accent) or dialects.get("neutral_hindi") or {}
        return info.get("accent_for_voice") or "speaks natural Indian Hindi"

    def describe_for_prompt(self, character_profile: dict[str, Any]) -> str:
        p = character_profile
        return (f"accent {p['accent']}, dialect intensity {p['dialect_intensity']:.2f}, English mixing "
                f"{p['english_mixing']:.2f}, Urdu influence {p['urdu_influence']:.2f}, style: {p['speaking_style']}"
                + (f", habits: {'; '.join(p['speech_habits'])}" if p.get("speech_habits") else ""))

    def expressions_for_prompt(self, intensity: float, limit: int = 14) -> str:
        rows = [f"{e['roman']} ({e.get('devanagari', '')}) = {e['meaning']} [{e.get('register', 'any')}; {e.get('use', '')}]"
                for e in self.expressions(intensity)[:limit]]
        return "\n".join(rows) or "(none)"

    def slang_density(self, text: str) -> int:
        """How many dialect markers a line contains."""
        norm = f" {_norm(text)} "
        return sum(1 for m in self.expression_markers() if f" {m} " in norm)
