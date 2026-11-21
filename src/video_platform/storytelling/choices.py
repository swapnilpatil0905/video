"""Language choice as input: region, language style and slang level → saved project settings.

    bash start.sh run --region mumbai --language hinglish --slang low
    bash start.sh configure --region hyderabad --slang high        # save only

The choice is written to projects/<id>/local.yaml (gitignored) and merged over config.yaml on every
command, so resume / approve / later runs keep it without repeating the flags.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..core.errors import ConfigError
from .profiles import PROFILE_DIR

LANGUAGE_STYLES = {
    "hindi": "natural_hindi", "natural_hindi": "natural_hindi",
    "hinglish": "natural_hinglish", "natural_hinglish": "natural_hinglish",
    "formal": "formal_hindi", "formal_hindi": "formal_hindi", "shuddh": "formal_hindi",
    "dakhni": "dakhni", "hyderabadi": "dakhni",
}
SLANG_LEVELS = {"none": 0.0, "off": 0.0, "low": 0.25, "light": 0.25, "medium": 0.5, "high": 0.8, "max": 1.0}
REGION_ALIASES = {"hyd": "hyderabad", "bombay": "mumbai", "bangalore": "bengaluru", "calcutta": "kolkata",
                  "madras": "chennai", "neutral": "neutral_hindi", "none": "neutral_hindi", "standard": "neutral_hindi",
                  "punjab": "punjabi_hindi", "punjabi": "punjabi_hindi", "telugu": "telugu_hindi"}


def available_regions(directory: Path | None = None) -> list[str]:
    return sorted(p.stem for p in (directory or PROFILE_DIR).glob("*.yaml"))


def parse_slang(value: str | float) -> float:
    if isinstance(value, (int, float)):
        level = float(value)
    else:
        text = str(value).strip().lower()
        if text in SLANG_LEVELS:
            level = SLANG_LEVELS[text]
        else:
            try:
                level = float(text)
            except ValueError:
                raise ConfigError(f"--slang must be none|low|medium|high|max or 0–1, not '{value}'") from None
    if not 0 <= level <= 1:
        raise ConfigError(f"--slang must be between 0 and 1, not {level}")
    return level


def localization_overrides(region: str | None = None, language: str | None = None, slang: str | float | None = None,
                           directory: Path | None = None) -> dict[str, Any]:
    """The `localization` settings implied by the three inputs (only the ones given)."""
    out: dict[str, Any] = {}
    if region:
        key = REGION_ALIASES.get(region.strip().lower(), region.strip().lower())
        regions = available_regions(directory)
        if key not in regions:
            raise ConfigError(f"unknown region '{region}'. Available: {', '.join(regions)}")
        profile = yaml.safe_load(((directory or PROFILE_DIR) / f"{key}.yaml").read_text(encoding="utf-8")) or {}
        dialects = list((profile.get("dialects") or {}).keys())
        regional = [d for d in dialects if d != "neutral_hindi"]
        out["regional_profile"] = key
        out["dialect"] = {"name": (regional or dialects or ["neutral_hindi"])[0]}
        out["character_names"] = {"region": profile.get("city") or None}
    if language:
        style = LANGUAGE_STYLES.get(language.strip().lower())
        if style is None:
            raise ConfigError(f"--language must be one of {sorted(set(LANGUAGE_STYLES))}, not '{language}'")
        out["writing_style"] = style
    if slang is not None:
        level = parse_slang(slang)
        out.setdefault("dialect", {})["intensity"] = level
        out["dialect"]["forced_slang"] = False
    return out


def deep_update(base: dict, extra: dict) -> dict:
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def save_local(project_dir: Path, overrides: dict[str, Any]) -> Path:
    """Merge into projects/<id>/local.yaml (keeps any other local settings)."""
    path = project_dir / "local.yaml"
    current = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}) if path.exists() else {}
    deep_update(current.setdefault("localization", {}), overrides)
    header = ("# Local choices for this project (gitignored). Written by: bash start.sh configure / run --region …\n"
              "# Merged over config.yaml on every command. Delete this file to go back to config.yaml.\n")
    path.write_text(header + yaml.safe_dump(current, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def describe(loc: Any) -> str:
    """One line: region · style · dialect · slang."""
    return (f"region={loc.regional_profile} · language=Hindi ({loc.writing_style}) · dialect={loc.dialect.name} · "
            f"slang={loc.dialect.intensity:.2f} · names from={loc.character_names.region or loc.regional_profile}")
