"""DialogueDurationOptimizer: every line must fit its time window at a natural speaking rate."""

from __future__ import annotations

import re

_DEV_CONSONANT = re.compile("[\u0915-\u0939\u0958-\u095F]")
_DEV_VIRAMA = "\u094D"
_DEV_INDEP_VOWEL = re.compile("[\u0905-\u0914]")
_LATIN_VOWEL_GROUPS = re.compile(r"[aeiouy]+", re.I)
_PAUSE = re.compile(r"(\.\.\.|…|—|--)")


def syllables(text: str) -> int:
    """Approximate spoken syllables (Devanagari: consonants not followed by virama + independent vowels)."""
    text = text or ""
    count = 0
    for i, ch in enumerate(text):
        if _DEV_CONSONANT.match(ch):
            if not (i + 1 < len(text) and text[i + 1] == _DEV_VIRAMA):
                count += 1
        elif _DEV_INDEP_VOWEL.match(ch):
            count += 1
    latin = re.sub("[\u0900-\u097F]", " ", text)
    count += sum(max(1, len(_LATIN_VOWEL_GROUPS.findall(word))) for word in re.findall(r"[A-Za-z]+", latin))
    count += sum(ch.isdigit() for ch in text)  # digits are spoken as words
    return count


class DialogueDurationOptimizer:
    def __init__(self, syllables_per_second: float = 5.0, max_speedup: float = 1.12, min_fill: float = 0.35,
                 pause_seconds: float = 0.25) -> None:
        self.rate = syllables_per_second
        self.max_speedup = max_speedup
        self.min_fill = min_fill
        self.pause_seconds = pause_seconds

    def estimate_seconds(self, text: str, speaking_rate: float = 1.0) -> float:
        pauses = len(_PAUSE.findall(text or "")) * self.pause_seconds
        return syllables(text) / (self.rate * max(0.6, min(1.6, speaking_rate or 1.0))) + pauses

    def budget(self, available_s: float) -> dict[str, int]:
        """Syllable target / maximum for a slot, used in the writer prompt."""
        return {"target": max(2, int(available_s * self.rate * 0.9)), "max": max(3, int(available_s * self.rate * self.max_speedup))}

    def check(self, text: str, available_s: float, speaking_rate: float = 1.0, pause_before: float = 0.0,
              pause_after: float = 0.0) -> dict[str, object]:
        usable = max(0.3, available_s - pause_before - pause_after)
        est = self.estimate_seconds(text, speaking_rate)
        verdict = "ok"
        if est > usable * self.max_speedup:
            verdict = "too_long"
        elif est < usable * self.min_fill and available_s > 1.5:
            verdict = "too_short"
        return {"estimated_s": round(est, 2), "usable_s": round(usable, 2), "verdict": verdict,
                "fill": round(est / usable, 2) if usable else None}

    def choose(self, text: str, short: str, available_s: float, speaking_rate: float = 1.0) -> tuple[str, str]:
        """Pick the full or short variant ('text' | 'short') that fits; full wins when both fit."""
        if self.check(text, available_s, speaking_rate)["verdict"] != "too_long" or not short:
            return text, "text"
        return short, "short"
