"""EmotionAndDeliveryPlanner: an audio direction per line, and its translation into TTS controls.

The direction is written by the screenplay writer with the scene in view. TTS engines expose only
a few controls, so the mapping is explicit (and reported):
  - Chatterbox: exaggeration (expressiveness) and cfg_weight (lower = faster, more dramatic pacing)
  - every engine: pause before/after (timeline offset), volume (gain), speaking rate (tempo target)
  - voice breaks, crying and whispering cannot be synthesised by these engines; they are carried in the
    text (punctuation, broken words) and listed as "unsupported" in the audio report.
"""

from __future__ import annotations

from typing import Any

EMOTIONS = ("neutral", "happy", "sad", "crying", "angry", "frustrated", "sarcastic", "fearful", "nervous", "excited",
            "romantic", "whispering", "shouting", "shocked", "confident", "humorous", "suspicious", "restrained")
DELIVERY_STYLES = ("natural", "controlled_aggressive", "explosive", "soft", "whisper", "playful", "deadpan",
                   "trembling", "hesitant", "urgent", "warm", "cold", "teasing", "pleading")
VOLUMES = {"whisper": 0.5, "soft": 0.75, "normal": 1.0, "strong": 1.12, "shouting": 1.2}
# base expressiveness per emotion family (Chatterbox exaggeration at intensity 0.5)
BASE_EXAGGERATION = {"neutral": 0.45, "happy": 0.6, "excited": 0.75, "sad": 0.55, "crying": 0.7, "angry": 0.8,
                     "frustrated": 0.7, "sarcastic": 0.6, "fearful": 0.65, "nervous": 0.6, "romantic": 0.55,
                     "whispering": 0.4, "shouting": 0.95, "shocked": 0.8, "confident": 0.55, "humorous": 0.65,
                     "suspicious": 0.55, "restrained": 0.4}
UNSUPPORTED = ("voice_break", "crying", "whisper", "overlap")


def _num(value: Any, default: float, lo: float, hi: float) -> float:
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return default


class EmotionAndDeliveryPlanner:
    def __init__(self, audio: dict[str, Any], default_accent: str = "neutral_hindi", style: str = "natural_hindi") -> None:
        self.audio = audio
        self.default_accent = default_accent
        self.style = style

    def schema(self) -> str:
        return ('"direction": {"emotion": one of ' + "|".join(EMOTIONS) + ', "emotion_detail": str (e.g. "angry but '
                'controlled", "sad but hiding it"), "intensity": float 0-1, "delivery": one of ' + "|".join(DELIVERY_STYLES) +
                ', "speaking_rate": float 0.8-1.25, "pitch_variation": "low"|"medium"|"high", "volume": '
                '"whisper"|"soft"|"normal"|"strong"|"shouting", "pause_before": seconds 0-1, "pause_after": seconds 0-1, '
                '"voice_break": bool, "overlap_previous": bool (interrupts the previous speaker)}')

    def normalize(self, raw: Any, *, character_id: str | None, accent: str | None, fallback_emotion: str = "neutral"
                  ) -> dict[str, Any]:
        raw = raw if isinstance(raw, dict) else {}
        emotion = str(raw.get("emotion") or fallback_emotion or "neutral").lower().strip()
        if emotion not in EMOTIONS:
            emotion = next((e for e in EMOTIONS if e in emotion or emotion in e), "neutral")
        delivery = str(raw.get("delivery") or "natural").lower().strip().replace(" ", "_")
        volume = str(raw.get("volume") or "normal").lower().strip()
        max_pause = float(self.audio.get("max_pause_seconds", 0.8)) if self.audio.get("conversational_pauses", True) else 0.0
        return {
            "character_id": character_id, "emotion": emotion, "emotion_detail": str(raw.get("emotion_detail") or "")[:80],
            "intensity": round(_num(raw.get("intensity"), 0.5, 0, 1), 2),
            "delivery": delivery if delivery in DELIVERY_STYLES else "natural",
            "speaking_rate": round(_num(raw.get("speaking_rate"), 1.0, 0.8, 1.25), 2),
            "pitch_variation": raw.get("pitch_variation") if raw.get("pitch_variation") in ("low", "medium", "high") else "medium",
            "volume": volume if volume in VOLUMES else "normal",
            "pause_before": round(_num(raw.get("pause_before"), 0.0, 0, max_pause), 2),
            "pause_after": round(_num(raw.get("pause_after"), 0.0, 0, max_pause), 2),
            "voice_break": bool(raw.get("voice_break")), "overlap_previous": bool(raw.get("overlap_previous")),
            "regional_accent": accent or self.default_accent, "dialogue_style": self.style,
        }

    def tts_controls(self, direction: dict[str, Any], *, base_cfg: float, emotion_aware: bool = True) -> dict[str, Any]:
        """Map a direction onto engine controls. Returns exaggeration, cfg_weight, gain, tempo_target, unsupported."""
        if not emotion_aware:
            return {"exaggeration": 0.5, "cfg_weight": base_cfg, "gain": 1.0, "tempo_target": 1.0, "unsupported": []}
        emotion, intensity = direction.get("emotion", "neutral"), float(direction.get("intensity", 0.5))
        exaggeration = BASE_EXAGGERATION.get(emotion, 0.5) + 0.5 * (intensity - 0.5)
        if direction.get("delivery") in ("controlled_aggressive", "cold", "deadpan") or emotion == "restrained":
            exaggeration -= 0.1  # controlled: intensity in the words and pacing, not in volume swings
        if direction.get("delivery") in ("explosive", "urgent"):
            exaggeration += 0.1
        rate = float(direction.get("speaking_rate", 1.0))
        cfg = base_cfg - 0.35 * (rate - 1.0) - (0.1 if exaggeration > 0.75 else 0.0)  # Chatterbox: low cfg = faster
        unsupported = [u for u in UNSUPPORTED if (
            (u == "voice_break" and direction.get("voice_break")) or (u == "crying" and emotion == "crying") or
            (u == "whisper" and (emotion == "whispering" or direction.get("volume") == "whisper")) or
            (u == "overlap" and direction.get("overlap_previous")))]
        return {"exaggeration": round(max(0.25, min(1.2, exaggeration)), 3), "cfg_weight": round(max(0.2, min(0.8, cfg)), 3),
                "gain": VOLUMES.get(direction.get("volume", "normal"), 1.0), "tempo_target": rate, "unsupported": unsupported}

    def apply_timing(self, line: dict[str, Any], direction: dict[str, Any]) -> dict[str, Any]:
        """Shift a placed line by its pause_before and keep pause_after free (within its slot)."""
        before = min(float(direction.get("pause_before", 0.0)), max(0.0, line["available_s"] - 0.6) * 0.3)
        after = min(float(direction.get("pause_after", 0.0)), max(0.0, line["available_s"] - before - 0.6) * 0.3)
        if direction.get("overlap_previous"):
            before = 0.0
        return {**line, "out_start": round(line["out_start"] + before, 3),
                "available_s": round(max(0.3, line["available_s"] - before - after), 3),
                "pause_before_applied": round(before, 3), "pause_after_applied": round(after, 3)}
