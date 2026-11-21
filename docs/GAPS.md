# Gap review (v0.5)

| # | Gap | Impact on the trailer | Status |
|---|---|---|---|
| 1 | The legacy analysis looked at **one frame per shot**, so main characters could be missed. Their shots then kept the original actor. | High | **Fixed in v0.5.** Faces are now detected on 2 frames/s at 720p (SCRFD). Each trailer clip's own frame is also matched, and Light-ASD speaking evidence counts as "visible". |
| 2 | Active-speaker detection was a motion heuristic | High (wrong voice or face) | **Fixed in v0.5.** Trained Light-ASD, with the heuristic as fallback. |
| 3 | Hindi naturalness depended on one LLM | High | **Improved in v0.5.** A Sarvam-M polish pass over Qwen's draft (draft kept on failure). |
| 4 | Hindi TTS engine choice was fixed | Medium–High | **Improved in v0.5.** Automatic Chatterbox vs IndicF5 bake-off (back-transcription CER). |
| 5 | Chinese voices leaking into the music/effects bed | Medium | **Improved in v0.5.** BS-RoFormer separation (Demucs fallback). |
| 6 | Identity drift across a character's expression sheet | Medium | **Improved in v0.5.** Qwen-Image-Edit-2509, plus the existing identity checks and exclusion of drifting variants. |
| 7 | Wan2.2-Animate replaces **one person per shot**; other people in two-shots keep the source footage | Medium | **Open.** Needs per-person masks and multiple passes (future). |
| 8 | Non-speech vocal sounds (laughs, cries, screams) are removed together with the Chinese dialogue | Low–Medium | **Open.** They could be restored from the vocals stem where no line is spoken (future). |
| 9 | Overlapping speech: one speaker per segment | Low | **Open** (rare in this drama) |
| 10 | Emotion control: IndicF5 has none (Chatterbox has exaggeration) | Low–Medium | Mitigated: the bake-off prefers Chatterbox on a tie |
| 11 | The story model loads separately for narrative, design, localization and polish (~2–3 min each) | Time only | Accepted (≈10 min total) |
| 13 | Chinese ASR accuracy (Whisper on Mandarin) | Medium | **Improved in v0.5.** FunASR Paraformer-zh, with faster-whisper as fallback. |
| 14 | Diarization error rate | Medium | **Improved in v0.5.** DiariZen, with pyannote as fallback. |
| 12 | Hindi lip-sync quality is not measured objectively | Medium | **Open.** Judge it by watching; LatentSync and SyncNet scoring are possible later. |

Every v0.5 upgrade is optional: if a model can't be installed or fails, the stage records a warning and uses the previous method. The trailer's `quality_report.json` → `optimization_notes`, and the stage warnings, show which path was taken.
