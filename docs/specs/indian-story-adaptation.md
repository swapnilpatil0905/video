# Indian Story Adaptation, Regional Dialogue and Cinematic Storytelling

Status: implemented (v1) · Applies to: every project · Code: `src/video_platform/storytelling/`, `src/video_platform/stages/adaptation.py` · Data: `configs/regional_profiles/*.yaml`

## 1. Objective

Turn a source story into Indian entertainment that sounds as if an Indian screenwriter had written it for Indian audiences. The system must not produce literal translations.

The core decision is the **order of work**:

| # | Phase | Stage / module | Output |
|---|---|---|---|
| 1 | Understand the source: story, characters, emotions | `narrative`, `character_design` (existing) | `narrative.json`, `character_designs.json` |
| 2 | Indian identities: names, forms of address, a speaking profile per character | `character_identity` · IndianNameGenerator, RegionalLanguageProfileManager | `character_identity.json` |
| 3 | Understand every scene before writing a word | `scene_analysis` · DialogueContextAnalyzer, ScreenplayEnhancementEngine, RegionalEnvironmentAdapter | `scenes.json` |
| 4 | Write the Indian screenplay, scene by scene, with an audio direction per line | `localization` · IndianDialogueRewriter, NaturalConversationGenerator, EmotionAndDeliveryPlanner | `localized_lines.json` |
| 5 | Evaluate timing and quality; rewrite weak lines and keep every version | `localization` · DialogueDurationOptimizer, ScreenplayQualityEvaluator, CreativeReferenceManager | `screenplay_quality.json`, `screenplay_versions.json`, `creative_references.json` |
| 6 | Emotion-aware voice generation with regional accents | `voice_design`, `sample_production` · EmotionAndDeliveryPlanner | voices, `audio_quality_report.json` |
| 7 | Lip-sync and the final video | `sample_production` / `full_production` | `trailer.mp4`, `screenplay_review.md` |

The pipeline never translates first and adds emotion afterwards. The writer always receives the analysed scene: mood, intentions, listener, subtext, relationships, location plan and the allowed enhancements.

## 2. Indian character naming (IndianNameGenerator)

- **One model call covers all characters,** so names are unique across the cast and family names stay coherent within a family. The inputs are:
  - each character's gender, age, tier, role and description;
  - the story's relationships;
  - the region's naming conventions (`naming` in the regional profile);
  - the example names from the profile. These are examples, not a fixed list.
- **Each registry entry records:**
  - given name, family name, full name and display name;
  - the name in Devanagari (what TTS reads) and a pronunciation hint;
  - nickname, family group, profession, background and personality;
  - forms of address (who calls them what, with honorifics);
  - a speaking profile.
- **Deterministic checks.** Identical or confusingly similar given names are rejected (normalised edit distance below `character_names.min_name_distance`). Related characters with different family names are flagged; marriage and in-law relations are exempt. On a problem the model is asked once more, with the taken names listed as forbidden.
- **No stereotyping.** Religion and community are never chosen for flavour. They appear only when the story implies them.
- **Consistency.** The registry is the single source for names in every later prompt. The quality check flags old design names and source names in dialogue. Source names come from the romanised names that scene analysis lists as `names_mentioned`.

## 3. Regional language and accent (RegionalLanguageProfileManager)

- Regional profiles are data files in `configs/regional_profiles/`, one per region: `hyderabad`, `neutral_hindi`, `mumbai`, `delhi`, `lucknow`, `bengaluru`, `kolkata`, `chennai`, `pune`, `jaipur`, `punjabi_hindi` and `telugu_hindi`.
- Each profile defines:
  - **dialects**: a description, and the accent phrase used for voice design;
  - **expressions**: roman and Devanagari forms, meaning, register, usage note, and the minimum dialect intensity at which a character may use them;
  - **naming conventions and honorifics**;
  - **culture**: food, festivals, everyday life;
  - **the location database** (§10).
- Hyderabad comes first: Dakhni Urdu, Hyderabadi Hindi and natural Hinglish. Writing styles are `natural_hindi`, `natural_hinglish`, `formal_hindi` and `dakhni`.
- **Every character gets its own profile:** language, region, accent, `dialect_intensity` (0–1), `english_mixing`, `urdu_influence`, speaking style and speech habits. The project's `dialect` section is the default; the model varies it per character according to background. A warning is raised when all characters share one profile, because the trailer then cannot show two accents.
- **Where a profile takes effect:**
  - Dialogue: the writer sees each speaker's profile, and only the expressions allowed at that intensity.
  - Voice: the voice-design description gets the accent phrase, e.g. "speaks Hindi with a strong Hyderabadi Dakhni accent".
  - Slang control: lines with ≥3 dialect markers, or any marker from a character with intensity < 0.25, are flagged `excessive_slang` and rewritten.
- **Script.** All dialogue is written in Devanagari, with English words spelled in Devanagari as Indians write them, so TTS pronounces it. A roman version is kept for review.

## 4. Context-aware dialogue rewriting (IndianDialogueRewriter)

The writer prompt for a batch of lines (12 by default) includes:

- the scene summary, mood, intensity, setting and relationships;
- the enhancement plan (§5);
- the real places it may mention and the fictional ones (§10);
- the characters present, with names, forms of address, personality and speaking profile;
- the regional expressions allowed;
- the four lines just before, already written where available.

Each line row carries the speaker → listener, available seconds, a syllable target and maximum (§9), the intention, the emotion and the original text.

The answer per line is `text` (Devanagari), `roman`, `short` (a shorter version), `meaning_en` and `direction` (§7).

The system prompt asks for a scene written as an Indian screenwriter would write it, not translated word by word. It must keep the meaning, intention and story information, and use each character's speaking profile. Example: "Why did you come here?" becomes "अरे, तुम यहाँ क्या कर्रे हो?" for a casual Hyderabadi friend, "May I know what brings you here?" for a professional, and "मैं तुमसे पूछ रहा हूँ, यहाँ आए क्यों?" for an angry character. The choice follows the scene analysis.

## 5. Situation-based enhancement (ScreenplayEnhancementEngine)

- **Candidates** for a scene come from its mood plus the analyzer's suggestions: comedy, romance, emotion, tragedy, suspense or confrontation.
- **A candidate is allowed only if:**
  - its `storytelling.enhance_*` switch is on, and
  - it fits the scene mood. Comedy is never added to tragedy, emotional, suspense, thriller or confrontation scenes.
- **Forced humour is avoided.** With `avoid_forced_humour`, comedy needs either a comedy scene, or a light scene (intensity ≤ 0.6) where the analyzer saw an opening.
- **Guidance per enhancement** follows the spec. For example, tragedy means restraint and silence, never speeches. Confrontation means escalation, interruptions with "—", and controlled threats.
- **Always on:** `preserve_story_logic` and `preserve_character_motivations`. `mode: faithful_localization` turns every enhancement off.

## 6. Creative references (CreativeReferenceManager)

- References teach technique only. The writer gets a technique guide written for this project: scene construction, escalation, comedy timing, emotional pacing, suspense, rhythm and cliffhangers. **No third-party text is ever put in a prompt.**
- The prompt forbids quoting or closely paraphrasing famous film or TV dialogue, songs, catchphrases or recognisable scenes.
- `creative_references.sources` lists any reference with its title, kind, licence and notes. That list is the provenance record, saved as `creative_references.json`. A local `path` to a text you have the rights to keep is used **only** for the originality check: a line sharing `max_ngram_overlap` (default 6) or more consecutive words with a reference text is flagged and rewritten.
- `allow_dialogue_copying: true`, `allow_scene_copying: true` or `require_original_expression: false` is a **configuration error**. Copying is not supported.

## 7. Emotional audio direction (EmotionAndDeliveryPlanner)

- **Each line gets a direction**, written by the writer with the scene in view:
  - emotion: one of 18, including restrained, sarcastic and whispering;
  - `emotion_detail` for combinations like "angry but controlled";
  - intensity, delivery style, speaking rate, pitch variation, volume;
  - pause before / after, voice break, overlap with the previous speaker;
  - regional accent and dialogue style.
- **Mapping onto the TTS engines:**

| Direction | Chatterbox | Every engine |
|---|---|---|
| emotion + intensity + delivery | `exaggeration` (base per emotion, ±intensity; controlled styles lower, explosive higher) | — |
| speaking rate, high drama | `cfg_weight` (lower = faster, more dramatic) | tempo fit stays within `max_speedup` |
| pause before / after | — | the line is shifted by its pause, and the slot after it is kept free (up to 30% of the slot) |
| volume (whisper … shouting) | — | gain in the dialogue stem (0.5 … 1.2) |
| voice break, crying, whisper, overlap | not synthesisable | carried by the text (broken words, "...", "—") and listed as `unsupported_direction` in `audio_quality_report.json` |

- **Engine choice.** With `emotion_aware_tts`, the TTS bake-off keeps Chatterbox unless IndicF5 is clearly more intelligible (back-transcription CER margin 0.08), because IndicF5 has no emotion controls.

## 8. Natural conversation (NaturalConversationGenerator)

- **The writer is asked for real conversation:**
  - short replies and incomplete sentences ("...");
  - interruptions ("—"), with the next speaker cutting in;
  - sparing fillers (अरे, यार, मतलब, अच्छा), repetition for emphasis, hesitation and silence;
  - varied sentence length between characters.
  
  Story facts and cause and effect must be kept.
- **Scene-level checks flag:**
  - every line being a long complete sentence (`robotic`);
  - fillers opening most lines;
  - repeated lines.

## 9. Duration and lip-sync (DialogueDurationOptimizer)

- **Timing estimate.** Spoken syllables are estimated per line: Devanagari consonants not followed by a virama, plus independent vowels, plus Latin vowel groups. The natural rate is `quality.syllables_per_second` (default 5.0). Explicit pauses add time, and the direction's speaking rate is applied.
- **The writer gets a budget for every line:** a syllable target (90% of the slot) and a hard maximum (`max_speedup` × the slot).
- **Verdicts.** `too_long` means it won't fit even with the allowed speed-up, and the short version doesn't fit either. `too_short` means under `min_fill` of a slot longer than 1.5 s. `too_long` lines are rewritten.
- **In production,** the full or short variant is chosen by the measured TTS duration, with tempo fitting within `production.tts.max_speedup`. Lip-sync uses the final audio.
- **Not implemented:** re-editing a scene when dialogue can't fit naturally. Lines that still overflow are reported in `quality_report.json`.

## 10. Regional environment and geography (RegionalEnvironmentAdapter, LocationDatabase)

- **Each profile has a location database of real places only:**
  - landmarks, neighbourhoods, roads, junctions, metro and railway stations, markets and lifestyle entries;
  - each with a Devanagari form, a kind, a **zone**, and tags such as `it_professionals`, `affluent`, `heritage` or `romance`.
- **Coverage today:**

  | City | Places |
  |---|---|
  | Hyderabad | 55 |
  | Mumbai | 39 |
  | Delhi | 32 |
  | Bengaluru | 22 |
  | Chennai | 17 |
  | Kolkata | 16 |
  | Pune | 15 |
  | Jaipur | 13 |
  | Lucknow | 11 |

  New cities are added as data, not code.
- **Geography is expressed only as zone relations:** same zone = near, a listed `adjacent_zones` pair = adjacent, anything else = far. **The database contains no travel times,** and the writer is told never to state one.
- **Real business names are avoided.** Lifestyle entries are generic ("an Irani café near Charminar"). Invented businesses, buildings and addresses go in the scene's `fictional_places` and are never presented as real.
- **The location is understood before any dialogue is written.** Scene analysis returns a `location_plan`:
  - the setting, as the story shows it;
  - `real_place`, only if the setting matches a database place;
  - `destinations` the characters talk about going to;
  - `fictional_places`;
  - `place_references_allowed`, true only if the story itself calls for naming a place.
- **Validation of the plan.** Places not in the database are dropped and recorded as `rejected`. A plan without a real place disallows place references.
- **Line checks** flag:
  - a real place not in the scene's plan (`unplanned place reference`);
  - "near / paas / bagal" between places in different zones;
  - a stated travel time next to a place.
  
  Flagged lines are rewritten.
- **Selective use.** The writer may mention only the planned places, "only if natural". The rules ask for at most one real place in a short exchange, and many scenes have none.
- **Visual scope.** The video is source footage with replaced characters, so Hyderabad appears in dialogue and character styling, not in new backgrounds. Faces are generated from the identity registry (age, region, profession, background) — see [`character-faces-and-cast.md`](character-faces-and-cast.md).

## 11. Quality control (ScreenplayQualityEvaluator)

- **Deterministic line checks:**
  - missing translation, timing (`too_long`), roman-script lines (`unnatural_hinglish`, since TTS needs Devanagari);
  - old or source names, slang density against the speaker's profile, geography, originality.
- **Scene checks:** repetition and robotic rhythm.
- **Naturalness review** by the Indic model (`models.naturalness_reviewer`, default Sarvam-M):
  - `natural` and `regional` scores (0–100);
  - flags (literal translation, unnatural Hinglish, excessive slang and more);
  - an improved line, which is accepted only if it has no more hard issues than the current one.
- **Screenplay review** per scene (`models.screenplay_reviewer`):
  - the 14 category scores: naturalness, regional authenticity, name usage, voice consistency, emotional accuracy, cultural accuracy, scene continuity, story coherence, dialogue duration, lip-sync compatibility, humour appropriateness, emotional impact, repetition and translation completeness;
  - issues per line, with a rewrite.
  
  Measured scores override model estimates for duration, name usage, repetition, completeness and lip-sync compatibility.
- **Rewrite rounds** (`quality.max_rewrite_rounds`, default 1). Lines with hard issues, or natural < `min_score` (default 70), go back to the writer with the reviewer's feedback. With `final_naturalness_check` on, rewritten lines are reviewed again.
- **Version history.** Every version is stored in `screenplay_versions.json`, including rejected ones and their reasons.
- **Low quality is never silently accepted.** Lines still failing after the rounds are marked `quality: below_threshold`, listed in `screenplay_quality.json` and as a warning, and down-weighted in trailer selection. The human approval of the trailer remains the final gate.

## 12. Sample trailer

- **Showcase selection** (`sample.showcase_moods`, `sample.showcase_min_accents`). After important characters are covered, trailer selection adds one line from each mood group the source offers:
  - tragedy, emotional or family drama;
  - comedy;
  - confrontation;
  - romance, suspense or thriller.
  
  It then adds a line from a second speaking profile if needed. Any group the source can't supply is a warning, never an invented scene.
- **Reports in `projects/<id>/sample/`:**
  - `screenplay_review.md` and `.json`: original, English meaning, adapted line (Devanagari and roman), character and accent, scene mood, location, direction, quality flags and the number of versions;
  - `audio_quality_report.json`: engine, TTS controls per line, pauses, gain, tempo, overflow, back-transcription CER and unsupported directions;
  - `screenplay_quality.json` and `creative_references.json` (from localization), plus the existing quality, cost and character-coverage reports.

## 13. Configuration

Under `localization:` in the project config. All keys are optional; defaults are shown in `src/video_platform/config/schema.py`.

```yaml
localization:
  country: India
  language: Hindi
  writing_style: natural_hinglish          # natural_hindi | natural_hinglish | formal_hindi | dakhni
  regional_profile: hyderabad              # configs/regional_profiles/<name>.yaml
  dialect: {name: dakhni_urdu, intensity: 0.65, english_mixing: 0.45, urdu_influence: 0.70, forced_slang: false}
  character_names: {mode: culturally_adaptive, region: Hyderabad, preserve_relationships: true}
  storytelling: {mode: creative_adaptation, enhance_emotions: true, enhance_comedy: true, enhance_romance: true,
                 enhance_suspense: true, enhance_tragedy: true, preserve_story_logic: true,
                 preserve_character_motivations: true, avoid_forced_humour: true}
  audio: {natural_delivery: true, emotion_aware_tts: true, conversational_pauses: true,
          character_voice_consistency: true, regional_pronunciation: true}
  creative_references: {enabled: true, learn_storytelling_techniques: true, allow_dialogue_copying: false,
                        allow_scene_copying: false, require_original_expression: true, sources: []}
  models: {analyzer: qwen3-32b, writer: qwen3-32b, naturalness_reviewer: sarvam-m, screenplay_reviewer: qwen3-32b}
  quality: {min_score: 70, max_rewrite_rounds: 1, syllables_per_second: 5.0, max_speedup: 1.12}
```

`models.<role>` may be a list. The first model that fits the GPU is used (`hardware.min_vram_gb` in the model registry).

## 14. Architecture

- **Modules are independent and model-agnostic.** They build prompts and validate answers; the stages call the models through the LLM worker (vLLM or transformers).
- **Several models collaborate:** the analyzer reads scene context, the writer writes the dialogue, an Indic model judges naturalness, and the screenplay reviewer scores the scene.
- **Model loads per run.** Each role call loads its model once. A default run does about 7 loads: identity, scenes, writer, naturalness, review, rewrite and naturalness again. That costs about 1–2 minutes each.
- **Caching:**
  - Stages are content-addressed: changing `localization.*` re-runs only identity, scenes and localization, and what depends on them.
  - Character reference images are not affected, because they depend on `character_design`.
  - A worker call identical to an earlier one in the same stage reuses its saved answer.
- **Provenance.** Every LLM answer is kept in `state/work/<stage>/<id>/llm*.answers.jsonl`, with models and versions in each stage's provenance.

## 15. Acceptance criteria and how they are checked

| # | Criterion | Check |
|---|---|---|
| 1 | Consistent Indian names | registry + `incorrect_name` flag; `name_usage` score |
| 2 | Conversational Hyderabadi Hindi | naturalness review (`natural`, `regional`); native-speaker trailer review |
| 3 | Natural Hinglish | `unnatural_hinglish` flag; naturalness review |
| 4 | Distinct speaking profiles | `distinct_profiles` metric; warning if 1; trailer shows ≥2 |
| 5 | Dialogue reflects the situation | scene analysis precedes writing; `emotional_accuracy`, `scene_continuity` scores |
| 6 | Comedy only where appropriate | enhancement gating; `inappropriate_humour` flag; `humour_appropriateness` |
| 7 | Convincing emotional scenes | direction per line; showcase mood; `emotional_impact` |
| 8 | Voice reflects intensity | direction → exaggeration / cfg / pauses / gain; `audio_quality_report.json` |
| 9 | Dialogue fits duration | syllable budget, `too_long` rewrite, measured fit + tempo |
| 10 | Consistent personalities | registry personality + profile in every prompt; `voice_consistency` |
| 11 | Trailer demonstrates capabilities | showcase moods + accents; `screenplay_review.md` |
| 12 | No copied material | prompt rule, technique-only references, n-gram originality check, copying rejected in config |

Criteria 2, 3, 5, 7 and 8 involve taste. Model scores are estimates, and the human review of the trailer is the real gate.

## 16. Known limitations

- Accents in voices come from Indic Parler-TTS descriptions and from the dialect words in the text. Accent control in open TTS models is approximate.
- Voice breaks, crying, true whispering and overlapping speech cannot be synthesised by Chatterbox or IndicF5. They are carried by the text and reported.
- Visual regionalisation is limited to characters. Backgrounds are the source footage.
- The location databases are curated and incomplete. Places outside them are never used as real places.
