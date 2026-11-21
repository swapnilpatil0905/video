# Realistic Character Faces and Personal Cast

Status: implemented (v1) · Stage: `character_references` (v2) · Code: `src/video_platform/storytelling/casting.py`, `src/video_platform/stages/assets.py`, `src/video_platform/workers/images.py` · Config: `production.character_images`, `cast/cast.yaml`

## 1. Goals

- Every replacement character has an **original, realistic face** that fits their new Indian identity: age, region, profession, background.
- **Faces are never derived from celebrities, actors or anyone who has not agreed.** The original actor is never reproduced.
- **Consenting real people** (the project owner and spouse) can shape characters in every project. By default this is a *blend*: a new person who clearly resembles them but is not them.
- **Each decision is measured and reviewable:** candidates, scores and a contact sheet, plus manual override.

## 2. Policy (not configurable)

| Allowed | Not allowed |
|---|---|
| Fully synthetic faces from text-to-image models | Faces of celebrities, Bollywood or other actors, or public figures, alone or blended |
| Photos of people who gave consent, recorded in the config (who, date, scope) | Photos of anyone without recorded consent; photos fetched from the internet (URLs are rejected) |
| Blending a consenting person with synthetic faces | A character that resembles the original actor (ArcFace similarity > 0.45 → regenerated) |
| | A synthetic character that resembles a cast member it was not built from (> 0.32 → rejected) |

- **Cast photos stay private.** They stay in `cast/` or `projects/<id>/input/`, both gitignored, and are never stored as artifacts. Only the derived faces are.
- **The provenance records consent:** name, mode, consent details and the number of photos, in the `character_references` record.

## 3. Pipeline

```
character_design ─┐
character_identity ┼─► character_references ─► voice/TTS … ─► Wan2.2-Animate (replaces the actor in source shots)
face_clustering ───┘        │                                     └─► per-clip identity check (≥ 0.40) → LatentSync
                            └─ contact_sheet.jpg, character_references.json
```

**1. Identity-aware prompt** (`casting.face_prompt`). The prompt combines:
- the design's portrait prompt (face, skin tone, hair, build, wardrobe), plus:
- **identity context**: age, gender, region, profession and a background line. The **name is never used**, because names can pull towards real people;
- "an ordinary real person, not a celebrity or model";
- the realism suffix.

The realism suffix asks for a candid photo with natural skin texture, visible pores and small imperfections, slight asymmetry, natural hair, everyday light, a 50 mm lens and no retouching. The negative prompt excludes CGI, airbrushing, beauty filters, flawless skin, perfect symmetry, model, influencer and stock-photo looks, glamour lighting, heavy makeup and celebrities.

**2. Candidates.** `character_images.candidates` (default 3) text-to-image portraits per character with Qwen-Image, one per seed.

**3. Cast blend** (cast characters only). Qwen-Image-Edit-2509 gets the cast member's photos (up to 2) plus the synthetic candidate. The prompt is: keep the candidate's pose, framing, clothing, hair and light; give the face a *subtle / clear / strong family resemblance* to the reference person; it must **not** be the same person; natural skin; no beauty filter. `as_is` mode keeps the person's exact face, restyled into the character.

**4. Measurement.** ArcFace (InsightFace buffalo_l) measures every image:

| Check | Rule |
|---|---|
| Face found, detection score | ≥ 0.6 |
| Face size (face height / image height) | ≥ 0.10 |
| Similarity to the original actor | ≤ 0.45 |
| Similarity to other cast members | ≤ 0.32 |
| Blend: similarity to the cast member | inside `similarity` (default **0.30–0.55**: clearly related, not identical) |
| as_is: similarity to the cast member | ≥ max(low bound, 0.55) |

- **Score:** detection quality + face size − similarity to the original actor. For blends, also minus the distance from the middle of the target range. The best valid candidate wins.
- **Retries:** up to 3 attempts with new seeds. If every blend is too weak, the resemblance strength goes up one step (subtle → clear → strong). If every blend is too close, it goes down.

**5. Manual pick.** `contact_sheet.jpg` shows one row per character. Each tile is labelled with its index and its similarity to the original actor and to the cast member. Green = chosen, grey = valid, red = rejected. To override, set `character_images.pick: {GEN_CHAR_0080: 2}`. Only selection and the pose sheet re-run, because candidates are cached per item.

**6. Pose and expression sheet.** Qwen-Image-Edit generates the configured variants from the winner. Variants with identity similarity < 0.5 to the master are dropped.

**7. In the video.** Wan2.2-Animate uses the master and the sheet to replace the actor in each source shot. Each clip is checked against the master (≥ 0.40, else regenerated), then LatentSync syncs the lips.

## 4. Personal cast for every project

**Photo folders (simplest).** Put one folder of photos per person in `<repo>/input/`, which is gitignored:
- `input/Male Lead/` goes to the most important male character;
- `input/Female Lead/` goes to the most important female character.

**How the folders are used:**
- They are discovered for every project and turned into cast entries (`production.character_images.cast_folders` in `configs/default.yaml`). Consent is recorded there; without it, the run stops with an error.
- **Group photos are fine.** In each photo the person's face is chosen automatically. Single-face photos anchor the identity, and in group photos the face closest to that anchor wins. Without a solo photo, the face that appears across all the person's photos wins.
- **The blend model gets the 2 clearest solo photos.**
- HEIC files are skipped unless a JPEG copy with the same name exists.

**Appearance** (`character_images.appearance`, default for all characters): a slim, fit build and good-looking in a natural way, "like a film actor on set without heavy makeup". It is added to every portrait, blend and pose prompt; the build overrides the design's build. The realism rules still apply, so the result is attractive but real, not retouched. Use `apply_to: leads` or `none` to limit it.

**Cast file (advanced).** `cast/cast.yaml` is for explicit control:

- **File.** `cast/cast.yaml` at the repository root, gitignored, with the photos next to it. The template is `configs/cast.example.yaml`.
- **It is merged into every project** when the config is loaded. A project opts out with `production.character_images.use_global_cast: false`. A project's own `cast` entry with the same `name` takes precedence.
- **Photo changes re-run faces automatically.** Each photo's sha256 is added to the config at load time.

```yaml
cast:
  - name: me
    character: lead_male          # lead_male | lead_female (most important character of that gender) | SRC_CHAR_xxxx
    mode: blend                   # blend (new person resembling) | as_is (own face)
    similarity: [0.30, 0.55]      # optional
    photos:
      - {path: couple_1.jpg, face: right}     # group photo: which face (left | right | largest)
      - {path: me_2.jpg}
    consent: {given_by: "self", date: "2026-11-21"}
  - name: wife
    character: lead_female
    mode: blend
    photos:
      - {path: couple_1.jpg, face: left}
      - {path: wife_selfie_1.jpg}
    consent: {given_by: "wife — agreed for this personal, private project", date: "2026-11-21"}
```

**Photo guidance:** 3–5 per person. Face towards the camera, daylight, no beauty filter, glasses off if possible. Mirror selfies are fine if the face is clear. Filtered or smoothed photos make the generated faces look less real.

## 5. Outputs

- `character_references.json`, per character:
  - master and variants;
  - similarity to the original actor;
  - variant consistency;
  - `cast` (name, mode, measured similarity);
  - `chosen` and `chosen_by` (automatic or manual);
  - every candidate with its metrics and rejection reasons.
- `contact_sheet.jpg`, also exported to `sample/character_references/`.
- Metrics: characters with references, failed characters, cast characters, candidates generated, mean sheet identity.

## 6. Limitations

- The realism of the portraits depends on Qwen-Image. "Polished AI look" is reduced, not eliminated. In motion, Wan2.2-Animate can soften faces or drift in long or fast shots.
- Blend quality depends on the edit model following a multi-image instruction. It is measured, not assumed. When no candidate lands in range after 3 attempts, the character is reported and its shots fall back to the source.
- ArcFace similarity measures identity, not realism. You judge realism on the contact sheet and trailer.
- **Family resemblance between characters** (siblings, parents) is not modelled yet. Each face is generated independently.
