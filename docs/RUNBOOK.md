# Runbook: run the whole pipeline on Vast.ai

## Installation (trailer only, ~7–14 h, ~$8–16)

### 1. Rent the machine on Vast.ai

| Setting | Value |
|---|---|
| **GPU** | 1× **RTX PRO 6000 (96 GB)**. Alternatives: H100 SXM, A100 80 GB. Must have ≥ 80 GB VRAM. |
| **Filter** | Reliability ≥ 98%, verified host, RAM ≥ 64 GB, max CUDA ≥ 12.8 (Blackwell) or ≥ 12.4 (H100/A100), download ≥ 500 Mbps |
| **Template** | **NVIDIA CUDA** (12.4 / 12.6 / 12.8, any plain CUDA template the machine accepts). Avoid app templates (ComfyUI, Hashcat, …). |
| **Launch mode** | **Jupyter + SSH** (Jupyter is the easiest way to upload the zip) |
| **Container / disk size** | **400 GB** (at least 300 GB for the trailer) |
| **On-start script** | `bash /workspace/video/scripts/onstart.sh` (optional; auto-resumes after a restart) |

The image's CUDA version doesn't matter much. The pipeline installs its own CUDA libraries, and what counts is the **host driver**: `nvidia-smi` must show `CUDA Version: 12.8` or higher on RTX PRO / Blackwell.

### 2. Upload and unzip

Upload `video-platform-0.4.0.zip` to `/workspace` (Jupyter → Upload), then open a Jupyter **Terminal**:

```bash
nvidia-smi                                   # check GPU + "CUDA Version"
cd /workspace && python3 -m zipfile -e video-platform-0.4.0.zip video && cd video
```

### 3. `.env` is already included

The zip contains a `.env` with your Hugging Face token, your `NOTIFY_EMAIL`, a random ntfy topic and the GPU price.

- **Email without any password:** notifications are forwarded to your inbox by **ntfy.sh**. They are text only, the sender is ntfy.sh, and they may land in spam the first time.
- **Optional, for emails with the trailer preview attached:** add a Gmail app password (created at https://myaccount.google.com/apppasswords):
  ```bash
  cat >> .env <<'ENV'
  SMTP_HOST=smtp.gmail.com
  SMTP_PORT=587
  SMTP_USERNAME=you@gmail.com
  SMTP_PASSWORD=your16charapppassword
  ENV
  ```
- If you rented at a different price, edit `VP_GPU_PRICE_PER_HOUR` in `.env`.

### 4. Test email (seconds, no setup needed)

```bash
bash start.sh notify-test       # "[mail] sent" → check your inbox (and spam)
```

### 5. Start the trailer (runs in the background)

```bash
nohup bash start.sh > /workspace/run.log 2>&1 &
tail -f /workspace/run.log      # Ctrl+C stops watching; the run continues
bash start.sh watch             # filtered live view: stages, ⏳ heartbeats, warnings, errors
bash start.sh progress          # one-shot snapshot: current stage, worker progress, GPU, disk
```

While a model step runs, `run.log` gets a `⏳` heartbeat every 60 s (`runtime.heartbeat_seconds`): elapsed time, GPU use/VRAM and the latest line of that worker's log (e.g. `[wan] item 7/38 (16% done) 24m elapsed, eta 2h05m` or vLLM's `Processed prompts: 45%`). Each finished stage logs its duration and key metrics.

You get an **email when the trailer is ready**, with a summary, a compressed preview and the thumbnail. If anything fails, the email contains the error.

### 6. Download and shut down

All results are in `/workspace/video/projects/project_001/sample/`:

| File | What it is |
|---|---|
| `trailer_high.mp4` | The trailer (full quality) |
| `high/trailer_with_subtitles.mp4`, `high/trailer.srt` | Hindi subtitles |
| `character_references/` | The new characters' identity photos |
| `voice_preview/` | Their voices |
| `high/quality_report.json`, `high/trailer_timeline.json`, `high/cost_report.json` | Reports |

Download the `sample/` folder (Jupyter → right-click → Download, or zip it first: `cd projects/project_001 && python3 -m zipfile -c /workspace/sample.zip sample`), then **destroy the instance**.

### Useful commands

| Command | Purpose |
|---|---|
| `bash start.sh status` | Progress and cache status |
| `bash start.sh stop` | Stop the pipeline and its model workers. Finished work stays cached; start again with `nohup bash start.sh > /workspace/run.log 2>&1 &`. |
| `tail -f /workspace/run.log` | Live log |
| `nohup bash start.sh > /workspace/run.log 2>&1 &` | Resume after an error or restart (finished work is reused) |
| `bash start.sh regen-shot SH_00042`, then `bash start.sh` | Redo one shot |
| `bash start.sh regen-line SEG_000123`, then `bash start.sh` | Redo one dialogue line |
| `bash start.sh smoke` | Optional quick test on the first 3 minutes (needs `HF_TOKEN`) |

To also compare a cheaper `balanced` trailer, set `compare_profiles: [high, balanced]` under `sample:` in `projects/project_001/config.yaml`.

## v0.5 model upgrades (automatic, each with a fallback)

| Stage | New default | Falls back to |
|---|---|---|
| Chinese ASR | FunASR Paraformer-zh (+ VAD + punctuation) | faster-whisper large-v3 |
| Diarization | DiariZen (WavLM) | pyannote community-1 |
| Faces | SCRFD + ArcFace on 2 frames/s at 720p | — |
| Emotion | emotion2vec+ large | SenseVoice |
| Active speaker | Light-ASD (trained) | lower-face-motion heuristic |
| Music/voice separation | BS-RoFormer | Demucs |
| Hindi dialogue | Qwen3 draft + **Sarvam-M** polish | Qwen3 draft |
| Hindi voices | Chatterbox **vs** IndicF5 bake-off (lower error wins) | Chatterbox |
| Character sheets | Qwen-Image-Edit-2509 | (same pipeline) |
| Missed characters | clip-level identity check + Light-ASD evidence | shot keyframe only |
| Disk | `runtime.prune_weights: auto` keeps all weights, and deletes finished models only if the remaining stages wouldn't fit (keeping `prune_disk_reserve_gb` free). `always` / `never` are also available. | keep all |

Known remaining gaps: [`docs/GAPS.md`](GAPS.md).

## Indian story adaptation (names, Hyderabadi dialogue, places, emotional voices)

Configured under `localization:` in the project config (project_001: Hyderabad, Dakhni Urdu, natural Hinglish). Order: scene analysis → Indian screenplay → quality + timing review → emotion-aware voices → lip-sync. Details: [`docs/specs/indian-story-adaptation.md`](specs/indian-story-adaptation.md).

- Change city: `localization.regional_profile: mumbai` (also `delhi`, `lucknow`, `bengaluru`, `kolkata`, `chennai`, `pune`, `jaipur`, `neutral_hindi`, …). Add a city by copying `configs/regional_profiles/hyderabad.yaml`.
- Review after the trailer: `sample/screenplay_review.md` (original vs adapted, accent, direction, flags), `sample/audio_quality_report.json`, `screenplay_quality.json`.
- Changing `localization.*` re-runs only character identity, scene analysis, localization and what depends on them (character images stay cached).

## Personal cast (your faces, blended into new characters, in every project)

Spec: [`docs/specs/character-faces-and-cast.md`](specs/character-faces-and-cast.md). Only people who agreed; never actors or celebrities.

Photos live in `input/Male Lead/` and `input/Female Lead/` (gitignored — never on GitHub). On the Mac they are zipped as `input/cast_photos.zip`. Upload that zip in Jupyter to `/workspace/video/input/`, then:

```bash
cd /workspace/video/input && python3 -m zipfile -e cast_photos.zip . && ls "Male Lead" "Female Lead"
```

Group photos are fine (the right face is picked automatically). HEIC: convert to JPEG first (Mac: `sips -s format jpeg IMG.heic --out IMG.jpg`).
For explicit control instead: `cp configs/cast.example.yaml cast/cast.yaml`.

- Default: blend — the male / female lead become new people who clearly resemble you / your wife (measured similarity 0.30–0.55).
- After the faces stage: check `sample/character_references/contact_sheet.jpg`. To choose another candidate: `production.character_images.pick: {GEN_CHAR_0080: 2}` in the project config.
- Replacing photos re-runs only the faces (and the shots that use them).

## What the run does

The first run goes through these steps:

1. **Setup** (once, idempotent): system packages and 9 isolated model environments, all from public sources. It downloads Wan2.2-Animate-14B and LatentSync. Other weights download on first use into `/workspace/vp/hf`.
2. **Imports your existing analysis** (transcript, diarization, shots, faces, characters, timeline) as cache hits, so none of it re-runs.
3. **New analysis:**
   - Source separation (music and effects kept, voices removed)
   - Speech emotion
   - Active-speaker detection
   - Speaker → character mapping
   - Character importance
4. **Story:** synopsis, roles, turning points and spoiler levels (Qwen3, self-hosted).
5. **Character design:** new fictional, photorealistic replacement characters (names, looks, wardrobe, voice description).
6. **Localization:** context-aware, colloquial Hindi for every line, including a shorter variant per line.
7. **Character references:** one master identity image per character, plus expression and pose variants edited from that master.
   - Every face is identity-checked.
   - Anything that looks too much like the original actor is regenerated.
8. **Voices:** one original designed voice per speaker. These are never clones of the source actors.
9. **Trailer selection:** about 90 s, covering every important speaking character, spread across the story, avoiding major spoilers, in chronological order.
10. **Trailer production:**
    - Hindi TTS
    - Wan2.2-Animate character replacement, using the same master reference in every shot
    - An identity check per shot (retried with a new seed if the face drifts)
    - LatentSync lip-sync
    - Music and effects bed
    - Loudness normalization
    - Subtitles
    - QA reports
11. **Emails you and stops.** No GPU work runs afterwards; download the results and destroy the instance.

## Optional: compare profiles and produce the full film

`bash start.sh` makes **two trailers from the same selection, characters, voices and Hindi lines**:

| Profile | Settings | Expected cost |
|---|---|---|
| `high` | 720p-area generation, 20 Wan steps, lip-sync on every speaking shot | baseline |
| `balanced` | 480p-area generation + Real-ESRGAN upscaling, **4-step distilled Wan** (LightX2V LoRA), SageAttention, 12 lip-sync steps, lip-sync only where faces are visible | roughly 5–10× cheaper (measured in `comparison.json`) |

Everything is in `projects/project_001/sample/`:

- `trailer_high.mp4` and `trailer_balanced.mp4`: watch them side by side.
- `comparison.json`: *measured* GPU-seconds per output second, the estimated full-film GPU-hours and cost per profile, identity scores, lip-sync coverage, TTS accuracy, and notes on any fallbacks.
- `high/` and `balanced/`: the full reports per profile (quality, coverage, timeline, cost, summary, subtitles, thumbnail).
- `voice_preview/`, `character_references/`, `character_selection.json`, `dialogue_selection.json`: shared by both profiles.

Approve the one you prefer:

```bash
bash start.sh approve balanced    # or: bash start.sh approve high
bash start.sh reject              # stop; nothing is deleted
```

- Full production uses the approved profile. Approval is bound to that exact trailer file.
- The output goes to `projects/project_001/output/`.
- **Speed-up fallbacks:** if a speed-up can't be used on your machine (for example the LoRA doesn't load, or SageAttention or Real-ESRGAN fail), the profile continues with the standard method, and `comparison.json` → `notes` says so.

## Interruptible (spot) instances: 30–60% cheaper

The pipeline caches every finished shot, line and stage, so it can resume after an interruption:

1. Rent an **interruptible** instance.
2. In the Vast.ai template, set **On-start Script** to: `bash /workspace/video/scripts/onstart.sh`
3. After any restart it runs `bash start.sh resume` in the background (log: `/workspace/vp/logs/run.log`). It continues the trailer, or the full film if one is approved. Nothing is redone.
4. To pause auto-resume: `touch /workspace/vp/logs/AUTO_RESUME_DISABLED`

## Choosing a GPU

Relative speed for Wan and LatentSync. These are rough, unmeasured estimates; your `comparison.json` gives real numbers.

| GPU | VRAM | Speed vs RTX PRO 6000 | Notes |
|---|---|---|---|
| A100 80 GB | 80 GB | ~0.8–1.0× | Cheap and mature; slower |
| **RTX PRO 6000 (Blackwell)** | 96 GB | 1× | Good value; newest architecture, so some optional speed-ups may fall back |
| H100 PCIe | 80 GB | ~1.3–1.5× | |
| **H100 SXM** | 80 GB | ~1.5–2× | Best-supported for flash-attention, SageAttention and vLLM |
| H200 | 141 GB | ~1.7–2.3× | Same compute as H100 with faster memory; worth it only near H100 price |
| **B200** | ~180 GB | ~3–5× | Fastest single GPU if available; needs recent CUDA; some optional speed-ups may fall back |
| L40S / RTX 6000 Ada | 48 GB | ~0.5–0.7× | Offloads to CPU; not recommended |

**Rule of thumb:** pick the GPU with the lowest **$/hour ÷ relative speed**. Several GPUs of the same kind divide the wait time, not the cost; chunks are split across GPUs automatically.

## GPU utilization and consistency (automatic)

| Phase | How it uses the hardware |
|---|---|
| Analysis | Independent stages run at the same time (`runtime.max_parallel_stages`, within a GPU-memory budget). The LLM and generation stages always get the GPU to themselves. |
| Story / localization | One vLLM process batches every prompt. Qwen3-32B in bf16 on 96 GB. |
| TTS | 4 parallel Chatterbox workers on ≥ 80 GB GPUs |
| Clip cutting / encoding | CPU pool, plus NVENC hardware encoding when the container supports it (checked with a real test encode) |
| Wan preprocessing | 3 parallel processes on ≥ 80 GB GPUs |
| **Wan generation** | **One persistent process per GPU** loads the 14B model once and renders its whole queue. Offloading is off on ≥ 80 GB. Clips are split across GPUs by duration. If Wan's Python interface doesn't match, it falls back to per-clip runs automatically. |
| Identity check | One batched pass per round, with a retry round for drifting clips |
| Lip-sync | 2 parallel LatentSync processes on ≥ 80 GB GPUs (or one per GPU) |

**Chunking without losing consistency:**
- Video is cut **only at the original shot boundaries**, so joins land on real cuts and are invisible.
- A shot is split only if it's longer than `production.video.max_clip_seconds` (30 s). Inside a clip, Wan keeps motion and identity continuous on its own.
- Every chunk of a character uses **the same master reference image, settings and seed**, so the result doesn't depend on which process or GPU rendered it.
- Each chunk is checked against the master identity, and drifting chunks are regenerated.
- All chunks are re-encoded once at the end into a single seamless file.

**Tuning:** set the worker counts under `production.parallel.*` (`null` = automatic). `cost_report.json` records the time spent in each phase, so you can compare settings.

## Fix individual items

```bash
bash start.sh regen-shot SH_00042      # redo one shot (all clips of it)
bash start.sh regen-line SEG_000123    # redo one dialogue line
bash start.sh                          # re-produce the trailer: only those items are regenerated
```

You can change any setting in `projects/project_001/config.yaml`, for example `production.llm.model: qwen3-32b`, `sample.duration_seconds: 60`, voice descriptions, or video and lip-sync command templates. Only the stages affected by the change re-run.

## Other commands

| Command | Purpose |
|---|---|
| `bash start.sh status` | Plan, cache status, approval state and history |
| `bash start.sh smoke` | Quick end-to-end test on the first 3 minutes, as a separate project. Runs *all* analysis and needs `HF_TOKEN`. |
| `bash start.sh test` | Unit tests |
| `PROJECT=projects/<id>/config.yaml bash start.sh` | Another project |

`.env`, optional, next to `start.sh`:

```
HF_TOKEN=hf_xxx              # only for gated models (pyannote for new projects, FLUX if you switch to it)
VP_GPU_PRICE_PER_HOUR=1.80   # your Vast.ai $/h → fills the cost estimate
```

## Troubleshooting

| Symptom | Where to look |
|---|---|
| Setup failed | `/workspace/vp/logs/setup_<runtime>.log`. Fix the problem, then run `bash start.sh` again (finished runtimes are skipped). |
| A stage failed | `projects/<id>/state/logs/<stage>.traceback.txt` and `worker_*.log`, `wan_*.log`, `latentsync.log` |
| Shots fall back to source footage | `wan_generate.log`. Command-line flags differ between Wan releases; adjust `production.video.*_command` in the config. |
| Lip-sync failed | `latentsync.log`. Adjust `production.lipsync.command`. |
| Out of GPU memory | Use an 80 GB GPU, or set `production.llm.model: qwen3-14b` (default) and keep a single instance. |
| Disk fills up, or "prefetch paused" for a model you already downloaded | A second model cache (the Vast template presets `HF_HOME=/workspace/.hf_home`). The pipeline always uses `/workspace/vp/hf`; setup warns if another cache holds data. Free it: `rm -rf /workspace/.hf_home && ln -s /workspace/vp/hf /workspace/.hf_home`. |
| `LLM returned unusable output` / many `missing` lines | Answers were cut off. Lower `localization.batch_lines` (default 12) or raise `localization.max_tokens` (default 8192). The worker log shows `[llm] attempt …: parsed / truncated / split` per pass. |
| LLM stages slow (low tokens/s) | On a 96 GB GPU the bf16 Qwen3-32B leaves ~17 GB for the KV cache. `production.llm.model: qwen3-32b-fp8` halves weights (~33 GB) and runs far more prompts at once. Changing it re-runs the story stages. |

## Honest limitations

- This is the first version and **has not been run end to end yet**. The first run is its real test, so expect to fix some model-specific flags or package versions from the logs.
- **Active-speaker detection is a heuristic** (lower-face motion plus identity matching), not a trained ASD model. Mappings below the confidence threshold stay `uncertain` and are reported.
- Wan2.2-Animate replaces **one character per shot** (the speaking or most important visible one). Other people in multi-person shots keep the source footage.
- Shots with no designed character keep the source footage (`rights.reuse_source_footage: true` for this private learning project).
- Hindi lip-sync and emotional TTS quality must be judged by watching. The quality report contains only objective metrics.
- Full-length production of 111 minutes is very long: many GPU-days. The trailer's `cost_report.json` gives a measured estimate.
