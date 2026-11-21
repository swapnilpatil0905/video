# Technical Proposal: Universal AI Video Remake, Localization and Generation Platform

| Field | Value |
|---|---|
| Status | **Approved 2026-11-21. Milestone 1 implemented** (see `docs/milestones/M1-report.md`). |
| Date | 2026-11-21 |
| First project | `project_001`: Chinese short drama → Hindi remake |
| Companion spec | [`email-approval-workflow.md`](./email-approval-workflow.md). It defines the approval gate and supersedes the terminal-based approval in Part 5.12. |
| Usage context | **Personal self-improvement learning project, non-commercial.** Open-source / open-weight models and open-source software only, with no paid AI APIs. GPUs are rented on the owner's **personal Vast.ai account**. |

> **How to read the model and licence tables.** Model capabilities, licences and VRAM figures come from my knowledge as of mid-2026. I have **not** yet checked them against the live repositories and model cards. The first deliverable of Milestone 1 is to verify each one: record the LICENSE file, the HF model-card licence, the commit/revision and the date in `model_registry/`. Entries marked **⚠ verify** are ones where I am unsure, or where I know the licence has changed in the past. Nothing should be downloaded for production until its registry entry is verified.

---

## 0. Executive Summary

- The repository already holds about **70% of a usable analysis stage** for project_001. That covers the transcript, diarization, shot detection, face clusters, a master timeline and speaker-character *candidates*. All of it is produced by 25 ad-hoc scripts with hardcoded paths. **I will keep the data and refactor the logic. Nothing gets thrown away.**
- The biggest technical gap before generation can start is **active speaker detection**. Today's matching uses scene co-occurrence only, and it is clearly biased toward the two highest-screen-time faces (§3.3).
- The platform is split into:
  - a **CPU control plane**: approval service, Postgres, object storage, job queue;
  - a **GPU data plane**: ephemeral Vast.ai workers running containerized stages.

  Every artifact is content-addressed and stored in S3-compatible storage. Vast.ai disks are only a cache.
- Orchestration uses a **lightweight custom stage graph with content-hash invalidation**, not Temporal or Prefect. That fits the core requirement ("change Hindi line → regenerate only dependent shots") better and adds less infrastructure (§11).
- The whole stack is **open-source / open-weight** and self-hostable, with no paid AI APIs. The project is personal learning (`licensing.usage: personal_learning`), so **non-commercial-licence models are allowed** and are benchmarked against the Apache/MIT options. The best model per task wins. That means the following are all in scope:
  - InsightFace (the existing embeddings are kept) and PuLID / InstantID;
  - F5-TTS / IndicF5 and XTTS-v2 (Hindi);
  - FLUX.1/2-dev / Kontext;
  - MusicGen;
  - Ultralytics YOLO.

  Licences are still *recorded* for every model, so a future commercial project can be filtered automatically. The core stack is Wan 2.2 (Animate, I2V, S2V, VACE), Qwen-Image / FLUX, LatentSync / InfiniteTalk / MuseTalk, Chatterbox / Indic Parler / IndicF5 / XTTS-v2, and Qwen3 / Qwen-VL.
- **I cannot honestly estimate full-production cost yet.** The order of magnitude is roughly **100–600 GPU-hours** on 80 GB-class GPUs for 111 minutes. The sample trailer exists to replace that range with a measured number (§17).
- A photorealistic, identity-consistent, lip-synced remake of 111 minutes is at the **frontier of open-source capability**. Plan for per-shot QA, some shots that need several attempts, and human review of flagged shots. These parts cannot be fully automated reliably (§18).

### Decisions needed from you before Milestone 1

| # | Decision | Why it matters | My recommendation |
|---|---|---|---|
| D1 | ✅ **Usage: personal learning, non-commercial** | Non-commercial models are allowed | `licensing.usage: personal_learning`. Licences are recorded, not used as a filter. |
| D2 | ✅ **Source use: personal learning** | Outputs are for private study and are not distributed | `rights.distribution: private`. Reusing the original music/SFX (M&E) is **allowed for project_001** (`reuse_source_music_and_effects: true`), because it saves GPU time and keeps the soundtrack quality. The final report watermarks outputs as "personal learning project, not for distribution". |
| D3 | ✅ **LLM: open models only** | Hindi localization quality is critical | Self-hosted **Qwen3** by default (vLLM on Vast.ai). Benchmark it against Gemma 3, Llama 3.3 and Sarvam-M ⚠ verify for natural Hindi. No API adapter is configured by default; the adapter interface stays open. |
| D4 | ⏳ **Control-plane hosting**: no VPS or domain yet (answered 2026-11-21) | Email approval needs an always-reachable HTTPS endpoint. That is only needed from **M5**. | **Deferred to M5.** M1–M4 run without a control plane: local SQLite index and local artifact store on the Mac, with Vast.ai for GPU stages. At M5 the choice is a personal VPS (any provider) plus a free dynamic-DNS hostname, running the compose stack. The email-approval design stays as specified, and the public endpoint is **not** on the Mac. |
| D7 | ✅ **Vast.ai: personal account, spend caps open** | You pay directly | `budget.max_total_spend_usd` and `max_sample_spend_usd` default to `null` (open, no cap). The launcher still enforces `max_instances: 1` and idle self-destroy, shows spend so far in `--status`, and has a `--vast-stop-all` kill switch. Set a number at any time to turn on a hard cap. |
| D5 | ✅ **Duplicate source deleted** (2026-11-21) | 22 GB free (98% full) | `midsummer _ Full Series _ Short Drama English.mp4` was deleted after sha256 matched `input/source.mp4` (`69eb4aed…7862f`). Legacy data stays on the Mac and is adopted by hardlink (no extra disk). |
| D6 | ✅ **Version control** | The directory is not a git repository. You will commit later. | Everything is created locally in this directory, with no `git init`. A `.gitignore` excludes data, weights, secrets and `.venv*`, so your first commit contains code and configs only. Push to a **personal** remote (e.g. github.com), never a work account. |

---

## 1. Architecture Diagram

```
┌──────────────────────────────── CONTROL PLANE (always-on, CPU, cheap) ───────────────────────────────┐
│                                                                                                      │
│   run.py / run.sh  ──►  Planner ──► Stage Graph (DAG + content-hash cache) ──► Work Queue (Postgres)  │
│   (CLI, any host)        │                                                         ▲                 │
│                          ▼                                                         │                 │
│                    Project Registry      Approval Service (FastAPI)  ◄── email ──  Recipient         │
│                    (Postgres)            Notification Dispatcher                                     │
│                                          GPU Launcher (Vast.ai API)                                  │
└─────────────┬──────────────────────────────────────────────┬─────────────────────────────────────────┘
              │ artifacts (content-addressed)                │ lease work items
              ▼                                              ▼
┌──────────────────────────────┐        ┌──────────────── DATA PLANE (ephemeral GPU) ───────────────────┐
│  Object Storage (MinIO, S3)  │◄──────►│  Vast.ai Worker  (Docker image per runtime family)            │
│  sources · analysis · refs   │        │  ┌──────────────────────────────────────────────────────────┐ │
│  audio · shots · renders     │        │  │ Stage executors → Model adapters → Model runtime         │ │
│  reports · model weights*    │        │  │ (load → run batch → unload; VRAM-aware; checkpoint/shot) │ │
└──────────────────────────────┘        │  └──────────────────────────────────────────────────────────┘ │
                                        │  Local NVMe = cache only (weights, decoded frames, scratch)    │
                                        └───────────────────────────────────────────────────────────────┘

PIPELINE (each box = stage with typed I/O contract; arrows = artifact dependencies)

 S0 Ingest ─┬─► S1 Video analysis ─┬─► S3 Active speaker ─► S4 Character DB ─┐
            │   (shots, tracks,     │      + speaker↔char                       │
            │    VLM scene tags)    │                                           ▼
            └─► S2 Audio analysis ──┘                               S5 Narrative understanding
                (ASR, diarization, separation, emotion, events)                 │
                                                                                ▼
                                      S6 Localization ─► S7 Timing-aware dialogue adaptation
                                                                                │
             ┌──────────────── SAMPLE (Part 5) uses the same stages S8–S15 on a selected subset ─────┐
             ▼                                                                                       │
 S8 Replacement characters ─► S10 Voices ─► S11 Shot generation ─► S12 Lip-sync ─► S13 Audio mix ─►  │
 S9 Environments ───────────────────────────────┘                                   S14 Assembly ─► S15 QA
             │                                                                                       │
             └───► APPROVAL GATE (email) ──PROCEED──► FULL PRODUCTION (S11–S15 on all shots, reuse) ──┘
```

---

## 2. Repository Architecture

The current directory is a data workspace, not a repository. Platform source files are written in this same directory (this directory) for you to commit later. They run on the **Mac** (development, tests, light analysis; public package indexes only, A1–A3), on a **personal VPS** (control plane) and on **Vast.ai** (GPU stages). **Code** is kept separate from **project data**: data lives in MinIO, cached under `projects/<id>/` on workers, and is gitignored. In the tree below, `ai-video-platform/` means this directory.

```
ai-video-platform/
├── run.py                     # CLI entry (Typer); thin
├── run.sh                     # wrapper: venv/docker detection, passes args through
├── pyproject.toml             # one package, optional extras per runtime family
├── docker/
│   ├── analysis.Dockerfile    # CUDA + ASR/diarization/vision (torch pinned for pyannote)
│   ├── generation.Dockerfile  # CUDA + diffusion video/image/lip-sync
│   ├── audio.Dockerfile       # TTS + separation + mixing
│   └── control.Dockerfile     # approval service, dispatcher, launcher (CPU)
├── docker-compose.yml         # VPS control plane: caddy, approval service, postgres, minio (+ mailpit for dev)
├── configs/
│   ├── default.yaml           # platform defaults (every project inherits)
│   ├── models.yaml            # model registry index → model_registry/*.yaml
│   ├── hardware.yaml          # per-stage GPU requirements, Vast.ai offer filters
│   ├── profiles.yaml          # fast_preview | balanced | high | max
│   └── pipeline.yaml          # stage graph declaration
├── model_registry/            # one YAML per model: licence, revision, sha256, VRAM, benchmarks
├── projects/
│   ├── project_template/      # config.yaml + README; `run.py init` copies this
│   └── project_001/           # gitignored except config.yaml + README
├── src/video_platform/
│   ├── core/                  # ids, time (seconds), errors, logging, Secret, hashing
│   ├── config/                # Pydantic config schema, loader, overrides, secret resolution
│   ├── schemas/               # versioned data contracts (§12)
│   ├── storage/               # ArtifactStore (local / S3), manifests, CAS, retention
│   ├── orchestration/         # StageGraph, planner, cache/invalidation, work queue, worker
│   ├── models/                # registry, adapter base classes, VRAM manager, weight fetcher
│   ├── ingestion/             # S0
│   ├── perception/            # S1 + S3: shots, frames, detection, tracking, faces, ASD, VLM tags
│   ├── audio_analysis/        # S2: ASR, diarization, separation, emotion, events
│   ├── characters/            # S4: identity clustering, character DB, character bible
│   ├── narrative/             # S5
│   ├── localization/          # S6 + S7 (dialogue adaptation is localization's second pass)
│   ├── generation/
│   │   ├── characters/        # S8
│   │   ├── environments/      # S9
│   │   ├── voice/             # S10
│   │   ├── video/             # S11 + shot router
│   │   └── lipsync/           # S12
│   ├── audio_production/      # S13
│   ├── compositing/           # S14
│   ├── quality/               # S15 metrics + judges + reports
│   ├── sample/                # Part 5: selection, trailer edit, coverage
│   ├── approval/              # state machine, approval service, notifications (email spec)
│   ├── compute/               # Vast.ai provider, local provider, hardware probe, cost meter
│   └── reporting/             # execution / cost / licence / final reports
├── tests/{unit,integration,acceptance}/  + tests/fixtures/ (short synthetic + licensed clips)
└── docs/
```

This differs from the suggested structure in four ways:

1. `diarization`, `face_analysis` and `active_speaker` are merged into `perception/` and `audio_analysis/`, because they share track and segment data structures.
2. `dialogue` is merged into `localization`.
3. `image_generation` and `video_generation` sit under `generation/`.
4. There is a separate Docker image per *runtime family*, because pyannote, the diffusion stacks and the TTS stacks pin conflicting torch/CUDA versions. One image would be fragile.

---

## 3. Existing Code and Asset Reuse Plan

### 3.1 Data inventory (project_001)

| Artifact | Location | Size / count | State | Plan |
|---|---|---|---|---|
| Source video | `input/source.mp4` | 1.7 GB; 6669.58 s; 720×1280; 30 fps; H.264 + AAC 44.1 kHz stereo | ✅ Valid | Becomes `projects/project_001/input/source.mp4`, **read-only**, with a sha256 recorded |
| Duplicate source | `./midsummer _ Full Series _ Short Drama English.mp4` | 1.7 GB, same byte size | Duplicate | Verify the checksum, then delete **with your approval (D5)** |
| Audio | `audio/source.wav` (16 kHz mono) and test WAVs | 221 MB | ✅ | Reuse as `analysis/audio/speech_16k.wav`. A 48 kHz stereo extract is also needed for separation and mixing (new). |
| Transcript | `transcripts/chinese_transcript{,_clean}.json` | 2755 raw / 2752 clean segments | ✅ Usable. 3 segments have zero/negative duration (ids 482, 2369, 2370). | Import as ASR v1. Benchmark against SenseVoice or Paraformer on a 10-minute slice before deciding whether to re-run. |
| Diarization | `diarization/speaker_turns.json` | 2947 turns, 9 speakers; took 79.4 min on M3 Max | ✅ | Import. Over-split check with speaker embeddings in S3 (several speakers map to the same face). |
| Shots | `scenes/scene_timeline.csv` | 2702 "scenes" (avg 2.5 s) | ✅ but mislabelled. These are **shots**, not scenes. | Import as `shots`. Scenes (groups of shots by location/time) are a new layer. |
| Frames | `scenes/frames/` | 1 midpoint frame per shot | ⚠ Insufficient for Part 4 S1 | Keep as thumbnails. Re-sample at 2–4 fps for tracking (cheap). |
| Faces | `faces/face_manifest.csv`, crops, embeddings | 3012 detections → 2947 embeddings (InsightFace buffalo_l) | ⚠ One frame per shot; licence issue | Keep for project_001 analysis-only use. Re-extract on tracks with a commercial-safe embedder if D1 = commercial. |
| Face clusters | `faces/cluster_*`, `characters/character_database.json` | DBSCAN eps 0.35 → 110 clusters, 689 noise; top: CHAR_0003 (612), CHAR_0001 (604), CHAR_0009 (191), CHAR_0006 (167), CHAR_0068 (140), CHAR_0007 (97) | ✅ Provisional | Import as provisional identities. Re-cluster on track-level embeddings. Run a merge pass (110 clusters is probably over-segmented). |
| Master timeline | `timeline/master_timeline.json` | 2749 valid segments; speaker match 1608 strong / 856 moderate / 222 weak / 63 none | ✅ Good foundation | Becomes `AudioSegment` + alignment records. Its logic is reused in `audio_analysis/alignment.py`. |
| Speaker↔character | `characters/speaker_character_{candidates,refined}.json` | 9 speakers × candidates | ⚠ Co-occurrence only | Kept as *prior* evidence. Superseded by ASD-based mapping (§15). |
| Verification clips | `verification/clips/` | 72 clips (8 per speaker) | ✅ Useful | Seed a human-labelled ASD calibration set (§15.4) |
| Model | `models/facedetection/face_detection_yunet_2023mar.onnx` | 232 KB | ✅ | Move to the weight cache and register |

### 3.2 Code inventory

| Script | Lines | Classification | Action |
|---|---|---|---|
| `extract_audio.py` | 25 | Reusable | → `ingestion/audio.py` (parameterized; add a 48 kHz stem) |
| `transcribe.py` | 25 | Reusable logic, **Apple-only backend** (`mlx_whisper`) | → `audio_analysis/asr/` with `MlxWhisperAdapter` (local Mac) and `FasterWhisperAdapter` (CUDA/Vast) |
| `test_transcribe.py` | 28 | Obsolete | Delete after migration; replaced by integration test |
| `clean_transcript.py` | 36 | **Project-specific** (hardcoded "优优独播剧场" watermark) | → generic `transcript_filters` with patterns in project config |
| `detect_scenes.py` | 55 | Reusable | → `perception/shots.py` (adapter: PySceneDetect / TransNetV2) |
| `extract_scene_frames.py` | 69 | Reusable, limited | → `perception/frames.py`. Multi-sample; frame-accurate seeking via PyAV/decord rather than `CAP_PROP_POS_MSEC`. |
| `detect_faces.py` | 127 | Reusable | → `perception/faces/detect.py` (adapter) |
| `extract_face_embeddings.py` | 96 | Reusable, licence-flagged | → `perception/faces/embed.py` (adapter) |
| `evaluate_clustering.py` | 75 | Reusable | → `characters/cluster_eval.py` |
| `cluster_characters.py` + `audit_character_clusters.py` | 223 + 133 | Duplicated DBSCAN | Merge → `characters/clustering.py` (one fit; audit is a report) |
| `create_contact_sheets.py`, `create_face_contact_sheets.py` | 76 + 74 | Duplicated | Merge → `reporting/contact_sheets.py` |
| `build_character_database.py` | 146 | Reusable | → `characters/database.py` (Pydantic `Character`) |
| `diarize_speakers.py` | 72 | Obsolete test (hardcoded 300 s) | Delete |
| `diarize_speakers_2.py` | 247 | Reusable | → `audio_analysis/diarization/pyannote.py` |
| `build_master_timeline.py` | 693 | Reusable core, well-designed (flags, not clips, bad timestamps) | → `audio_analysis/alignment.py` |
| `audit_timeline_quality.py`, `diagnose_timeline_alignment.py` | 228 + 398 | Reusable diagnostics | → `quality/analysis_checks.py` |
| `inspect_*` (3 scripts) | ~140 | Ad-hoc | Replace with `run.py --status --details` |
| `build_speaker_character_candidates.py` | 323 | Superseded by refined | Delete (refined covers it) |
| `refine_speaker_character_matching.py` | 452 | Reusable as prior | → `perception/speaker_mapping/cooccurrence.py` |
| `build_visual_speaker_audit.py` | 443 | Reusable | → `quality/review_clips.py` (human verification clips) |

Cross-cutting issues found:

- **Hardcoded CWD-relative paths in all 25 scripts.**
- `load_json`, `find_segment_list`, `first_value` and `load_scenes` are copy-pasted into three or four scripts.
- Scene IDs are strings in the timeline but ints elsewhere.
- No checksums, manifests or provenance anywhere.
- `character_id` is always `null` in the master timeline.
- Overlapping speech is reduced to a single speaker by maximum overlap.
- `find_overlaps` scans `records[:upper]` linearly. It is O(n²)-ish, but fine at this scale.

### 3.3 Evidence that speaker matching needs ASD

From `characters/speaker_character_refined.json`:

| Speaker | Speech (s) | Top candidate (single-char s) | Next candidates |
|---|---|---|---|
| SPEAKER_02 | 1368 | CHAR_0003 (662) | CHAR_0001 (196), CHAR_0068 |
| SPEAKER_05 | 946 | CHAR_0001 (491) | CHAR_0003 (122), CHAR_0006 |
| SPEAKER_00 | 426 | CHAR_0001 (73) | CHAR_0003 (55) |
| SPEAKER_04 | 410 | CHAR_0001 (74) | CHAR_0003 (47), CHAR_0014 (43) |
| SPEAKER_06/07/08/03/01 | 164–463 | CHAR_0009 / 0006 / 0068 / 0007 / 0050 | CHAR_0001 / 0003 in every list |

- Three speakers point to CHAR_0001.
- CHAR_0001 and CHAR_0003 appear in the top three for **all nine** speakers, because they are on screen most of the time (reaction shots, two-shots).

The current script correctly labels its output "not proof of speaking identity". The new pipeline must add mouth-movement and AV-sync evidence (§15).

### 3.4 Migration procedure (non-destructive)

1. Code files are written into this folder. There is no `git init` (D6). A `.gitignore` is written so a later commit picks up code and configs only.
2. Create a fresh project venv from **public PyPI** using the project `pip.conf`. The existing `.venv*` are not reused.
3. `run.py migrate-legacy --project project_001` (on the Mac):
   - writes `migration/manifest_v0.json` with sha256 for every legacy file (read-only pass);
   - **hardlinks** legacy artifacts into `projects/project_001/` (same APFS volume, so no extra disk). The originals stay where they are until you approve the cleanup.
   - converts the legacy CSV/JSON into versioned schemas (`schema_version: 1`), recording each legacy file's hash in the artifact `inputs`. The cache therefore treats the imported results as valid and **does not re-run** transcription, diarization or shot detection.
4. When the VPS is ready, `run.py storage sync --project project_001` uploads to MinIO, and the manifest is verified remotely.
5. Delete duplicates or legacy folders on the Mac **only with your explicit go-ahead** (D5).

---

## 4. Pipeline Stage Breakdown

Each stage has the following, and each stage is a pure function of its declared inputs:

- a typed input contract and a typed output contract (§12);
- a `params` block whose hash forms part of the cache key;
- a `hardware` requirement;
- the work-item granularity (whole project, per shot, per segment, per character);
- objective acceptance metrics.

| Stage | Work-item granularity | Key outputs | Hardware | Acceptance metric (initial thresholds, tuned on project_001) |
|---|---|---|---|---|
| S0 Ingest | project | `VideoMetadata`, stream checksums, `speech_16k.wav`, `full_48k.wav`, storage/processing estimate, `ingestion_report.json` | CPU | ffprobe decodes the full stream with no errors. Duration delta between container and streams < 0.1 s. Disk check passes. |
| S1 Video analysis | shot | shots (TransNetV2/PySceneDetect), scenes (shot grouping by VLM embedding + time), 2–4 fps samples, person/face tracks, VLM tags (shot type, camera angle/movement, indoor/outdoor, environment, lighting, objects), technical quality | GPU ≥ 16 GB (VLM) | Shot F1 ≥ 0.9 on a hand-labelled 5-minute slice. Track ID switches < 5% on the labelled slice. |
| S2 Audio analysis | segment | VAD, ASR (+ word timestamps), diarization, speaker embeddings, overlap detection, source separation (vocals / music / SFX), emotion, delivery (whisper, shout, cry, laugh, sing), rate, intensity, audio events | GPU ≥ 12 GB | CER on a hand-checked 10-minute slice is reported. Diarization DER is reported on a labelled 5-minute slice. Every segment has a confidence. |
| S3 Active speaker + mapping | segment | per-track speaking scores, `SpeakerCharacterMapping` with evidence, uncertainty reasons | GPU ≥ 8 GB | Precision ≥ 0.9 on accepted mappings for the calibration set. Uncertain mappings are flagged, not forced. |
| S4 Character DB | character | source `Character` records, merged clusters, importance, relationships (from S5), source character bible | CPU | Each important character has ≥ 5 high-quality reference crops. Human confirmation for main characters (see §18). |
| S5 Narrative | project → scene | synopsis, scene summaries, relationship graph, arcs, turning points, important dialogue | LLM GPU ≥ 48 GB (or API) | Every claim cites `segment_id`s. A citation-validity check passes 100%. |
| S6 Localization | segment (with scene context window) | Hindi line + alternatives, adaptation notes, glossary | LLM | Back-translation semantic similarity reported. Native-speaker review on a sample of 50 lines. Glossary terms are 100% consistent. |
| S7 Timing adaptation | segment | target duration, shorter / conversational variant, pauses, pronunciation hints (Devanagari plus transliteration for names), emotion / delivery tags | LLM + TTS duration probe | Synthesized duration within ±15% of the source slot, or the overflow is explicitly accepted. Zero dropped lines without a trace record. |
| S8 Replacement characters | character | master identity, face/body/expression/outfit/lighting sheets, identity-conditioning assets (LoRA and/or reference set), generated character bible | GPU 48–80 GB | Identity self-consistency across the sheet ≥ threshold (§16). Human approval in the sample. |
| S9 Environments | location | environment references and plates, lighting notes | GPU 24–80 GB | Continuity across connected scenes (VLM plus embedding similarity) |
| S10 Voices | character, then segment | `VoiceProfile`, per-line audio | GPU 8–24 GB | Hindi ASR back-transcription CER ≤ threshold. Speaker-similarity ≥ threshold within a character and < threshold between characters. UTMOS/DNSMOS reported. |
| S11 Shot generation | shot | generated shot video via shot router | GPU 80 GB preferred | Identity similarity, flicker, motion and artifact checks (§16) |
| S12 Lip-sync | shot | lip-synced shot | GPU 24–80 GB | SyncNet LSE-C ≥ threshold and LSE-D ≤ threshold. Identity drift ≤ threshold. |
| S13 Audio production | scene | dialogue stem, M&E (separated original or regenerated), ambience/SFX, mix, loudness | GPU 8–24 GB / CPU | Integrated loudness −14 LUFS ±1 (web/social; configurable). True peak ≤ −1 dBTP. Dialogue-to-bed ratio meets the configured value. |
| S14 Assembly | project | final timeline, encode, subtitles (SRT/VTT/ASS) | CPU | Frame count = expected. No black or freeze frames beyond intentional ones (ffmpeg `blackdetect`/`freezedetect`). A/V offset < 1 frame. |
| S15 QA | all | `quality_report.json` (objective vs. subjective separated) | GPU 16 GB | Every check has a status. Failures are quarantined, not silently passed. |

---

## 5. Model Selection Matrix

**Selection rule:** the registry holds candidates. The **benchmark harness** (Milestone 1–5) runs them on project_001 clips, and the best passing model for the project's licence policy is selected per task, then pinned by revision and sha256.

| Task | Primary candidate | Alternatives to benchmark | Notes |
|---|---|---|---|
| Shot detection | **TransNetV2** | PySceneDetect ContentDetector/Adaptive (current) | Keep the PySceneDetect results for project_001. TransNetV2 for new projects if it benchmarks better. |
| Scene/VLM tagging | **Qwen2.5-VL-7B / Qwen3-VL** (size by VRAM) | InternVL3, MiniCPM-V | Multiple frames per shot. Outputs are labelled *model judgement*. |
| Person detection | **Ultralytics YOLO11/YOLOv8** (AGPL-3.0; fine for personal learning) | RT-DETR v2 (Apache-2.0), RTMDet | |
| Tracking | **ByteTrack / BoT-SORT** | OC-SORT | Face tracks drive ASD and identity. |
| Face detection | **SCRFD (InsightFace)** for accuracy; **YuNet** (current) for speed | RetinaFace, MediaPipe Face Detector | |
| Face landmarks / mouth | **MediaPipe Face Landmarker** | — | Mouth-opening features for ASD fallback and lip-sync QA |
| Face embedding (identity) | **InsightFace ArcFace (buffalo_l / antelopev2)**, current; non-commercial weights are fine for personal learning | AdaFace ⚠ verify; DINOv2 (fallback) | The existing 2947 embeddings are **kept**. It also enables InsightFace-based identity adapters (PuLID, InstantID). |
| ASR (zh) | **Whisper large-v3 / turbo** (current) | **SenseVoice / Paraformer (FunASR)** ⚠ verify licence; WhisperX for word alignment | Run on Vast via **faster-whisper**. `mlx-whisper` only for local Mac. |
| VAD | **Silero VAD** | pyannote segmentation | |
| Diarization | **pyannote community-1** (current) | NeMo Sortformer / MSDD ⚠ verify weight licence | |
| Overlap detection | pyannote segmentation (overlap head) | — | |
| Source separation | **Demucs htdemucs / htdemucs_ft** | BS-RoFormer / Mel-RoFormer (via `audio-separator`) ⚠ verify per-weight licence | Produces the M&E stem. Reuse requires D2. |
| Speech emotion / events | **SenseVoice** (emotion + audio events), **emotion2vec+** ⚠ verify | PANNs / BEATs for events | Crying, laughing and singing come from AED plus a VLM cross-check |
| Active speaker | **Light-ASD** and **TalkNet-ASD** | LoCoNet; SyncNet AV-offset as a feature; MediaPipe mouth-motion heuristic as fallback | Trained on AVA-ActiveSpeaker (mostly non-Chinese, non-vertical video), so calibration is mandatory. ASDNet is not recommended (unmaintained research code) unless the benchmark proves otherwise. |
| Story / localization LLM | **Qwen3** self-hosted via **vLLM** on Vast.ai (32B dense or 30B-A3B MoE; 235B-A22B on multi-GPU for max quality) | Gemma 3 27B ⚠ terms; Llama 3.3 70B ⚠ community licence; Sarvam-M ⚠ verify; Aya Expanse (CC-BY-NC, fine here) | **Open-weight only (D3); no paid APIs.** Chinese comprehension is strongest in Qwen. A Hindi naturalness bake-off is run on 50 lines, and a **native reviewer** judges. |
| zh→hi MT fallback / QA | **IndicTrans2** (en↔Indic) via an English pivot; **NLLB-200** (CC-BY-NC, fine here) direct zh→hi | — | Only for back-translation QA and cross-check, not as the primary translator |
| Character image generation | **FLUX.1-dev / FLUX.2-dev** (non-commercial, fine here) and **Qwen-Image / Qwen-Image-Edit** (Apache-2.0); benchmark both | FLUX.1 Kontext dev (editing), HunyuanImage ⚠, SDXL | Qwen-Image-Edit / Kontext are used for turnaround, expression and outfit sheets from one master |
| Identity conditioning | **Per-character LoRA** trained on the approved reference sheet (Wan, FLUX or Qwen-Image), plus **PuLID / InstantID** for stills and **reference-image conditioning** in video (Wan-Animate / VACE / Phantom ⚠ verify) | IP-Adapter FaceID | Identity is never text-only (Part 4 S4) |
| Video: character replacement (keep source motion) | **Wan2.2-Animate-14B** (replacement mode) | VACE (Wan2.1) masked editing; HunyuanCustom ⚠ licence | **Default strategy for project_001.** It preserves camera intent, blocking and acting, which is the most reliable route to correct narrative and timing. |
| Video: background / environment | **Wan VACE** inpainting with SAM 2 masks | Qwen-Image-Edit plate + compositing; Wan I2V | |
| Video: I2V / T2V (new shots) | **Wan2.2 I2V-A14B / T2V-A14B**; **TI2V-5B** for previews | LTX-Video / LTX-2 ⚠ licence and revenue threshold; HunyuanVideo 1.5 ⚠ territory restrictions; CogVideoX ⚠; SkyReels-V2 (Wan-based, long video) | SVD is not recommended (dated quality, Stability community licence) |
| Lip-sync / dubbing | **LatentSync 1.6** (512 px) on replacement shots; **InfiniteTalk** ⚠ verify (Wan-based audio-driven dubbing, head and body motion) | MuseTalk 1.5 (256 px face region, fast); Wan2.2-S2V; Wav2Lip (allowed here, but 96 px and dated, so a baseline only) | Hindi is **not** a primary training language for any of these. Benchmark on Hindi TTS output. |
| Segmentation / matting | **SAM 2** | SAM 3 ⚠ verify licence; BiRefNet (MIT) | RobustVideoMatting is GPL-3.0; avoid as a linked library |
| Upscaling (720p → 1080×1920) | **SeedVR2** ⚠ verify; **Real-ESRGAN** (BSD-3) | Wan native 720p, then upscale. Assess identity and texture after upscaling. | |
| Optical flow (QA) | **RAFT** (BSD-3) | — | Warping-error flicker metric |
| Hindi TTS: voice design | **Indic Parler-TTS** (Apache-2.0, description-controlled: "a calm 30-year-old male, deep voice…") | — | Creates **original** voices with no cloning of source actors |
| Hindi TTS: consistent per-line synthesis | Benchmark **Chatterbox Multilingual** (MIT, Hindi, emotion control), **IndicF5** (AI4Bharat, F5-based, Indic data ⚠ verify), **XTTS-v2** (Hindi, CPML non-commercial, fine here) and **F5-TTS** with a Hindi fine-tune (CC-BY-NC, fine here) | CosyVoice 2/3 (no official Hindi ⚠ verify); IndexTTS2 (duration and emotion control, zh/en) | The voice identity reference is a clip **generated** by Indic Parler-TTS, never a source actor's voice. The winner is chosen on Hindi CER, UTMOS, speaker similarity and a native rating. |
| Music | **Original M&E stem** (Demucs-separated; allowed for personal learning, D2) | ACE-Step (Apache-2.0), MusicGen (CC-BY-NC, fine here) for new cues | |
| SFX / Foley | **MMAudio** ⚠ verify weight licence | Stable Audio Open ⚠ community licence; HunyuanVideo-Foley ⚠ | |
| TTS / voice QA | Whisper / IndicConformer for Hindi CER; ECAPA (SpeechBrain) for speaker similarity; UTMOS / DNSMOS for naturalness | — | |
| Video QA | SyncNet (LSE-C/D), RAFT flicker, identity cosine, ffmpeg detectors, DOVER ⚠ verify, VBench dimensions | VLM judge (subjective, labelled as such) | |
| Trailer selection solver | **OR-Tools CP-SAT** (Apache-2.0) | PuLP / HiGHS | |

---

## 6. Model Licence Assessment

Policy field in project config: `licensing.usage: personal_learning | non_commercial | commercial`. project_001 is **`personal_learning`**, so non-commercial and research-only weights are **allowed**. The planner records every licence and writes `licence_report.json`, but only blocks models whose terms forbid even personal use, or whose gated terms you have not yet accepted on Hugging Face (e.g. pyannote, FLUX.1-dev). The *Commercial?* column below matters only for a future commercial project. **Licence verification:** whenever a model card or repo is ambiguous, I will ask you for the repo or document link instead of guessing, and record it in `model_registry/`.

The rows below are my current understanding. **Every row must be verified (M1).** Code and weight licences are recorded separately.

| Model | Code licence | Weight licence | Commercial? | Notes |
|---|---|---|---|---|
| Whisper (OpenAI) | MIT | MIT | ✅ | |
| faster-whisper / CTranslate2 | MIT | (Whisper) | ✅ | |
| mlx-whisper | MIT | (Whisper) | ✅ | Apple only |
| pyannote.audio / community-1 | MIT | CC-BY-4.0 (gated; attribution) | ✅ with attribution | ⚠ verify the current model-card terms |
| SenseVoice / FunASR models | MIT (code) | FunASR model licence | ⚠ verify | |
| Silero VAD | MIT | MIT | ✅ | |
| Demucs | MIT | MIT | ✅ | |
| PySceneDetect | BSD-3 | — | ✅ | |
| TransNetV2 | MIT | MIT | ✅ ⚠ verify | |
| YuNet (OpenCV Zoo) | MIT ⚠ verify | MIT ⚠ verify | ✅ | |
| InsightFace (buffalo_l, antelopev2, SCRFD) | MIT | **Non-commercial research only** | ❌ | Currently used in `extract_face_embeddings.py`. Commercial licences are sold by InsightFace. |
| MediaPipe | Apache-2.0 | Apache-2.0 | ✅ | |
| Ultralytics YOLO | **AGPL-3.0** | AGPL-3.0 | ❌ unless licensed | Avoid |
| RT-DETR (HF) | Apache-2.0 | Apache-2.0 ⚠ verify | ✅ | |
| TalkNet-ASD / Light-ASD | MIT ⚠ verify | trained on AVA ⚠ verify | ⚠ | |
| SyncNet (syncnet_python) | MIT ⚠ verify | ⚠ verify | ⚠ | Used for QA metrics only |
| Qwen3, Qwen2.5-VL (most sizes), Qwen-Image / Edit | Apache-2.0 | Apache-2.0 | ✅ | Some Qwen2.5-VL sizes (72B, 3B) use other Qwen licences ⚠ verify per size |
| IndicTrans2 | MIT | MIT | ✅ ⚠ verify | |
| Wan 2.1 / 2.2 (incl. Animate, S2V, VACE) | Apache-2.0 | Apache-2.0 | ✅ | Wan 2.5+ were API-first at my knowledge cutoff ⚠ verify the open-weight status of newer versions |
| InfiniteTalk / MultiTalk | Apache-2.0 ⚠ verify | ⚠ verify | ⚠ | |
| LTX-Video / LTX-2 | Apache-2.0 / custom ⚠ | LTXV open-weights licence (revenue threshold) ⚠ | ⚠ | |
| HunyuanVideo (1.0/1.5, Avatar, Custom, Foley) | Tencent Hunyuan Community | same | ⚠ excludes EU/UK/South Korea; MAU cap | |
| CogVideoX | Apache-2.0 | 2B Apache; 5B CogVideoX licence ⚠ | ⚠ | |
| Stable Video Diffusion / Stable Audio Open | — | Stability Community (revenue cap) | ⚠ | |
| FLUX.1/FLUX.2 [dev], Kontext [dev] | — | **Non-commercial** (outputs clause ⚠ verify) | ❌ | schnell is Apache-2.0 |
| LatentSync | Apache-2.0 | Apache-2.0 ⚠ verify | ✅ | |
| MuseTalk | MIT | ⚠ verify (MuseTalk weights plus SD-VAE, Whisper, DWPose, face-parsing components) | ⚠ | |
| Wav2Lip | **Non-commercial** | **Non-commercial** | ❌ | |
| SAM 2 | Apache-2.0 | Apache-2.0 | ✅ | SAM 3 ⚠ verify |
| Real-ESRGAN | BSD-3 | BSD-3 | ✅ | |
| SeedVR2 | Apache-2.0 ⚠ verify | ⚠ verify | ⚠ | |
| Indic Parler-TTS | Apache-2.0 | Apache-2.0 | ✅ | |
| Chatterbox (Multilingual) | MIT | MIT | ✅ | Built-in PerTh watermark (keep it on; it supports transparency) |
| XTTS-v2 | MPL-2.0 (code) | **Coqui Public Model Licence, non-commercial** | ❌ | Coqui is defunct, so no commercial licence path exists |
| F5-TTS | MIT | **CC-BY-NC-4.0** (Emilia data) | ❌ | |
| CosyVoice 2/3 | Apache-2.0 | Apache-2.0 ⚠ verify | ✅ | No official Hindi ⚠ verify |
| ACE-Step | Apache-2.0 | Apache-2.0 ⚠ verify | ✅ | |
| MusicGen | MIT | **CC-BY-NC-4.0** | ❌ | |
| MMAudio | MIT | ⚠ verify | ⚠ | |
| OR-Tools | Apache-2.0 | — | ✅ | |

Consent and rights tracking:

- `rights.yaml` per project records: source-video rights holder, scope (remake / adaptation / reuse of footage / reuse of M&E), evidence reference, and expiry.
- Every source-derived artifact used **in output** (not just in analysis) carries `rights_basis`. The assembly stage refuses to include source pixels or audio in the final output unless a matching rights record exists.
- Source voices are **never** used as TTS references. The voice stage asserts that every `VoiceProfile.reference_origin == "generated"`, or that it is a licensed voice with a consent record.
- Generated replacement faces are checked for similarity against the source-actor embeddings. If similarity exceeds a configurable threshold, the face is flagged as "too similar", to avoid recreating a real person.

---

## 7. Model Compatibility Assessment

| Concern | Finding | Mitigation |
|---|---|---|
| torch / CUDA pinning | pyannote 4.x, diffusion stacks (Wan, LatentSync) and TTS repos pin different torch and transformers versions | Use 3–4 runtime-family Docker images (§2). Each adapter declares its `runtime`. The worker only leases work items its image supports. |
| Apple Silicon vs. CUDA | `mlx_whisper` (current) is Mac-only. Most generation models are CUDA-only or very slow on MPS. | Mac stays a dev and analysis-only environment. Every production adapter targets CUDA. |
| Research-repo code | Many generation repos are scripts, not libraries. Some have hardcoded paths, global state or no pip package. | Wrap them in **subprocess adapters** with a pinned commit, a JSON request/response contract and a timeout. Use in-process adapters only for well-packaged libraries. |
| Aspect ratio | The project is 9:16 vertical. Most video models are trained mostly on 16:9. Wan supports arbitrary resolutions such as 720×1280 (⚠ benchmark quality). ASD and face models are trained on landscape video. | Treat 9:16 quality as a benchmark item in the sample. Vertical crops are fine for face-centric models. |
| Clip length | Wan generates about 5 s (81 frames @ 16 fps) per pass. Wan-Animate and InfiniteTalk support longer clips by chunked continuation with overlap ⚠ verify limits. | Shot router splits shots longer than N s into chunks with an overlap and a last-frame or motion-frame carryover. Seams are QA-checked. |
| Frame rate | Wan generates at 16 fps, and TI2V-5B at 24 fps. Target is 30 fps. | Use the native generation fps, then interpolate (RIFE ⚠ verify licence) or, for Animate, follow source motion and retime. A QA check covers judder. |
| Hindi phonemes in lip-sync | Audio encoders (Whisper / wav2vec features) are language-agnostic, but the training data is mostly English and Chinese | Benchmark LSE-C/D and a human rating on Hindi in the sample. Fallback order: LatentSync → InfiniteTalk → MuseTalk. |
| Multi-person shots | Wan-Animate replaces one subject per pass. ASD and lip-sync usually operate on one face. | Multi-pass per character with SAM 2 masks. Order of passes: background → characters (far to near) → lip-sync per speaking face. Higher failure rate expected (§18). |
| LLM context | 2752 segments of Chinese text is about 60–90k tokens | Hierarchical summarization: scene, then act, then global. Localization uses a sliding window plus the global character bible and glossary. |

---

## 8. Hardware and VRAM Requirements

These are planning figures from published model cards and general experience. **Every figure will be measured by the benchmark harness and stored in `model_registry/<model>.yaml → benchmarks`.**

| Workload | Model | Min VRAM (with offload / quantization) | Recommended VRAM | Notes |
|---|---|---|---|---|
| ASR | Whisper large-v3 (faster-whisper fp16) | 6 GB | 10 GB | |
| Diarization | pyannote community-1 | 4 GB | 8 GB | The current Mac run took 79 min. A CUDA GPU should be much faster ⚠ measure. |
| Separation | Demucs htdemucs | 4 GB | 8 GB | Chunked |
| VLM tagging | Qwen2.5-VL-7B bf16 | 16 GB | 24 GB | |
| ASD | Light-ASD / TalkNet | 2 GB | 8 GB | |
| LLM | Qwen3-32B (AWQ/int4) | 24 GB | 48 GB | 80 GB for bf16 30B-A3B with long context |
| Image generation | Qwen-Image (20B) | 24 GB (fp8 + offload) | 48–80 GB | |
| LoRA training | Wan2.2 / Qwen-Image LoRA | 24 GB (low-rank, offload) | 48–80 GB | About 30–90 min per character ⚠ measure |
| Video generation | Wan2.2 A14B (T2V/I2V/Animate) | 24 GB (fp8 + block-swap, slow) | **80 GB** | Official guidance is 80 GB for single-GPU 720p |
| Video preview | Wan2.2 TI2V-5B | 24 GB | 24–32 GB | 720p @ 24 fps |
| Lip-sync | LatentSync 1.6 | 18 GB ⚠ | 24 GB | MuseTalk: about 6 GB |
| Lip-sync / dubbing | InfiniteTalk (14B) | 24 GB (low-VRAM mode) ⚠ | 80 GB | |
| TTS | Chatterbox / Indic Parler | 4 GB | 8–12 GB | |
| Upscale | SeedVR2 3B/7B | 16–24 GB ⚠ | 48–80 GB | |

Vast.ai GPU classes and how to use them:

- **RTX 4090 / 5090** (24 / 32 GB): analysis, TTS, MuseTalk/LatentSync, TI2V-5B previews.
- **L40S / A6000** (48 GB): LLM, image generation, LoRA training.
- **A100 / H100 80 GB**: Wan A14B / Animate / InfiniteTalk at 720p.
- **H200** (141 GB): only if benchmarks show a throughput or cost win.

`configs/hardware.yaml` maps each stage to `{min_vram_gb, preferred_gpu_classes, max_price_per_hour, disk_gb, cuda_min}`.

---

## 9. Vast.ai Execution Architecture

```
GPU Launcher (control plane, every 60 s)
  ├─ reads queued work items grouped by required runtime + hardware class
  ├─ searches offers via Vast.ai API (filters: gpu_ram, cuda_vers, reliability ≥ 0.98, inet_down,
  │   disk, dph_total ≤ budget, verified hosts)
  ├─ creates instance with image docker/<family>, env (storage creds via Vast env / secrets), onstart → worker
  └─ enforces caps: max concurrent instances, project budget ceiling, max idle minutes

Worker (inside instance)
  1. hardware probe → nvidia-smi, torch.cuda, VRAM, CUDA, disk → register capabilities
  2. lease work items matching capabilities (Postgres SKIP LOCKED; heartbeat 60 s; lease 10 min)
  3. fetch inputs from object storage → local NVMe cache (verify sha256)
  4. fetch weights (HF revision pinned → local cache; optional mirror bucket) → verify sha256
  5. run adapter (VRAM manager: load → batch → unload); checkpoint per item
  6. upload outputs → write artifact records → complete item
  7. idle N minutes → self-terminate via Vast API (destroy, not stop, unless volume reuse is configured)
```

| Mode | Description |
|---|---|
| `local` | Mac or local CUDA box. Same worker loop with `LocalComputeProvider`. Used for analysis (current), development and tests. |
| `single` | One Vast.ai instance runs every stage sequentially, loading and unloading models. Simplest; recommended for the first sample. |
| `distributed` | Multiple instances, each for a hardware class (e.g. 4090s for TTS and lip-sync, H100s for Wan). No code change, because work items are already per shot or per line. |

- **Interruptible ("bid") instances** are supported. Work is checkpointed per item (and per chunk for long shots), so preemption loses at most one item.
- **Persistence:** object storage is the source of truth. Vast "volumes" are tied to a host machine, so they are an optional cache only, never authoritative.
- **Personal Vast.ai account (D7).**
  - `VAST_API_KEY` is read from the environment or a local `.env` (gitignored) and is never written to config or logs.
  - Default guardrails, set in `configs/hardware.yaml` and overridable per project:

    | Guardrail | Default |
    |---|---|
    | `max_instances` | 1 |
    | `max_price_per_hour_usd` | `null` = open |
    | `max_total_spend_usd` | `null` = open (your choice); a number turns on a hard cap |
    | `max_instance_hours` | 24 |
    | `idle_destroy_minutes` | 10 |
    | `require_verified_hosts` | true |

  - If a cap is set, the launcher **refuses to start** an instance when the projected spend would exceed it. With caps open, it launches but still shows running spend.
  - `run.py --vast-status` shows running instances and spend so far. `run.py --vast-stop-all` is the kill switch.
  - Account credit top-ups and billing remain manual in the Vast.ai console. The platform never handles payment.
  - Optional **manual mode** (`compute.launcher: manual`): you start the instance yourself from the Vast console with the published image, and the worker just drains the queue.
- **Cost meter:** at instance creation, record `dph_total` (the hourly price) from the API. Bill `wall_time × dph` to the work items executed, prorated by item GPU time, and roll the totals up into `cost_report.json`.

---

## 10. Storage Architecture

The default backend is **MinIO** (open source, S3 API) on the personal VPS. The `ArtifactStore` interface also supports the local filesystem (Mac dev / tests) and any S3-compatible endpoint. Vast.ai workers use the S3 API over HTTPS with **per-project scoped access keys**.

```
s3://<bucket>/            # MinIO bucket on the VPS by default
  projects/<project_uuid>/
    source/<sha256>.mp4                     # immutable, never deleted by platform
    artifacts/<stage>/<artifact_id>/<file>  # content-addressed outputs (artifact_id = hash of inputs+params+model rev)
    samples/v<N>/...                        # trailer + review assets (email spec)
    renders/<run_id>/final.mp4, subtitles/, reports/
    quarantine/<artifact_id>/...            # failed QA, kept for inspection
  weights-mirror/<model_id>/<revision>/...  # optional mirror for faster/safer pulls
```

- **Artifact record** (Postgres plus a JSON sidecar):
  - `artifact_id` = sha256 of (stage, stage_version, sorted input artifact ids, params hash, model id + revision + weight sha256);
  - output file checksums;
  - size; timestamps; processing duration; metrics; status (`valid` / `quarantined` / `superseded`).
- **Large files** are streamed through ffmpeg/PyAV pipes and decoded in chunks. Whole videos are never loaded into RAM, and uploads are multipart.
- **Retention policy** is set per project:
  - `temp` (decoded frames, scratch): deleted after the stage completes;
  - `intermediate` (generation attempts that were not selected): kept for N days;
  - `approved` and `final`: kept indefinitely;
  - `source`: never deleted.
- **Layout separation:** code (repo), weights (`$VP_WEIGHTS_DIR` / mirror), sources (`projects/*/input`), and generated outputs (`$VP_ARTIFACT_ROOT` / bucket). Generated files are never written inside model repo checkouts.

---

## 11. Orchestration Strategy

**Recommendation: build a small in-house stage graph with a content-addressed cache (in the spirit of make, DVC or Dagster assets), backed by Postgres for work items. Do not adopt Temporal or Prefect now.**

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| **Custom StageGraph + Postgres queue** | Exact fit for artifact-level invalidation. The same Postgres is already needed for approval. No extra service. Easy to test. | We own the code, about 1–1.5k lines | ✅ **Recommended** |
| Dagster | Asset model and lineage UI are close to our needs | Heavy daemon and webserver. Per-shot dynamic partitions are awkward at about 2.7k shots × attempts. | Reconsider if a UI becomes a priority |
| Prefect | Easy retries and UI | Task-level, not artifact-level, caching. Needs a server or cloud. | No |
| Temporal | Excellent durability | Operationally heavy. Overkill for batch media jobs. | No |

How the stage graph works:

- `pipeline.yaml` declares stages, their inputs (other stages' outputs or config keys), work-item fan-out (`per: shot | segment | character | project`) and the runtime.
- The **planner** computes the expected `artifact_id` for every work item. Any item with an existing `valid` artifact is a cache hit and is skipped. Changing a Hindi line therefore changes the hash of only that line's TTS → lip-sync → shot → scene mix → assembly. Face and shot detection hashes stay the same, so they never re-run.
- **Invalidation is computed, not manual:**
  - `--regenerate-shot X` marks shot X's current artifact `superseded` and adds an attempt salt;
  - `--regenerate-dialogue Y` does the same for line Y;
  - character design changes propagate to every shot that references that `character_id`.
- **Retries:**
  - each item has `max_attempts`;
  - errors are classified as `transient` (OOM, network, preemption) → retry; `model_failure` (QA fail) → retry with a new seed or the fallback model from the registry; or `permanent` (bad input) → fail and report;
  - **OOM recovery** steps down: smaller batch → enable offload → lower tiling → alternate model.
- **Parallelism:**
  - items without dependencies run concurrently across workers;
  - within a worker they are batched by model to minimize load/unload cost (e.g. run all Chatterbox lines, then all LatentSync shots).
- **Progress and logs:** structured JSON logs (structlog) per item. `--status` aggregates counts per stage per status, ETA (from measured throughput), and cost so far.

---

## 12. Data Schemas (Pydantic v2, versioned)

All timestamps are `float` seconds on the **source timeline** (canonical), and source timestamps are preserved. Every IDs is a stable string, as follows:

| Kind | ID format |
|---|---|
| Shots | `SH_00001` (from shot order) |
| Scenes | `SC_0001` |
| Audio segments | `SEG_000001` |
| Speakers | `SPK_00` (from diarization label) |
| Source characters | `SRC_CHAR_0001` |
| Replacement characters | `GEN_CHAR_0001` |
| Voices | `VOICE_0001` |

Legacy `CHAR_0001` IDs map to `SRC_CHAR_0001` through an alias table, so existing reports remain readable.

```python
class ArtifactRef(BaseModel):
    artifact_id: str; uri: str; sha256: str; media_type: str

class ArtifactRecord(BaseModel):            # every generated artifact
    schema_version: Literal[1] = 1
    project_id: str; stage: str; stage_version: str; artifact_id: str
    work_item: str                           # e.g. "shot:SH_00412"
    model_id: str | None; model_revision: str | None; weights_sha256: str | None
    inputs: list[ArtifactRef]; outputs: list[ArtifactRef]
    params: dict; params_hash: str
    created_at: datetime; processing_seconds: float; gpu_type: str | None
    metrics: dict[str, Metric]               # Metric = value, unit, kind: objective|subjective, threshold, passed
    status: Literal["valid", "quarantined", "superseded", "failed"]
    rights_basis: str | None

class VideoMetadata(BaseModel):
    path_sha256: str; container: str; duration_s: float; width: int; height: int
    display_aspect: str; fps: Fraction; video_codec: str; pix_fmt: str
    audio_streams: list[AudioStreamInfo]; rotation: int; bitrate: int | None

class Shot(BaseModel):
    shot_id: str; scene_id: str | None; start_s: float; end_s: float
    shot_type: Labeled[ShotType]             # Labeled[T] = value, confidence, source(model_id)
    camera_angle: Labeled[str]; camera_motion: Labeled[str]
    indoor_outdoor: Labeled[str]; environment_id: str | None; lighting: Labeled[str]
    objects: list[Labeled[str]]; tracks: list[str]; keyframes: list[ArtifactRef]
    technical_quality: dict[str, float]

class FaceTrack(BaseModel):
    track_id: str; shot_id: str; start_s: float; end_s: float
    boxes: ArtifactRef                       # per-frame boxes in parquet
    embedding: list[float] | None; quality: float
    character_id: str | None; character_confidence: float | None
    asd_scores: ArtifactRef | None           # per-frame speaking probability

class AudioSegment(BaseModel):
    segment_id: str; start_s: float; end_s: float
    text: str; language: str; asr_confidence: float; words: list[Word] | None
    speaker_id: str | None; speaker_confidence: float; overlapping_speakers: list[str]
    emotion: Labeled[Emotion]; delivery: list[Labeled[Delivery]]   # whisper, shout, cry, laugh, sing, narration, offscreen
    speaking_rate_sps: float; intensity_db: float; acoustic: dict[str, float]
    flags: list[str]                         # e.g. "invalid_timestamp", "asr_hallucination_suspect"

class SpeakerCharacterMapping(BaseModel):
    speaker_id: str; character_id: str | None   # None = could not be reliably matched
    status: Literal["accepted", "uncertain", "rejected", "human_confirmed"]
    confidence: float; supporting_scenes: int; supporting_segments: int
    supporting_speech_s: float
    evidence: list[Evidence]                 # kind: asd|cooccurrence|voice_cluster|human, weight, refs
    contradictions: list[Evidence]; uncertainty_reason: str | None

class Character(BaseModel):
    character_id: str; kind: Literal["source", "replacement"]
    source_character_id: str | None          # for replacement
    display_name: str | None; importance: Importance; importance_score: float
    appearance: AppearanceSpec               # hair, clothing, accessories, body, face (structured)
    references: list[ArtifactRef]; identity_assets: list[ArtifactRef]   # LoRA, ref sets
    scene_ids: list[str]; screen_time_s: float; dialogue_s: float
    speaker_ids: list[str]; voice_id: str | None
    relationships: list[Relationship]; identity_confidence: float

class LocalizedLine(BaseModel):
    segment_id: str; character_id: str | None; speaker_id: str | None; voice_id: str | None
    source_text: str; target_text: str; target_language: str
    alternatives: dict[Literal["shorter", "conversational"], str]
    emotion: Emotion; delivery_instructions: str
    target_duration_s: float; speaking_rate: float; pauses: list[Pause]
    pronunciation: list[PronunciationHint]
    adaptation_log: list[Change]             # every rewrite or shortening is traceable
    status: Literal["draft", "reviewed", "approved"]

class ShotPlan(BaseModel):                   # shot router output
    shot_id: str; strategy: Literal["animate_replace", "vace_edit", "i2v", "t2v", "keep_source", "hybrid"]
    model_id: str; profile: str; chunks: list[ChunkSpec]
    characters: list[CharacterInShot]        # source→replacement, mask refs, speaking flags
    environment_id: str | None; lines: list[str]; seed: int
```

Further schemas: `VoiceProfile`, `GeneratedAudio`, `GeneratedImage`, `GeneratedVideo`, `LipSyncResult`, `QualityResult`, `Environment`, `ModelRecord`, `ExecutionRecord`, `ProjectConfig`, `TrailerTimelineEntry`. They follow the same pattern.

Legacy formats such as Whisper JSON, pyannote turns and InsightFace outputs are converted inside adapters and never leak past them.

---

## 13. Sample-Trailer Architecture

The sample is the **same stage graph, run on a selected subset of work items**, with the `sample` profile and resolution. There is no separate preview pipeline.

```
analysis (complete) ─► S5 narrative ─► Character Selection ─► Segment Candidates ─► Trailer Solver (CP-SAT)
                                                                                         │
                                       ┌─────────────────────────────────────────────────┘
                                       ▼
                    Trailer Edit Planner (LLM, constrained) ─► trailer_timeline.json (shots + order + transitions)
                                       │
          S8 chars ─► S9 envs ─► S10 voices (+voice_preview/) ─► S11 shots ─► S12 lip-sync ─► S13 mix ─► S14 edit
                                       │
                                       ▼
                     S15 trailer QA ─► reports ─► review assets upload ─► APPROVAL (email spec)
                                       │
                   PROCEED → full production reuses every approved artifact_id (cache hits)
                   STOP / revise → targeted regeneration commands → new sample version → new approval
```

- **Segment candidates** are contiguous source spans of 2–8 s that are aligned to shot boundaries. Each candidate carries:
  - its characters (from tracks) and speaking character (from S3);
  - emotion and delivery;
  - narrative tags from S5 (act, turning point, conflict, spoiler level);
  - visual quality and generation feasibility (face size, occlusion, number of people, motion complexity);
  - shot type and location.
- **Trailer edit planner:**
  - The LLM receives *only* the solver-selected candidates and their summaries. It proposes an order and structure: hook → introductions → rising conflict → emotional beats → closing hook.
  - It may pick transitions and music cues, and it may add **title cards** (no fabricated dialogue).
  - A validator rejects plans that reference segments outside the candidate set, reorder events in a way that contradicts S5 causality (configurable), or exceed the duration.
- **Approval actions** (Part 5.12), mapped to CLI and stage-graph operations:

  | Action | Mapped to |
  |---|---|
  | approve | email PROCEED, or the admin `--approve-sample` |
  | reject | email STOP |
  | regenerate all | `--mode sample --regenerate all` |
  | regenerate one shot | `--regenerate-shot` |
  | change character appearance | `--edit-character GEN_CHAR_x --from file.yaml` |
  | change voice | `--regenerate-voice` |
  | rewrite dialogue | `--edit-dialogue SEG --text …` |
  | improve lip-sync | `--regenerate-shot X --stage lipsync --model …` |
  | change style or environment | config change, after which the cache invalidates the dependent artifacts automatically |
  | retry failed generation | `--retry` |

  Each action creates a new sample version and a new approval request.
- **Reconciliation with the email spec:** `--approve-sample` stays as an **admin override** (✅ confirmed 2026-11-21; `notifications.approval.allow_cli_override: true`). It records a decision with `actor=cli:<user>` through the same state-machine transaction and audit log, so production is still authorized only by an explicit recorded decision.

Outputs exactly as specified in Part 5.10. `production_summary.json` doubles as the data source for the email template.

---

## 14. Character Selection Algorithm

1. **Inputs per source character `c`:** screen time `T_c` (from tracks), appearance count `A_c` (scenes), dialogue segments `D_c` and speaking duration `S_c` (from accepted or uncertain mappings), narrative centrality `N_c` (S5 relationship-graph PageRank plus LLM role label), recurrence `R_c` (fraction of acts with an appearance), and identity confidence `I_c`.
2. **Normalize** each feature to [0, 1] with rank percentiles, which are robust to the heavy-tailed distribution seen here (612 vs. 15 faces).
3. **Importance score:** `score = 0.25·S + 0.20·T + 0.20·N + 0.15·A + 0.10·D + 0.10·R`. The weights are configurable in `sample.selection.weights`.
4. **Tiering:**
   - `main`: LLM role ∈ {protagonist, antagonist, lead} **or** score ≥ 0.75;
   - `supporting`: ≥ 0.45;
   - `recurring_secondary`: appears in ≥ 2 acts and ≥ N scenes;
   - `minor`: anything else with dialogue;
   - `background`: no dialogue and screen time below threshold.

   Then a gap check: if the score distribution has a natural break within ±0.05 of a threshold, the threshold moves to the break and the report says so.
5. **Default selection:** all `main + supporting + recurring_secondary`, with **no top-K cap**. `minor` characters are included if the duration budget allows.
6. **Speaking requirement:** for each selected speaking character, the solver must include **at least one segment where that character is the ASD-verified speaker on screen** (`speaking_shot`). A silent appearance does not satisfy it. If none is feasible (for example the character only ever speaks off-screen), the character is marked `speaking_shot_infeasible` with a reason. In that case the voice preview is still generated.
7. **Uncertain identity:** characters with `I_c` below threshold, or with only `uncertain` speaker mappings, are still selected but flagged. The coverage report lists them, and the email highlights them.
8. **Trailer solver** (CP-SAT). Binary `x_i` per candidate segment.
   - Constraints:
     - Σ duration ≤ budget (±5%);
     - coverage per selected character (visible ≥ 1; speaking ≥ 1);
     - narrative buckets (first/middle/last-third quotas);
     - no more than `k` segments from any 10-minute window (prevents opening-heavy selection);
     - spoiler cap: segments tagged `major_twist` or `ending` ≤ the configured maximum (default 0 for twists).
   - Objective: maximize Σ (story importance + emotional diversity bonus per new emotion covered + dialogue quality + visual quality + feasibility). Minimize the number of multi-person segments beyond what coverage requires, since they are the most expensive and riskiest.
   - Emotion coverage is a soft objective, so emotions absent from the source are never forced.
9. **Outputs:** `character_selection.json` (per Part 5.2) and `dialogue_selection.json` (selected segments with a reason vector).

For project_001, the current evidence suggests about 6–9 important characters: the 9 diarized speakers, and face clusters with ≥ 90 faces (CHAR_0003, 0001, 0009, 0006, 0068, 0007), plus 0014 and 0050 with clear speaker links. Cluster merging may reduce this number. The algorithm decides; this is only a sanity check.

---

## 15. Speaker-Character Matching Strategy

### 15.1 Evidence sources

| Evidence | Signal | Weight basis |
|---|---|---|
| **ASD (primary)** | Per-frame speaking probability for each face track in the segment, from Light-ASD / TalkNet over audio + face crops | Calibrated probability |
| AV-sync | SyncNet offset/confidence between the track's mouth region and the segment audio | Feature into calibration |
| Mouth motion | MediaPipe lip-aperture variance aligned with speech energy | Fallback when the ASD model is unsure (profile view, small face) |
| Voice clustering | Speaker embeddings (pyannote/ECAPA) checked for over-split diarization labels that should merge | Structural prior |
| Co-occurrence | The existing `refine_speaker_character_matching` output | Weak prior only, and **capped**. It cannot alone produce `accepted`. |
| Human | Confirmations made on verification clips | Overrides, recorded as evidence |

### 15.2 Per-segment attribution

For each diarized segment, collect the face tracks visible during it:

- Speaker = argmax track **if** `p_max ≥ τ_accept` **and** `p_max − p_second ≥ margin` **and** the track covers ≥ 50% of the segment.
- Otherwise the segment is labelled one of: `offscreen` (no face tracks with mouth visible, or all `p < τ_low`), `ambiguous` (two high scores, possibly simultaneous speech; checked against overlap detection), `voice_over` (narration delivery tag plus no ASD support), or `insufficient_visibility` (faces too small, occluded or turned away).

### 15.3 Aggregation to `speaker_id → character_id`

- Build an evidence matrix `E[speaker, character]` from the sum of attributed speech seconds weighted by ASD probability. Keep separate counts for scenes and segments, plus contradiction counts (segments where a *different* character was the confident ASD speaker for this voice).
- The assignment allows **many speakers to one character** (diarization over-split) only when the voice embeddings support a merge. **One speaker to many characters** gets flagged (possible diarization under-split, or doubling actors).
- `confidence` is the posterior from a simple calibrated model (logistic regression on: share of the speaker's attributed time going to the top character, margin to the second, supporting segments, contradiction rate, mean ASD probability), fit on the calibration set.
- Status rules:
  - `accepted` if confidence ≥ 0.9 **and** supporting segments ≥ 10 **and** contradictions ≤ 10%;
  - `uncertain` with an explicit `uncertainty_reason` otherwise;
  - never forced.

### 15.4 Calibration set (needs about 1–2 hours of human time)

- Use the 72 existing verification clips plus about 100 more sampled segments, including hard cases (two-shots, reaction shots, off-screen speech).
- A minimal labelling UI or a CSV template captures `{segment, who is speaking: track/character/offscreen}`.
- This yields measured ASD precision and recall on *this* content (vertical, Chinese drama), which is a hard requirement before trusting automatic thresholds.

---

## 16. Trailer Quality Evaluation Strategy

`quality_report.json` separates **objective** metrics (computed, reproducible) from **subjective** judgments (VLM/LLM opinions or human ratings), and each item carries `kind`.

| Area | Objective metrics | Subjective (labelled) |
|---|---|---|
| Character identity | Cosine similarity between the generated face per frame and the character's master embedding (mean, p10, worst-shot); between-character separation; clothing colour/histogram continuity in torso regions | VLM: "same person?" pairwise with reasons |
| Face realism | Face-detector confidence and landmark plausibility (eye/mouth geometry outliers); blur/sharpness | VLM artifact critique (hands, teeth, eyes) |
| Temporal | RAFT-warped flicker error; frame-difference spikes; seam discontinuity at chunk boundaries; ffmpeg `freezedetect` / `blackdetect` | VLM on frame strips |
| Motion and camera | Optical-flow magnitude vs. source shot (camera-intent preservation); jitter | |
| Background and lighting | Background SSIM/LPIPS across shots of the same environment; luminance and colour-temperature consistency across shots in a scene | VLM continuity |
| Voice | Hindi ASR back-transcription CER vs. the target text; speaker similarity (ECAPA) within and between characters; UTMOS/DNSMOS; speaking-rate error vs. target; clipping and silence detection | Native-speaker rating (1–5) on pronunciation, naturalness and emotion |
| Lip-sync | SyncNet LSE-C / LSE-D; AV offset; mouth-region identity drift | Human rating |
| Narrative | Every trailer shot references a source segment (100%); character coverage targets met; act distribution; spoiler-cap check | LLM coherence critique of the edit |
| Technical | Resolution, fps, aspect ratio, duration, loudness (LUFS / true peak), container validity | — |

- Each check has a threshold in `configs/quality.yaml`, with a per-profile override. Status is `pass`, `warn`, `fail` or `not_measurable`.
- A failed shot is **quarantined** and listed with actionable recommendations, for example "identity p10 0.41 < 0.55 → regenerate with higher LoRA weight or switch to the animate_replace strategy".
- **The report states explicitly that subjective scores do not guarantee quality.** The human trailer review remains the gate.

---

## 17. Cost Estimation Strategy

This is a personal learning project on your own Vast.ai account, so the cost model exists to **protect your wallet**, not to quote anyone. It answers two questions: "what will the sample cost?" and "what will full production cost, roughly?". Hard spend caps (§9) enforce the answer.

1. **Before any benchmark:** produce only a *range*, labelled `unbenchmarked`. Example for project_001 (6670 s; 2702 shots; about 30–60% of shots contain speaking faces):
   - **Video generation:** at 720p with Wan A14B-class models, published and community throughput is roughly a few to over ten H100-minutes per 5 s clip. Distillation / few-step LoRAs (e.g. LightX2V ⚠ verify) cut that substantially. That gives about **1–5 GPU-minutes per output second**, so ~110–560 GPU-hours, plus lip-sync (~10–20% extra), plus upscaling, plus retries (assume 1.3–2.0× attempts).
   - **Other stages:** analysis, LLM and TTS are minor (< 10%).
   - I am stating this range only to set expectations. **It is not a quote.**
2. **After the sample:** each work item records GPU seconds and `$`. The estimator fits per-(stage, model, profile, shot-type) throughput, measured as seconds of GPU per second of output, for each shot-type bucket. It then multiplies by the full-project shot inventory per bucket and applies the observed retry rate and QA failure rate. Output: `p50` and `p90` GPU-hours and cost, at the Vast.ai price observed for the selected GPU class (queried live; the price history is stored).
3. **Continuous:** during full production the ETA and cost are re-estimated every N items. If the projected cost exceeds `budget.max_cost` × 1.1, production pauses and an email is sent.
4. **History:** `execution_history` across projects improves the priors for new projects.

---

## 18. Known Technical Limitations

These are risks I expect to be real, not hypothetical:

1. **Photorealism and identity over 111 minutes are not guaranteed.**
   - Current open models produce excellent shots *some of the time*.
   - Expect identity drift, hand and teeth artifacts, and weak multi-person interactions. Many shots will need regeneration.
   - A human review queue for flagged shots is part of the design, not a fallback.
2. **Multi-character shots** (two-shots, group scenes) need multiple masked passes and fail more often. Fallback strategies:
   - `animate_replace` on the dominant subject plus VACE for the others;
   - re-framing;
   - as a last resort, keeping some background extras untouched (only with rights).
3. **Hindi lip-sync quality is unknown** until it is benchmarked, because no evaluated model treats Hindi as a primary training language.
4. **Emotional Hindi TTS** covering crying, whispering, shouting and singing is limited in open models. Singing especially will likely fail. The plan is to flag it and use a music or source treatment.
5. **Speaker-character mapping cannot be fully automated reliably.** It needs a calibration set (§15.4), and human confirmation of main-character mappings is strongly recommended.
6. **Face-cluster identity** is also not fully reliable: clusters split by makeup or lighting, and different characters merge. A one-time human merge and confirm pass on the main characters (about 30 minutes using contact sheets) is recommended.
7. **Localization naturalness** cannot be validated automatically. It needs a native Hindi reviewer on a sample of lines, and ideally on all main-character lines.
8. **No objective photorealism metric exists.** VLM judges are subjective and can be fooled.
9. **Licensing:** this is not an issue for personal learning (D1). Moving to commercial use later would remove InsightFace-based identity, F5/XTTS, FLUX-dev and MusicGen, and quality would likely drop in places.
10. **9:16 and 30 fps output** are off-distribution for most video models. Interpolation and upscaling add their own artifacts.
11. **Vast.ai reliability varies by host.** Hosts can disappear, and download bandwidth to fetch about 100 GB of weights is significant. Mitigations are a weights mirror, reliability filters and checkpointing.
12. **Local machine:** 22 GB free disk. The Mac (M3 Max) is fine for development, tests and light analysis, but **cannot run the generation stages** (CUDA-only and/or 80 GB VRAM). Its machine-wide package settings are overridden per project.
13. **Watermarks, burned-in subtitles and logos** in the source ("优优独播剧场", YoYo) will propagate through `animate_replace` unless they are detected and inpainted. This is a new S1 task.

---

## 19. Implementation Milestones

| # | Milestone | Scope highlights |
|---|---|---|
| M1 | Inspection, architecture, skeleton | This document (approved). Local skeleton in the current directory plus a `.gitignore` (no git init). Pydantic config and schemas. `run.py` (validate-only, status, init). ArtifactStore (local + S3). StageGraph + cache. Model registry with **verified** licences. Legacy import of project_001 (hardlinks + manifest). Benchmark harness skeleton. |
| M2 | Video and audio analysis | S0–S4 as platform stages: adapters for current models plus faster-whisper; multi-frame sampling; tracking; ASD; separation; emotion and events; VLM tags; calibration set; speaker-character mapping v2; character DB v2; watermark detection |
| M3 | Story and localization | S5–S7, LLM adapters (Qwen3 + optional API), glossary, adaptation log, review export (CSV/Sheet) for native-speaker review |
| M4 | Character and voice production | S8 (Qwen-Image / Edit sheets, LoRA training), S10 (Parler design → Chatterbox), voice previews, similarity checks, per-line regeneration |
| M5 | Sample and trailer, plus approval | Selection, solver, edit planner, S9/S11/S12/S13/S14 on the subset, trailer QA, cost model v1, **email approval workflow** (companion spec), control plane deployment |
| M6 | Full video generation | Shot router; all strategies; chunking and continuation; temporal consistency; shot regeneration |
| M7 | Lip-sync and audio production | Production-grade S12/S13, M&E handling, loudness, SFX/music |
| M8 | Assembly and QA | Full timeline, encode, subtitles, complete QA, failure quarantine, final reports |
| M9 | Vast.ai production deployment | Images, launcher, distributed workers, interruptible support, cost tracking, weights mirror |
| M10 | Multi-project generalization | A second unrelated project (different language, 16:9) without core code changes |

M5 depends on M2–M4, which run on the Mac and on a single Vast.ai instance. M9's basics (a single Vast instance and the S3 store) are pulled forward into M4, because generation cannot run locally.

---

## 20. Acceptance Criteria per Milestone

| # | Acceptance criteria (all verified by tests or recorded artifacts) |
|---|---|
| M1 | `run.py --validate-only` passes for project_001 and fails clearly on 10 malformed configs. `--status` lists imported legacy artifacts as `valid` cache hits, so **re-running analysis triggers zero transcription, diarization or shot-detection work**. The sha256 manifest of the legacy workspace is unchanged after migration. Every registered model has a verified code and weight licence with source URLs. Unit tests cover time handling, schemas, config, hashing and the cache. |
| M2 | On a labelled 5-minute slice: shot F1 ≥ 0.9; CER reported; DER reported. ASD precision ≥ 0.9 at the chosen threshold on the calibration set. Every speaker has a mapping or an explicit `uncertain` reason with evidence. Main characters are human-confirmed. All outputs conform to the schemas. Running on a second short test video works with config only. |
| M3 | Every segment has a localized line or a traced reason. 100% citation validity in narrative outputs. Glossary consistency 100%. Native reviewer rates ≥ 80% of 50 sampled lines "natural" (4–5/5). Changing one line re-runs only that line's downstream items, verified by planner output. |
| M4 | Each important character has an approved reference sheet whose identity self-similarity meets the threshold. Each speaking character has a unique voice: between-character similarity < τ, within-character similarity > τ. Hindi CER on previews ≤ threshold. No source-voice references are used (assertion test). |
| M5 | `--mode analyze-and-sample` produces every Part 5.10 artifact. Every important speaking character has a speaking shot, or a recorded infeasibility reason. The trailer passes the technical QA checks. Cost model p50/p90 is produced from measured throughput. The email approval acceptance criteria (companion spec §16) all pass. PROCEED causes full production to reuse the sample artifacts (cache hits verified). |
| M6 | All shots have a plan. ≥ 95% of shots pass QA after ≤ N attempts. Failures are quarantined and listed. `--regenerate-shot` touches only dependent artifacts. |
| M7 | LSE-C/D thresholds are met on ≥ 90% of speaking shots. The mix meets loudness and true-peak targets. No missing dialogue (all approved lines are present in the mix, verified by ASR alignment). |
| M8 | Final frame count, fps, resolution and aspect are as configured. No unintended black or freeze frames. A/V sync < 1 frame. Subtitles are aligned. Final report lists completed, failed, skipped and retried items with zero "silent skips". |
| M9 | One-command run on Vast.ai from a clean instance. Killing the instance mid-run and resuming loses ≤ 1 work item. Cost report matches the Vast billing within 5%. Weights are verified by sha256. |
| M10 | A second project (e.g. English 16:9 → Telugu) runs end to end with config changes only, and `git diff src/` is empty. Projects are isolated in storage and DB. |

---

## Appendix A. Proposed Project Config Additions (beyond the Part 3.2 example)

```yaml
licensing:
  usage: personal_learning     # personal_learning | non_commercial | commercial
rights:
  purpose: self_improvement_learning
  distribution: private                  # outputs are not published
  reuse_source_footage: false            # generate new visuals (the learning goal)
  reuse_source_music_and_effects: true   # Demucs M&E stem, private use only
analysis:
  transcript_filters:
    drop_patterns: ["优优独播剧场", "YoYo Television Series Exclusive"]   # moved from code
  frame_sampling_fps: 3
  asd: { model: light_asd, accept_threshold: auto }   # auto = from calibration
profiles:
  sample: balanced
  full: high
budget:
  max_total_spend_usd: null        # open (no cap); set a number to enforce
  max_sample_spend_usd: null       # open
  pause_on_overrun_ratio: 1.1
compute:
  provider: vastai                 # local | vastai
  launcher: auto                   # auto (uses VAST_API_KEY) | manual
  mode: single                     # single | distributed
  max_instances: 1
  max_price_per_hour_usd: null     # open
  idle_destroy_minutes: 10
  interruptible: true
packages:                          # public sources only
  pip_index_url: https://pypi.org/simple
  container_registries: [docker.io, ghcr.io, nvcr.io]
  forbid_hosts_matching: []        # e.g. ["*.corp.example"]
```
