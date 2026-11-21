"""Validated project configuration schema (schema_version 1).

Pydantic models here are the single source of truth for configuration fields,
defaults and documentation (``run.py --config-schema`` prints the JSON schema;
``run.py --print-config`` prints the effective, merged configuration).

Every setting that may differ between projects lives here. Nothing project-specific
(language, character ids, filenames, resolution, scene ids) is hardcoded elsewhere.
"""

from __future__ import annotations

import re
from enum import Enum
from math import gcd
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from ..core.network_policy import DEFAULT_FORBIDDEN_HOST_PATTERNS, host_of, is_forbidden_host

SLUG_PATTERN = r"^[a-z0-9][a-z0-9_\-]{1,63}$"
LANG_PATTERN = r"^[a-z]{2,3}(-[A-Za-z0-9]{2,8})*$"
_EMAIL_RE = re.compile(r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)+$")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)


def parse_resolution(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d{2,5})x(\d{2,5})", value.strip())
    if not match:
        raise ValueError(f"resolution must look like WIDTHxHEIGHT (e.g. 1080x1920), got '{value}'")
    return int(match.group(1)), int(match.group(2))


def aspect_of(width: int, height: int) -> str:
    divisor = gcd(width, height)
    return f"{width // divisor}:{height // divisor}"


# ---------------------------------------------------------------------------
# Project identity and creative intent
# ---------------------------------------------------------------------------


class ProjectType(str, Enum):
    full_remake = "full_remake"
    localization = "localization"
    character_replacement = "character_replacement"
    background_replacement = "background_replacement"
    dubbing = "dubbing"
    dialogue_translation = "dialogue_translation"
    video_to_video = "video_to_video"
    text_to_video = "text_to_video"
    image_to_video = "image_to_video"
    partial_regeneration = "partial_regeneration"
    character_redesign = "character_redesign"
    environment_redesign = "environment_redesign"
    audio_replacement = "audio_replacement"
    lip_sync = "lip_sync"
    custom = "custom"


class ProjectSection(StrictModel):
    id: str = Field(pattern=SLUG_PATTERN, description="Stable project slug; internal only.")
    name: str = Field(min_length=1)
    project_type: ProjectType
    source_video: str = Field(description="Path to the source video, relative to the project directory.")
    description: str = ""


class Languages(StrictModel):
    source: str = Field(pattern=LANG_PATTERN)
    target: str = Field(pattern=LANG_PATTERN)
    target_region: str | None = None


class DialectSpec(StrictModel):
    name: str = Field(default="neutral_hindi", description="dialect id inside the regional profile, e.g. dakhni_urdu")
    intensity: float = Field(default=0.5, ge=0, le=1, description="project default; each character varies around it")
    english_mixing: float = Field(default=0.3, ge=0, le=1)
    urdu_influence: float = Field(default=0.3, ge=0, le=1)
    forced_slang: bool = Field(default=False, description="false = regional expressions only where they fit")


class CharacterNamesSpec(StrictModel):
    mode: Literal["culturally_adaptive", "keep_design_names"] = "culturally_adaptive"
    region: str | None = Field(default=None, description="naming conventions; defaults to localization.regional_profile")
    preserve_relationships: bool = True
    min_name_distance: float = Field(default=0.34, ge=0, le=1, description="reject names this similar (0 = identical)")


class StorytellingSpec(StrictModel):
    mode: Literal["creative_adaptation", "faithful_localization"] = "creative_adaptation"
    enhance_emotions: bool = True
    enhance_comedy: bool = True
    enhance_romance: bool = True
    enhance_suspense: bool = True
    enhance_tragedy: bool = True
    preserve_story_logic: bool = True
    preserve_character_motivations: bool = True
    avoid_forced_humour: bool = True
    scene_gap_seconds: float = Field(default=5.0, gt=0, description="a silence this long starts a new scene")
    scene_max_lines: int = Field(default=24, ge=4, le=60)


class AdaptationAudioSpec(StrictModel):
    natural_delivery: bool = True
    emotion_aware_tts: bool = True
    conversational_pauses: bool = True
    character_voice_consistency: bool = True
    regional_pronunciation: bool = True
    max_pause_seconds: float = Field(default=0.8, ge=0, le=3)


class ReferenceSource(StrictModel):
    title: str
    kind: Literal["public_domain", "licensed_book", "licensed_screenplay", "film", "tv_drama", "dubbed_film",
                  "regional_cinema", "original_example", "education"]
    licence: str = Field(description="public domain / licence name / 'viewed for technique study only'")
    path: str | None = Field(default=None, description="optional local text you have rights to; used ONLY for the "
                             "originality check (never sent to a model)")
    notes: str = ""


class CreativeReferencesSpec(StrictModel):
    enabled: bool = True
    learn_storytelling_techniques: bool = True
    allow_dialogue_copying: bool = False
    allow_scene_copying: bool = False
    require_original_expression: bool = True
    max_ngram_overlap: int = Field(default=6, ge=3, le=20, description="flag a line sharing this many consecutive words")
    sources: list[ReferenceSource] = Field(default_factory=list)

    @model_validator(mode="after")
    def _no_copying(self) -> "CreativeReferencesSpec":
        if self.allow_dialogue_copying or self.allow_scene_copying or not self.require_original_expression:
            raise ValueError("creative_references: copying dialogue or scenes is not supported; references are for "
                             "general storytelling techniques only (set allow_*_copying: false, "
                             "require_original_expression: true)")
        return self


class AdaptationModels(StrictModel):
    """Model per role. A list = preference order; the first that fits the GPU is used. null = production.llm.*"""
    analyzer: list[str] | str | None = None
    writer: list[str] | str | None = None
    naturalness_reviewer: list[str] | str | None = None
    screenplay_reviewer: list[str] | str | None = None


class QualitySpec(StrictModel):
    min_score: int = Field(default=70, ge=0, le=100, description="scenes/lines below this are rewritten")
    max_rewrite_rounds: int = Field(default=1, ge=0, le=4)
    syllables_per_second: float = Field(default=5.0, gt=1, description="natural Hindi speaking rate for timing")
    max_speedup: float = Field(default=1.12, ge=1, le=1.5, description="beyond this a line is 'too long' and rewritten")
    min_fill: float = Field(default=0.35, ge=0, le=1, description="below this share of the slot a line is 'too short'")
    final_naturalness_check: bool = True


class Localization(StrictModel):
    preserve_story: bool = True
    cultural_adaptation: bool = True
    natural_dialogue: bool = True
    generate_subtitles: bool = True
    # YAML key stays "register"; the attribute is renamed to avoid shadowing a Python built-in.
    model_config = ConfigDict(extra="forbid", validate_default=True, populate_by_name=True)
    speech_register: Literal["colloquial", "neutral", "formal"] = Field(default="colloquial", alias="register")
    batch_lines: int = Field(
        default=12, ge=3, le=60, description="lines per translation/polish prompt (keeps the JSON answer within max tokens)")
    max_tokens: int = Field(default=8192, ge=512, description="answer budget for translation/polish prompts")
    # ---- Indian story adaptation (docs/specs/indian-story-adaptation.md) ----
    country: str = "India"
    language: str = "Hindi"
    writing_style: Literal["natural_hindi", "natural_hinglish", "formal_hindi", "dakhni"] = "natural_hindi"
    regional_profile: str = Field(default="neutral_hindi", description="configs/regional_profiles/<name>.yaml")
    dialect: DialectSpec = Field(default_factory=DialectSpec)
    character_names: CharacterNamesSpec = Field(default_factory=CharacterNamesSpec)
    storytelling: StorytellingSpec = Field(default_factory=StorytellingSpec)
    audio: AdaptationAudioSpec = Field(default_factory=AdaptationAudioSpec)
    creative_references: CreativeReferencesSpec = Field(default_factory=CreativeReferencesSpec)
    models: AdaptationModels = Field(default_factory=AdaptationModels)
    quality: QualitySpec = Field(default_factory=QualitySpec)


class Creative(StrictModel):
    visual_style: str = "photorealistic_cinematic"
    character_style: str = "realistic_fictional"
    environment_style: str = "cinematic"
    preserve_scene_order: bool = True
    preserve_camera_intent: bool = True
    character_consistency: Literal["low", "medium", "high"] = "high"


class VideoSpec(StrictModel):
    target_resolution: str = "1080x1920"
    target_fps: float = Field(default=30, gt=0, le=120)
    target_aspect_ratio: str = Field(default="9:16", pattern=r"^\d+:\d+$")
    codec: Literal["h264", "h265", "av1"] = "h264"

    @field_validator("target_resolution")
    @classmethod
    def _valid_resolution(cls, value: str) -> str:
        parse_resolution(value)
        return value

    @model_validator(mode="after")
    def _aspect_matches(self) -> VideoSpec:
        width, height = parse_resolution(self.target_resolution)
        if aspect_of(width, height) != self.target_aspect_ratio:
            raise ValueError(
                f"target_resolution {self.target_resolution} has aspect {aspect_of(width, height)}, "
                f"but target_aspect_ratio is {self.target_aspect_ratio}"
            )
        return self


class AudioSpec(StrictModel):
    target_language: str = Field(pattern=LANG_PATTERN)
    voice_generation: bool = True
    preserve_emotion: bool = True
    speaker_voice_consistency: bool = True
    loudness_lufs: float = Field(default=-14.0, ge=-31, le=-5)
    true_peak_dbtp: float = Field(default=-1.0, ge=-9, le=0)


class GenerationSpec(StrictModel):
    replace_characters: bool = True
    regenerate_backgrounds: bool = True
    generate_new_shots: bool = True
    lip_sync: bool = True


class QualitySpec(StrictModel):
    automatic_quality_checks: bool = True
    review_low_confidence: bool = True
    human_approval_required: bool = True


class SampleSpec(StrictModel):
    enabled: bool = True
    duration_seconds: int = Field(default=90, ge=10, le=600)
    include_all_important_characters: bool = True
    include_all_speaking_characters: bool = True
    character_coverage: Literal["low", "medium", "high"] = "high"
    narrative_coverage: Literal["partial", "full"] = "full"
    emotional_diversity: bool = True
    generate_voice_previews: bool = True
    generate_subtitles: bool = True
    generate_quality_report: bool = True
    resolution: str = "720x1280"
    fps: float = Field(default=24, gt=0, le=120)
    visual_quality: Literal["preview", "balanced", "high", "max"] = "balanced"
    human_approval_required: bool = True
    spoiler_policy: Literal["avoid_major_twists", "allow"] = "avoid_major_twists"
    showcase_moods: list[str] = Field(
        default_factory=lambda: ["tragedy|emotional|family_drama", "comedy", "confrontation", "romance|suspense|thriller"],
        description="the trailer includes one line from a scene of each group, when the source has one")
    showcase_min_accents: int = Field(default=2, ge=0, le=5, description="distinct character speaking profiles to show")
    compare_profiles: list[str] = Field(
        default_factory=lambda: ["high", "balanced"],
        description="Produce one trailer per production profile so quality vs cost can be compared side by side")

    @field_validator("resolution")
    @classmethod
    def _valid_resolution(cls, value: str) -> str:
        parse_resolution(value)
        return value


# ---------------------------------------------------------------------------
# Licensing and rights
# ---------------------------------------------------------------------------


class Licensing(StrictModel):
    usage: Literal["personal_learning", "non_commercial", "commercial"] = "personal_learning"


class Rights(StrictModel):
    purpose: str = "self_improvement_learning"
    distribution: Literal["private", "public"] = "private"
    reuse_source_footage: bool = False
    reuse_source_music_and_effects: bool = False
    reuse_source_voices: bool = Field(
        default=False, description="Always false: source actors' voices are never cloned."
    )

    @field_validator("reuse_source_voices")
    @classmethod
    def _never_clone(cls, value: bool) -> bool:
        if value:
            raise ValueError("reuse_source_voices must be false: the platform never clones source voices")
        return value


# ---------------------------------------------------------------------------
# Analysis stage parameters (hashed into artifact ids; changing them re-runs the stage)
# ---------------------------------------------------------------------------


class AudioExtractParams(StrictModel):
    stream_index: int = Field(default=0, ge=0)
    sample_rate: int = Field(default=16000, ge=8000, le=192000)
    channels: int = Field(default=1, ge=1, le=8)
    codec: Literal["pcm_s16le", "pcm_f32le"] = "pcm_s16le"


class AsrParams(StrictModel):
    model: str = "faster-whisper-large-v3"
    backend: Literal["mlx_whisper", "faster_whisper", "whisperx", "funasr"] = "faster_whisper"
    fallback_model: str = Field(default="faster-whisper-large-v3", description="used if the primary ASR fails")
    language: str | None = None
    condition_on_previous_text: bool = False


class TranscriptFilterParams(StrictModel):
    drop_patterns: list[str] = Field(
        default_factory=list, description="Case-insensitive substrings; matching segments (e.g. watermarks) are dropped."
    )


class DiarizationParams(StrictModel):
    engine: Literal["diarizen", "pyannote"] = "pyannote"
    model: str = "pyannote-speaker-diarization-community-1"
    fallback_model: str = Field(default="pyannote-speaker-diarization-community-1",
                                description="pyannote model used if the primary engine fails")
    min_speakers: int | None = Field(default=None, ge=1)
    max_speakers: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _bounds(self) -> DiarizationParams:
        if self.min_speakers and self.max_speakers and self.min_speakers > self.max_speakers:
            raise ValueError("min_speakers must be <= max_speakers")
        return self


class ShotParams(StrictModel):
    detector: Literal["pyscenedetect_content", "pyscenedetect_adaptive", "transnetv2"] = "pyscenedetect_content"
    threshold: float = Field(default=27.0, gt=0)
    min_scene_len_frames: int = Field(default=15, ge=1)


class FrameParams(StrictModel):
    strategy: Literal["midpoint", "uniform"] = "uniform"
    fps: float | None = Field(default=2.0, gt=0, le=30, description="Sampling rate for 'uniform'.")
    max_per_shot: int = Field(default=12, ge=1, le=120)
    width: int = Field(default=720, ge=64, le=4096)
    jpeg_quality: int = Field(default=85, ge=1, le=100)


class FaceDetectionParams(StrictModel):
    model: str = "opencv-yunet-2023mar"
    confidence_threshold: float = Field(default=0.75, ge=0, le=1)
    nms_threshold: float = Field(default=0.3, ge=0, le=1)
    top_k: int = Field(default=5000, ge=1)
    padding: float = Field(default=0.35, ge=0, le=2)
    crop_size: int = Field(default=256, ge=32, le=2048)
    jpeg_quality: int = Field(default=90, ge=1, le=100)


class FaceEmbeddingParams(StrictModel):
    model: str = "insightface-buffalo-l"
    det_size: int = Field(default=320, ge=64, le=1280)
    execution_provider: str = "CUDAExecutionProvider"
    face_selection: Literal["largest"] = "largest"


class ClusteringParams(StrictModel):
    algorithm: Literal["dbscan"] = "dbscan"
    eps: float = Field(default=0.35, gt=0, lt=2)
    min_samples: int = Field(default=2, ge=1)
    metric: Literal["cosine"] = "cosine"


class CharacterDbParams(StrictModel):
    representatives_per_cluster: int = Field(default=5, ge=1, le=50)


class AlignmentParams(StrictModel):
    strong_ratio: float = Field(default=0.70, gt=0, le=1)
    moderate_ratio: float = Field(default=0.30, gt=0, le=1)

    @model_validator(mode="after")
    def _ordered(self) -> AlignmentParams:
        if self.moderate_ratio >= self.strong_ratio:
            raise ValueError("moderate_ratio must be < strong_ratio")
        return self


class EvidenceLevelRule(StrictModel):
    min_single_character_seconds: float = Field(ge=0)
    min_single_character_percentage: float = Field(default=0, ge=0, le=100)
    min_unique_scenes: int = Field(ge=0)


class SpeakerPriorParams(StrictModel):
    confidence_weights: dict[str, float] = Field(
        default_factory=lambda: {"strong": 1.0, "moderate": 0.7, "weak": 0.35, "unknown": 0.5}
    )
    stronger_candidate: EvidenceLevelRule = Field(
        default_factory=lambda: EvidenceLevelRule(
            min_single_character_seconds=30, min_single_character_percentage=50, min_unique_scenes=10
        )
    )
    moderate_candidate: EvidenceLevelRule = Field(
        default_factory=lambda: EvidenceLevelRule(min_single_character_seconds=10, min_unique_scenes=5)
    )


class SeparationParams(StrictModel):
    engine: Literal["bs_roformer", "demucs"] = Field(
        default="bs_roformer", description="bs_roformer (cleaner vocals removal); falls back to demucs on failure")
    roformer_model_file: str = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
    model: str = "demucs-htdemucs"
    demucs_model: str = "htdemucs"
    segment_seconds: float = Field(default=7.0, gt=1, le=7.8)


class EmotionParams(StrictModel):
    enabled: bool = True
    engine: Literal["emotion2vec", "sensevoice"] = Field(
        default="emotion2vec", description="emotion2vec+ large (speech emotion); SenseVoice as fallback")
    emotion2vec_repo: str = "emotion2vec/emotion2vec_plus_large"
    model: str = "sensevoice-small"
    hf_repo: str = "FunAudioLLM/SenseVoiceSmall"
    required: bool = Field(default=False, description="If false, a failure marks emotions 'unknown' with a warning.")


class ActiveSpeakerParams(StrictModel):
    method: Literal["light_asd", "lower_face_motion"] = Field(
        default="light_asd", description="trained Light-ASD model; falls back to the motion heuristic on failure")
    light_asd_weights: str = "weight/pretrain_AVA_CVPR.model"
    asd_fps: int = Field(default=25, ge=10, le=30)
    asd_max_seconds: float = Field(default=2.5, gt=0.5, le=10)
    speaking_probability: float = Field(default=0.5, gt=0, lt=1)
    frames_per_segment: int = Field(default=10, ge=3, le=60)
    max_segment_seconds: float = Field(default=4.0, gt=0.5)
    min_face_px: int = Field(default=40, ge=16)
    identity_threshold: float = Field(default=0.35, gt=0, lt=1, description="cosine to character centroid")
    min_visible_ratio: float = Field(default=0.5, ge=0, le=1)
    margin_ratio: float = Field(default=1.3, ge=1.0)


class SpeakerMappingParams(StrictModel):
    accept_confidence: float = Field(default=0.6, gt=0, le=1)
    min_supporting_segments: int = Field(default=5, ge=1)


class ImportanceParams(StrictModel):
    weights: dict[str, float] = Field(
        default_factory=lambda: {"speech": 0.35, "screen": 0.30, "scenes": 0.20, "recurrence": 0.15}
    )
    main_threshold: float = 0.60
    supporting_threshold: float = 0.35
    recurring_min_thirds: int = 2
    min_face_count: int = 8


class AnalysisSpec(StrictModel):
    audio: AudioExtractParams = Field(default_factory=AudioExtractParams)
    asr: AsrParams = Field(default_factory=AsrParams)
    transcript_filters: TranscriptFilterParams = Field(default_factory=TranscriptFilterParams)
    diarization: DiarizationParams = Field(default_factory=DiarizationParams)
    shots: ShotParams = Field(default_factory=ShotParams)
    frames: FrameParams = Field(default_factory=FrameParams)
    face_detection: FaceDetectionParams = Field(default_factory=FaceDetectionParams)
    face_embedding: FaceEmbeddingParams = Field(default_factory=FaceEmbeddingParams)
    clustering: ClusteringParams = Field(default_factory=ClusteringParams)
    character_database: CharacterDbParams = Field(default_factory=CharacterDbParams)
    alignment: AlignmentParams = Field(default_factory=AlignmentParams)
    speaker_mapping_prior: SpeakerPriorParams = Field(default_factory=SpeakerPriorParams)
    separation: SeparationParams = Field(default_factory=SeparationParams)
    emotion: EmotionParams = Field(default_factory=EmotionParams)
    active_speaker: ActiveSpeakerParams = Field(default_factory=ActiveSpeakerParams)
    speaker_mapping: SpeakerMappingParams = Field(default_factory=SpeakerMappingParams)
    importance: ImportanceParams = Field(default_factory=ImportanceParams)


# ---------------------------------------------------------------------------
# Production (story, localization, assets, generation) and runtime layout
# ---------------------------------------------------------------------------


class LlmSpec(StrictModel):
    model: str = "qwen3-14b"
    backend: Literal["vllm", "transformers"] = "vllm"
    max_model_len: int = Field(default=16384, ge=2048)
    max_tokens: int = Field(default=4096, ge=256)
    temperature: float = Field(default=0.3, ge=0, le=2)
    chunk_segments: int = Field(default=40, ge=5, le=200)
    tensor_parallel_size: int = Field(default=1, ge=1)
    gpu_memory_utilization: float = Field(default=0.88, gt=0.1, le=0.98)
    max_characters_to_design: int = Field(default=12, ge=1)
    polish_model: str | None = Field(
        default="sarvam-m", description="Indic-tuned LLM that rewrites the draft into natural target-language dialogue")


_DEFAULT_VARIANTS = {
    "full_body": "Full-body photo of the same person standing, same face, same clothing, neutral background.",
    "smile": "Close-up of the same person smiling warmly, same face, same hair, same clothing.",
    "sad": "Close-up of the same person looking sad with teary eyes, same face, same hair, same clothing.",
    "angry": "Close-up of the same person looking angry, same face, same hair, same clothing.",
    "profile": "Side profile view of the same person, same face, same hair, same clothing.",
}


class CastPhoto(StrictModel):
    path: str = Field(description="relative to the project directory, e.g. input/cast/me_1.jpg (gitignored)")
    face: Literal["auto", "largest", "left", "right", "only"] = Field(
        default="auto", description="auto = the face that matches this person's other photos (group photos); "
        "or left/right/largest")
    sha256: str | None = Field(default=None, description="filled in automatically; new photos re-run the faces")


class CastConsent(StrictModel):
    given_by: str = Field(description="who agreed, e.g. 'Swapnil (self)' or 'spouse'")
    date: str
    scope: str = "personal, private, non-commercial learning project; output not published"


class CastMember(StrictModel):
    """A real, consenting person whose face shapes one character (blend = a new person who resembles them)."""
    name: str
    photos: list[CastPhoto] = Field(min_length=1)
    character: str = Field(description="lead_male | lead_female | a character id such as SRC_CHAR_0080")
    mode: Literal["blend", "as_is"] = "blend"
    similarity: tuple[float, float] = Field(
        default=(0.30, 0.55), description="blend: ArcFace similarity to the person's photos must fall in this range "
        "(clearly related, not identical). as_is uses only the lower bound (default raised to 0.55).")
    consent: CastConsent

    @model_validator(mode="after")
    def _check(self) -> "CastMember":
        lo, hi = self.similarity
        if not 0 <= lo < hi <= 1:
            raise ValueError("cast similarity must be 0 <= low < high <= 1")
        for photo in self.photos:
            if "://" in photo.path:
                raise ValueError("cast photos must be local files (never URLs)")
        return self


class AppearanceSpec(StrictModel):
    """Look applied to generated characters (kept natural: real people, not retouched models)."""
    apply_to: Literal["all", "leads", "none"] = "all"
    build: str = "slim, fit build"
    look: str = Field(default="good-looking and attractive in a natural way, like a film actor on set without heavy makeup",
                      description="kept natural: the realism rules still apply")
    blend_note: str = "slimmer face and a defined jawline; keep the family resemblance"


class CastFolders(StrictModel):
    """Photo folders, one per person, discovered in <repo>/input/ and <project>/input/ (both gitignored)."""
    enabled: bool = True
    folders: dict[str, str] = Field(default_factory=lambda: {"lead_male": "Male Lead", "lead_female": "Female Lead"},
                                    description="character → folder name (matched case-insensitively, _ or space)")
    mode: Literal["blend", "as_is"] = "blend"
    consent: CastConsent | None = Field(default=None, description="required for folders to be used")


class CharacterImageSpec(StrictModel):
    model: str = "qwen-image"
    edit_model: str = "qwen-image-edit-2509"
    width: int = Field(default=768, ge=256)
    height: int = Field(default=1344, ge=256)
    steps: int = Field(default=40, ge=4)
    edit_steps: int = Field(default=30, ge=4)
    true_cfg_scale: float = Field(default=4.0, ge=0)
    seed: int = 20261121
    candidates: int = Field(default=3, ge=1, le=8, description="master portraits per character; the best one is kept")
    pick: dict[str, int] = Field(default_factory=dict,
                                 description="manual choice from the contact sheet: {GEN_CHAR_0080: 2} (candidate index)")
    style_suffix: str = (
        "candid photorealistic photo of a real ordinary person, natural skin texture with visible pores and small "
        "imperfections, slight facial asymmetry, natural hair with flyaways, realistic eyes with catchlights, "
        "everyday natural light, shot on a 50mm lens, true-to-life colours, film still, not retouched"
    )
    negative: str = (
        "cartoon, anime, illustration, painting, 3d render, cgi, plastic skin, doll, airbrushed, beauty filter, "
        "flawless skin, perfect symmetry, fashion model, influencer, stock photo, glamour lighting, heavy makeup, "
        "deformed face, asymmetric eyes, extra fingers, blurry, low quality, watermark, text, celebrity"
    )
    cast: list[CastMember] = Field(default_factory=list, description="real consenting people (see RUNBOOK)")
    use_global_cast: bool = Field(default=True, description="also use cast/cast.yaml (personal, gitignored) in this project")
    cast_folders: CastFolders = Field(default_factory=CastFolders)
    appearance: AppearanceSpec = Field(default_factory=AppearanceSpec)
    variants: dict[str, str] = Field(default_factory=lambda: dict(_DEFAULT_VARIANTS))


class VoiceDesignSpec(StrictModel):
    model: str = "indic-parler-tts"
    reference_text: str = "नमस्ते, मैं आज आपसे कुछ ज़रूरी बात करना चाहता हूँ। कृपया ध्यान से सुनिए।"
    default_description: str = (
        "A {gender} speaker with a clear, natural voice speaks at a moderate pace. "
        "The recording is very clear and close-sounding with no background noise."
    )


class TtsSpec(StrictModel):
    engine: Literal["auto", "chatterbox", "indicf5"] = Field(
        default="auto", description="auto = bake-off on a few lines, lower back-transcription error wins")
    bakeoff_lines: int = Field(default=12, ge=3, le=50)
    indicf5_model: str = "indicf5"
    model: str = "chatterbox-multilingual"
    cfg_weight: float = Field(default=0.5, ge=0, le=1)
    exaggeration: dict[str, float] = Field(
        default_factory=lambda: {
            "neutral": 0.45, "happy": 0.6, "sad": 0.55, "angry": 0.8, "fearful": 0.65,
            "surprised": 0.7, "disgusted": 0.6, "unknown": 0.5,
        }
    )
    max_speedup: float = Field(default=1.15, ge=1.0, le=1.5)
    sample_rate: int = 48000


class VideoGenSpec(StrictModel):
    enabled: bool = True
    model: str = "wan2.2-animate-14b"
    offload: Literal["auto", "on", "off"] = Field(
        default="auto", description="auto = off on GPUs with >= 80 GB (faster), on otherwise")
    persistent_worker: bool = Field(
        default=True, description="Keep Wan loaded across clips (falls back to per-clip CLI on failure)")
    wan_task: str = "animate-14B"
    frame_num: int = Field(default=77, ge=5, description="Wan clip length per pass (frames)")
    no_character_strategy: Literal["keep_source", "i2v"] = "keep_source"
    max_clip_seconds: float = Field(
        default=30.0, gt=1, description="Only shots longer than this are split; Wan keeps continuity inside a clip")
    retries: int = Field(default=1, ge=0, le=5)
    timeout_minutes: int = Field(default=90, ge=5)
    work_fps: int = Field(default=16, ge=8, le=60)
    preprocess_command: list[str] = Field(default_factory=lambda: [
        "{python}", "wan/modules/animate/preprocess/preprocess_data.py",
        "--ckpt_path", "{weights}/Wan2.2-Animate-14B/process_checkpoint",
        "--video_path", "{video}", "--refer_path", "{ref}", "--save_path", "{workdir}/process",
        "--resolution_area", "{area_w}", "{area_h}", "--iterations", "3", "--k", "7", "--w_len", "1", "--h_len", "1",
        "--replace_flag",
    ])
    generate_command: list[str] = Field(default_factory=lambda: [
        "{python}", "generate.py", "--task", "animate-14B", "--ckpt_dir", "{weights}/Wan2.2-Animate-14B",
        "--src_root_path", "{workdir}/process", "--refert_num", "1", "--replace_flag", "--use_relighting_lora",
        "--base_seed", "{seed}", "--save_file", "{out}", "--offload_model", "{offload}", "--convert_model_dtype",
        "--sample_steps", "{steps}",
    ])


class LipSyncSpec(StrictModel):
    enabled: bool = True
    model: str = "latentsync-1.6"
    timeout_minutes: int = Field(default=45, ge=5)
    command: list[str] = Field(default_factory=lambda: [
        "{python}", "-m", "scripts.inference",
        "--unet_config_path", "configs/unet/stage2_512.yaml",
        "--inference_ckpt_path", "checkpoints/latentsync_unet.pt",
        "--inference_steps", "{ls_steps}", "--guidance_scale", "1.5", "--enable_deepcache",
        "--video_path", "{video}", "--audio_path", "{audio}", "--video_out_path", "{out}",
    ])


class MixSpec(StrictModel):
    bed_gain_db: float = -4.0
    dialogue_gain_db: float = 0.0


class SelectionSpec(StrictModel):
    min_segment_seconds: float = Field(default=1.2, gt=0)
    max_segment_seconds: float = Field(default=8.0, gt=0)
    min_gap_seconds: float = Field(default=20.0, ge=0)
    spoiler_tail_fraction: float = Field(default=0.15, ge=0, lt=0.5)
    clip_padding_seconds: float = Field(default=0.3, ge=0)


class ParallelSpec(StrictModel):
    """Throughput settings. null = choose automatically from detected GPU memory / CPU cores."""

    cpu_workers: int | None = Field(default=None, ge=1, description="parallel ffmpeg jobs")
    tts_workers: int | None = Field(default=None, ge=1, description="parallel TTS model processes")
    preprocess_workers: int | None = Field(default=None, ge=1, description="parallel Wan preprocessing processes")
    lipsync_workers: int | None = Field(default=None, ge=1, description="parallel LatentSync processes")
    use_nvenc: bool = Field(default=True, description="GPU H.264 encoder for intermediates when available")
    wan_workers_per_gpu: int = Field(default=1, ge=1, le=4,
                                     description="concurrent Wan processes per GPU (generation is compute-bound)")
    gpus: list[int] | None = Field(default=None, description="GPU indices to use; null = all visible GPUs")


class ProfileSpec(StrictModel):
    """Quality / cost trade-offs for generation. Every speed-up falls back safely if unavailable."""

    description: str = ""
    gen_area: list[int] = Field(default_factory=lambda: [1280, 720], min_length=2, max_length=2,
                                description="Wan generation area (w, h); output keeps the video's aspect")
    wan_steps: int = Field(default=20, ge=1, le=100)
    wan_lora: str | None = Field(default=None, description="registry id of a step-distillation LoRA")
    wan_lora_strength: float = Field(default=1.0, gt=0, le=2)
    fallback_steps: int = Field(default=20, ge=1, description="steps used if the distillation LoRA cannot be applied")
    attention: Literal["default", "sage"] = "default"
    upscale: Literal["lanczos", "realesrgan"] = "lanczos"
    lipsync_steps: int = Field(default=20, ge=1, le=100)
    lipsync_min_face_ratio: float = Field(default=0.0, ge=0, le=1,
                                          description="skip lip-sync when face height / frame height is below this")


def _default_profiles() -> dict[str, ProfileSpec]:
    return {
        "high": ProfileSpec(description="maximum quality: 720p-area generation, 20 Wan steps, lip-sync on every speaking shot"),
        "balanced": ProfileSpec(
            description="~5-10x cheaper: 480p-area + Real-ESRGAN, 4-step distilled Wan, SageAttention, "
                        "lip-sync only where faces are visible",
            gen_area=[832, 480], wan_steps=4, wan_lora="lightx2v-i2v-14b-480p-distill", attention="sage",
            upscale="realesrgan", lipsync_steps=12, lipsync_min_face_ratio=0.12,
        ),
    }


class ProductionSpec(StrictModel):
    profile: str | None = Field(default=None, description="profile for full production; null = the approved one")
    profiles: dict[str, ProfileSpec] = Field(default_factory=_default_profiles)
    upscaler_model: str = "realesrgan-x2plus"
    llm: LlmSpec = Field(default_factory=LlmSpec)
    character_images: CharacterImageSpec = Field(default_factory=CharacterImageSpec)
    voice_design: VoiceDesignSpec = Field(default_factory=VoiceDesignSpec)
    tts: TtsSpec = Field(default_factory=TtsSpec)
    video: VideoGenSpec = Field(default_factory=VideoGenSpec)
    lipsync: LipSyncSpec = Field(default_factory=LipSyncSpec)
    mix: MixSpec = Field(default_factory=MixSpec)
    selection: SelectionSpec = Field(default_factory=SelectionSpec)
    parallel: ParallelSpec = Field(default_factory=ParallelSpec)


class RuntimeSpec(StrictModel):
    """Where per-model environments, repositories and weights live on the GPU machine."""

    root: str = Field(default="/workspace/vp", description="Created by scripts/setup_vast.sh")
    device: str = "cuda"
    venvs: dict[str, str] = Field(default_factory=dict, description="runtime → python path override")
    repos: dict[str, str] = Field(default_factory=lambda: {
        "wan": "repos/Wan2.2", "latentsync": "repos/LatentSync", "lightasd": "repos/Light-ASD"})
    weights_dir: str = "weights"
    worker_timeout_minutes: int = Field(default=720, ge=1)
    heartbeat_seconds: int = Field(default=60, ge=0, description="progress line in run.log while a model step runs (0 = off)")
    log_outputs: bool = Field(default=True, description="after each stage: preview its outputs in run.log + state/logs/stages/")
    log_output_items: int = Field(default=4, ge=1, le=50, description="sample items per output in run.log (report has 3x)")
    max_parallel_stages: int = Field(default=3, ge=1, description="independent pipeline stages run concurrently")
    prune_weights: Literal["auto", "always", "never"] = Field(
        default="auto",
        description="auto: delete finished models' weights only when free disk would fall below prune_disk_reserve_gb "
                    "for the remaining stages; always: delete after last use; never: keep everything")
    prune_disk_reserve_gb: float = Field(default=40.0, ge=0, description="free space to keep for renders and caches")
    prefetch_weights: bool = Field(
        default=True, description="download upcoming stages' models in the background while the GPU works")
    prefetch_disk_reserve_gb: float = Field(default=50.0, ge=0, description="never prefetch into the last N GB")


# ---------------------------------------------------------------------------
# Notifications / approval (see docs/specs/email-approval-workflow.md)
# ---------------------------------------------------------------------------


def _validate_email(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if not _EMAIL_RE.fullmatch(value) or "," in value or " " in value:
        raise ValueError(f"'{value}' is not a single valid email address")
    return value


class SmtpSpec(StrictModel):
    host: str
    port: int = Field(default=587, ge=1, le=65535)
    username: str
    password: SecretStr
    use_tls: bool = True


class SenderSpec(StrictModel):
    email: str
    name: str = "AI Video Production Platform"

    @field_validator("email")
    @classmethod
    def check_email(cls, value: str) -> str:
        return _validate_email(value)


class ApprovalSpec(StrictModel):
    enabled: bool = True
    expiration_hours: int = Field(default=72, ge=1, le=720)
    allow_resend: bool = True
    require_authentication: bool = False
    reminder_hours_before_expiry: int = Field(default=12, ge=0)
    allow_cli_override: bool = Field(
        default=True, description="Allow `run.py --approve-sample` as an audited admin override."
    )


_DEFAULT_EVENTS = {
    "sample_ready": True,
    "approval_received": True,
    "production_queued": True,
    "production_started": True,
    "production_progress": False,
    "production_completed": True,
    "production_failed": True,
    "approval_expired": True,
    "project_stopped": True,
}


class NotificationSpec(StrictModel):
    enabled: bool = False
    approval_email: str | None = None
    provider: Literal["smtp", "console", "file", "ses", "sendgrid"] = "console"
    smtp: SmtpSpec | None = None
    sender: SenderSpec | None = None
    approval: ApprovalSpec = Field(default_factory=ApprovalSpec)
    events: dict[str, bool] = Field(default_factory=lambda: dict(_DEFAULT_EVENTS))
    progress_milestones_percent: list[int] = Field(default_factory=lambda: [25, 50, 75])
    merge_ack_emails: bool = True

    @field_validator("approval_email")
    @classmethod
    def check_approval_email(cls, value: str | None) -> str | None:
        return _validate_email(value)

    @field_validator("events")
    @classmethod
    def _known_events(cls, value: dict[str, bool]) -> dict[str, bool]:
        unknown = set(value) - set(_DEFAULT_EVENTS)
        if unknown:
            raise ValueError(f"unknown notification events: {sorted(unknown)}")
        return {**_DEFAULT_EVENTS, **value}

    @model_validator(mode="after")
    def _complete_when_enabled(self) -> NotificationSpec:
        if not self.enabled:
            return self
        if not self.approval_email:
            raise ValueError("notifications.enabled is true but no approval_email is set (config or --email)")
        if self.provider == "smtp" and self.smtp is None:
            raise ValueError("provider 'smtp' requires an smtp block")
        if self.approval.enabled and not self.events.get("sample_ready", True):
            raise ValueError("events.sample_ready cannot be disabled while approval.enabled is true")
        return self


# ---------------------------------------------------------------------------
# Storage, compute, budget, packages, profiles
# ---------------------------------------------------------------------------


class StorageSpec(StrictModel):
    provider: Literal["local", "minio", "s3"] = "local"
    root: str = Field(default="artifacts", description="Local artifact root, relative to the project directory.")
    endpoint: str | None = None
    bucket: str | None = None
    prefix: str = "projects/"
    access_key_id: str | None = None
    secret_access_key: SecretStr | None = None
    presigned_url_ttl_minutes: int = Field(default=60, ge=1, le=1440)

    @model_validator(mode="after")
    def _remote_needs_endpoint(self) -> StorageSpec:
        if self.provider in ("minio", "s3") and not (self.endpoint and self.bucket):
            raise ValueError(f"storage provider '{self.provider}' requires endpoint and bucket")
        return self


class ComputeSpec(StrictModel):
    provider: Literal["local", "vastai"] = "local"
    launcher: Literal["auto", "manual"] = "manual"
    mode: Literal["single", "distributed"] = "single"
    max_instances: int | None = Field(default=1, ge=1, description="null = no limit")
    max_price_per_hour_usd: float | None = Field(default=None, gt=0, description="null = open")
    idle_destroy_minutes: int = Field(default=10, ge=1)
    interruptible: bool = True
    require_verified_hosts: bool = True


class BudgetSpec(StrictModel):
    max_total_spend_usd: float | None = Field(default=None, gt=0, description="null = open (no cap)")
    max_sample_spend_usd: float | None = Field(default=None, gt=0, description="null = open (no cap)")
    pause_on_overrun_ratio: float = Field(default=1.1, ge=1.0, description="Only applies when a cap is set.")


class PackagesSpec(StrictModel):
    pip_index_url: str = "https://pypi.org/simple"
    container_registries: list[str] = Field(default_factory=lambda: ["docker.io", "ghcr.io", "nvcr.io", "quay.io"])
    model_hubs: list[str] = Field(default_factory=lambda: ["huggingface.co", "github.com"])
    forbid_hosts_matching: list[str] = Field(default_factory=lambda: list(DEFAULT_FORBIDDEN_HOST_PATTERNS))

    @model_validator(mode="after")
    def _no_forbidden_sources(self) -> PackagesSpec:
        sources = [("pip_index_url", self.pip_index_url)]
        sources += [("container_registries", r) for r in self.container_registries]
        sources += [("model_hubs", h) for h in self.model_hubs]
        for label, value in sources:
            if is_forbidden_host(host_of(value), self.forbid_hosts_matching):
                raise ValueError(f"packages.{label} '{value}' matches a forbidden host pattern")
        return self


QualityProfile = Literal["fast_preview", "balanced", "high", "max"]


class ProfilesSpec(StrictModel):
    sample: QualityProfile = "balanced"
    full: QualityProfile = "high"


class ProjectConfig(StrictModel):
    schema_version: Literal[1] = 1
    project: ProjectSection
    languages: Languages
    localization: Localization = Field(default_factory=Localization)
    creative: Creative = Field(default_factory=Creative)
    video: VideoSpec = Field(default_factory=VideoSpec)
    audio: AudioSpec
    generation: GenerationSpec = Field(default_factory=GenerationSpec)
    quality: QualitySpec = Field(default_factory=QualitySpec)
    sample: SampleSpec = Field(default_factory=SampleSpec)
    licensing: Licensing = Field(default_factory=Licensing)
    rights: Rights = Field(default_factory=Rights)
    analysis: AnalysisSpec = Field(default_factory=AnalysisSpec)
    notifications: NotificationSpec = Field(default_factory=NotificationSpec)
    storage: StorageSpec = Field(default_factory=StorageSpec)
    compute: ComputeSpec = Field(default_factory=ComputeSpec)
    budget: BudgetSpec = Field(default_factory=BudgetSpec)
    packages: PackagesSpec = Field(default_factory=PackagesSpec)
    profiles: ProfilesSpec = Field(default_factory=ProfilesSpec)
    production: ProductionSpec = Field(default_factory=ProductionSpec)
    runtime: RuntimeSpec = Field(default_factory=RuntimeSpec)

    @model_validator(mode="after")
    def _cross_checks(self) -> ProjectConfig:
        if self.audio.target_language != self.languages.target:
            raise ValueError(
                f"audio.target_language ({self.audio.target_language}) must equal languages.target ({self.languages.target})"
            )
        if self.languages.source == self.languages.target and self.project.project_type in (
            ProjectType.localization,
            ProjectType.dialogue_translation,
        ):
            raise ValueError("source and target languages are identical for a translation project")
        known = set(self.production.profiles)
        for name in [*self.sample.compare_profiles, *( [self.production.profile] if self.production.profile else [])]:
            if name not in known:
                raise ValueError(f"profile '{name}' is not defined in production.profiles ({sorted(known)})")
        if not self.sample.compare_profiles:
            raise ValueError("sample.compare_profiles needs at least one profile")
        for label, url in (("storage.endpoint", self.storage.endpoint),):
            if url and is_forbidden_host(host_of(url), self.packages.forbid_hosts_matching):
                raise ValueError(f"{label} '{url}' matches a forbidden host pattern")
        return self
