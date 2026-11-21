"""Pure-logic tests for production, selection, alignment, importance and approval."""

import pytest

from video_platform.approval.state import ApprovalStore
from video_platform.core.errors import IllegalStateError
from video_platform.stages.analysis import align, classify_characters
from video_platform.stages.production import (
    Clip, char_error_rate, fit_tempo, full_edl, place_lines, sample_edl, select_trailer,
)
from video_platform.workers.common import extract_json
from video_platform.workers.emotion import parse_tags

SEL = {"min_segment_seconds": 1.0, "max_segment_seconds": 8.0, "min_gap_seconds": 5.0,
       "spoiler_tail_fraction": 0.15, "clip_padding_seconds": 0.2}


def _line(i, start, char, emotion="neutral", status="ok"):
    return {"segment_id": f"SEG_{i:06d}", "start_s": start, "end_s": start + 2.0, "character_id": char,
            "status": status, "emotion": emotion, "target_text": "नमस्ते", "short_text": "", "speaker_id": "SPK_00"}


def test_select_trailer_covers_important_speaking_characters_and_avoids_spoilers():
    shots = {f"SH_{i:05d}": {"shot_id": f"SH_{i:05d}", "start_s": i * 10.0, "end_s": i * 10.0 + 10} for i in range(100)}
    lines = [_line(i, i * 10.0 + 1, "SRC_CHAR_0001" if i % 2 else "SRC_CHAR_0002") for i in range(100)]
    lines.append(_line(500, 955.0, "SRC_CHAR_0003"))  # only line of char 3 is in the spoiler tail
    seg_shot = {ln["segment_id"]: f"SH_{int(ln['start_s'] // 10):05d}" for ln in lines}
    asd = {ln["segment_id"]: {"speaking_character": ln["character_id"]} for ln in lines}
    profiles = [
        {"character_id": "SRC_CHAR_0001", "tier": "main", "score": 0.9, "face_count": 100, "speech_s": 100, "shot_count": 50},
        {"character_id": "SRC_CHAR_0002", "tier": "supporting", "score": 0.5, "face_count": 50, "speech_s": 50, "shot_count": 30},
        {"character_id": "SRC_CHAR_0003", "tier": "supporting", "score": 0.4, "face_count": 20, "speech_s": 2, "shot_count": 3},
    ]
    chosen, characters = select_trailer(lines=lines, profiles=profiles, asd=asd, chunks=[], shots=shots,
                                        segment_shot=seg_shot, video_duration=1000.0, target_seconds=30, sel=SEL)
    assert {c["character_id"] for c in chosen} >= {"SRC_CHAR_0001", "SRC_CHAR_0002"}
    assert all(c["start_s"] < 850 for c in chosen)  # spoiler tail excluded
    assert [c["start_s"] for c in chosen] == sorted(c["start_s"] for c in chosen)
    assert 25 <= sum(c["clip_end"] - c["clip_start"] for c in chosen) <= 40
    starts = [c["start_s"] for c in chosen]
    assert all(b - a >= 5.0 for a, b in zip(starts, starts[1:]))
    third = {c["third"] for c in chosen}
    assert len(third) >= 2  # spread across the story
    char3 = next(c for c in characters if c["character_id"] == "SRC_CHAR_0003")
    assert not char3["selected"] and "spoiler" in char3["reason"]


def test_edls_and_line_placement():
    shots = [{"shot_id": "SH_00001", "start_s": 0.0, "end_s": 20.0}, {"shot_id": "SH_00002", "start_s": 20.0, "end_s": 23.0}]
    clips = full_edl(shots, max_seconds=8.0)
    assert [c.clip_id for c in clips] == ["SH_00001_00", "SH_00001_01", "SH_00001_02", "SH_00002_00"]
    assert abs(sum(c.seconds for c in clips) - 23.0) < 1e-6 and clips[-1].out_start == 20.0
    trailer = sample_edl([{"shot_id": "SH_00002", "clip_start": 20.5, "clip_end": 22.5},
                          {"shot_id": "SH_00001", "clip_start": 5.0, "clip_end": 9.0}])
    assert [c.out_start for c in trailer] == [0.0, 2.0]
    lines = [_line(1, 21.0, None), _line(2, 6.0, None), _line(3, 7.0, None)]
    placed = {p["segment_id"]: p for p in place_lines(lines, trailer, confine=True)}
    assert placed["SEG_000001"]["out_start"] == 0.5
    assert placed["SEG_000002"]["available_s"] == 1.0          # next line starts 1 s later
    assert placed["SEG_000003"]["available_s"] == 2.0          # confined to clip end (9.0)


def test_fit_tempo():
    assert fit_tempo(1.0, 2.0, 1.15) == (1.0, False)
    tempo, overflow = fit_tempo(2.2, 2.0, 1.15)
    assert abs(tempo - 1.1) < 1e-9 and not overflow
    assert fit_tempo(3.0, 2.0, 1.15) == (1.15, True)


def test_char_error_rate():
    assert char_error_rate("नमस्ते दोस्त", "नमस्ते दोस्त") == 0.0
    assert 0 < char_error_rate("abcd", "abed") <= 0.25
    assert char_error_rate("", "") == 0.0


def test_align_segments():
    transcript = [{"segment_id": "SEG_000000", "start_s": 0.0, "end_s": 2.0, "text": "a"},
                  {"segment_id": "SEG_000001", "start_s": 3.0, "end_s": 3.0, "text": "bad"}]
    turns = [{"start_s": 0.0, "end_s": 1.8, "speaker_id": "SPK_01"}, {"start_s": 1.8, "end_s": 2.0, "speaker_id": "SPK_02"}]
    shots = [{"shot_id": "SH_00001", "start_s": 0.0, "end_s": 1.5}, {"shot_id": "SH_00002", "start_s": 1.5, "end_s": 9.0}]
    out = align(transcript, turns, shots, 0.7, 0.3)
    assert len(out) == 1  # invalid timestamp skipped
    seg = out[0]
    assert seg.speaker_id == "SPK_01" and seg.speaker_match == "strong" and "overlapping_speech" in seg.flags
    assert seg.shot_id == "SH_00001" and seg.overlapping_shot_ids == ["SH_00001", "SH_00002"]


def test_classify_characters_tiers():
    chars = [
        {"character_id": "A", "face_count": 600, "shot_ids": ["SH_00001", "SH_00500", "SH_00900"]},
        {"character_id": "B", "face_count": 100, "shot_ids": ["SH_00010", "SH_00020"]},
        {"character_id": "C", "face_count": 9, "shot_ids": ["SH_00030"]},
        {"character_id": "D", "face_count": 2, "shot_ids": ["SH_00040"]},  # too few faces, silent → excluded
    ]
    params = {"weights": {"speech": 0.35, "screen": 0.30, "scenes": 0.20, "recurrence": 0.15},
              "main_threshold": 0.6, "supporting_threshold": 0.35, "recurring_min_thirds": 2, "min_face_count": 8}
    out = {p["character_id"]: p for p in classify_characters(chars, {"A": 900.0, "B": 100.0}, 1000, params)}
    assert set(out) == {"A", "B", "C"}
    assert out["A"]["tier"] == "main" and out["C"]["tier"] in ("background", "minor")


def test_extract_json_and_emotion_tags():
    assert extract_json('<think>hmm</think>```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! [1, 2]') == [1, 2]
    with pytest.raises(ValueError):
        extract_json("no json here")
    assert parse_tags("<|zh|><|ANGRY|><|Speech|><|woitn|>你好") == {"emotion": "angry", "events": ["speech"]}


def test_approval_state_machine(tmp_path):
    store = ApprovalStore(tmp_path / "a.sqlite")
    pid = "p"
    with pytest.raises(IllegalStateError):
        store.decide(pid, sample_artifact_id="s1", trailer_sha256="h1", decision="PROCEED", actor="t")
    for state in ("ANALYZING", "ANALYSIS_COMPLETED", "SAMPLE_GENERATING", "SAMPLE_COMPLETED", "APPROVAL_PENDING"):
        store.transition(pid, state, actor="t")
    with pytest.raises(IllegalStateError):
        store.transition(pid, "PRODUCTION_RUNNING", actor="t")  # cannot start production while pending
    assert store.decide(pid, sample_artifact_id="s1", trailer_sha256="h1", decision="PROCEED", actor="t") == "APPROVED"
    assert store.decide(pid, sample_artifact_id="s1", trailer_sha256="h1", decision="PROCEED", actor="t") == "APPROVED"
    with pytest.raises(IllegalStateError):
        store.decide(pid, sample_artifact_id="s1", trailer_sha256="h1", decision="STOP", actor="t")
    assert store.is_approved(pid, "s1", "h1") and not store.is_approved(pid, "s1", "other-trailer")
    store.transition(pid, "PRODUCTION_QUEUED", actor="t")
    store.transition(pid, "PRODUCTION_RUNNING", actor="t")
    store.transition(pid, "PRODUCTION_COMPLETED", actor="t")
    assert store.state(pid) == "PRODUCTION_COMPLETED"


def test_stopped_project_cannot_produce(tmp_path):
    store = ApprovalStore(tmp_path / "a.sqlite")
    for state in ("SAMPLE_GENERATING", "SAMPLE_COMPLETED", "APPROVAL_PENDING"):
        store.transition("p", state, actor="t")
    assert store.decide("p", sample_artifact_id="s", trailer_sha256="h", decision="STOP", actor="t") == "STOPPED"
    with pytest.raises(IllegalStateError):
        store.transition("p", "PRODUCTION_QUEUED", actor="t")
    assert not store.is_approved("p", "s", "h")


def test_clip_seconds():
    assert Clip("c", "SH_00001", 1.0, 3.5).seconds == 2.5


def test_balance_shards_by_duration():
    from video_platform.stages.production import Producer

    shards = Producer.balance([8.0, 1.0, 7.0, 2.0, 5.0, 5.0], 2, weight=lambda x: x)
    loads = sorted(sum(s) for s in shards)
    assert loads == [14.0, 14.0]
    assert Producer.balance([3.0], 4, weight=lambda x: x) == [[3.0]]  # never creates empty shards


def test_work_resolution_keeps_aspect_and_area():
    from video_platform.stages.production import work_resolution

    assert work_resolution([1280, 720], 720, 1280) == (720, 1280)      # high: native 9:16 720p
    assert work_resolution([832, 480], 1080, 1920) == (480, 832)        # balanced: ~480p area, 9:16, /16
    assert work_resolution([1280, 720], 480, 854) == (480, 854)         # never upscale past the output


def test_profiles_validated(make_project):
    import pytest as _pytest

    from video_platform.config.loader import load_project_config
    from video_platform.core.errors import ConfigError

    from ..conftest import minimal_config

    loaded = load_project_config(make_project())
    assert set(loaded.config.production.profiles) == {"high", "balanced"}
    assert loaded.config.production.profiles["balanced"].wan_steps == 4
    with _pytest.raises(ConfigError, match="not defined"):
        load_project_config(make_project(minimal_config(sample={"compare_profiles": ["ultra"]})))


def test_approval_records_profile(tmp_path):
    store = ApprovalStore(tmp_path / "a.sqlite")
    for state in ("SAMPLE_GENERATING", "SAMPLE_COMPLETED", "APPROVAL_PENDING"):
        store.transition("p", state, actor="t")
    store.decide("p", sample_artifact_id="s", trailer_sha256="hb", decision="PROCEED", actor="t", profile="balanced")
    assert store.approved_profile("p", "s") == "balanced"
    assert store.is_approved("p", "s", "hb") and not store.is_approved("p", "s", "hh")
