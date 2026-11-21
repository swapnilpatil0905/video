
import csv
import json
import math
from pathlib import Path


TIMELINE_FILE = Path("timeline/master_timeline.json")
TRANSCRIPT_FILE = Path("transcripts/chinese_transcript_clean.json")
DIARIZATION_FILE = Path("diarization/speaker_turns.json")

OUTPUT_DIR = Path("timeline/audit")


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created: {path} ({len(rows)} records)")


def nearest_speakers(start, end, turns):
    previous = None
    following = None

    for turn in turns:
        if turn["end"] <= start:
            if previous is None or turn["end"] > previous["end"]:
                previous = turn

        if turn["start"] >= end:
            if following is None or turn["start"] < following["start"]:
                following = turn

    return {
        "previous_speaker": previous["speaker_id"] if previous else None,
        "previous_speaker_end": previous["end"] if previous else None,
        "next_speaker": following["speaker_id"] if following else None,
        "next_speaker_start": following["start"] if following else None,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading files...")

    timeline_data = load_json(TIMELINE_FILE)
    transcript_data = load_json(TRANSCRIPT_FILE)
    diarization_data = load_json(DIARIZATION_FILE)

    timeline = timeline_data["segments"]
    transcript = transcript_data["segments"]

    turns = sorted(
        diarization_data["turns"],
        key=lambda x: float(x["start"])
    )

    # -------------------------------------------------
    # 1. INVALID TRANSCRIPT SEGMENTS
    # -------------------------------------------------

    invalid_rows = []

    for index, segment in enumerate(transcript):
        start = float(segment["start"])
        end = float(segment["end"])

        reasons = []

        if not math.isfinite(start) or not math.isfinite(end):
            reasons.append("non_finite_timestamp")
        elif end <= start:
            reasons.append("end_not_after_start")

        if start < 0:
            reasons.append("negative_start")

        if end > timeline_data["audio_duration_seconds"]:
            reasons.append("beyond_audio_duration")

        if reasons:
            invalid_rows.append({
                "segment_id": segment.get("id", index),
                "start": start,
                "end": end,
                "duration": end - start,
                "chinese_text": segment.get("text", "").strip(),
                "reason": "|".join(reasons),
            })

    # -------------------------------------------------
    # 2. UNMATCHED AND WEAK SPEAKER SEGMENTS
    # -------------------------------------------------

    unmatched_rows = []
    weak_rows = []

    for segment in timeline:
        confidence = segment["speaker_confidence"]

        if confidence not in ("none", "weak"):
            continue

        nearest = nearest_speakers(
            float(segment["start"]),
            float(segment["end"]),
            turns,
        )

        row = {
            "segment_id": segment["segment_id"],
            "start": segment["start"],
            "end": segment["end"],
            "duration": segment["duration"],
            "chinese_text": segment["chinese_text"],
            "speaker_id": segment["speaker_id"],
            "speaker_confidence": confidence,
            "speaker_overlap_seconds": segment["speaker_overlap_seconds"],
            "speaker_overlap_ratio": segment["speaker_overlap_ratio"],
            "scene_id": segment["scene_id"],
            "scene_confidence": segment["scene_confidence"],
            "overlapping_scene_ids": "|".join(
                segment["overlapping_scene_ids"]
            ),
            **nearest,
        }

        if confidence == "none":
            unmatched_rows.append(row)
        else:
            weak_rows.append(row)

    # -------------------------------------------------
    # 3. WRITE AUDIT FILES
    # -------------------------------------------------

    fields = [
        "segment_id",
        "start",
        "end",
        "duration",
        "chinese_text",
        "speaker_id",
        "speaker_confidence",
        "speaker_overlap_seconds",
        "speaker_overlap_ratio",
        "scene_id",
        "scene_confidence",
        "overlapping_scene_ids",
        "previous_speaker",
        "previous_speaker_end",
        "next_speaker",
        "next_speaker_start",
    ]

    invalid_fields = [
        "segment_id",
        "start",
        "end",
        "duration",
        "chinese_text",
        "reason",
    ]

    write_csv(
        OUTPUT_DIR / "invalid_transcript_segments.csv",
        invalid_rows,
        invalid_fields,
    )

    write_csv(
        OUTPUT_DIR / "unmatched_speaker_segments.csv",
        unmatched_rows,
        fields,
    )

    write_csv(
        OUTPUT_DIR / "weak_speaker_segments.csv",
        weak_rows,
        fields,
    )

    # -------------------------------------------------
    # 4. SUMMARY
    # -------------------------------------------------

    summary = {
        "invalid_transcript_segments": len(invalid_rows),
        "unmatched_speaker_segments": len(unmatched_rows),
        "weak_speaker_segments": len(weak_rows),
        "total_timeline_segments": len(timeline),
    }

    with open(
        OUTPUT_DIR / "audit_summary.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 55)
    print("TIMELINE AUDIT COMPLETED")
    print("=" * 55)

    print("Invalid transcript segments:", len(invalid_rows))
    print("Unmatched speaker segments:", len(unmatched_rows))
    print("Weak speaker segments:", len(weak_rows))

    print("\nAudit directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

