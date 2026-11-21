
import csv
import json
import math
import subprocess
from bisect import bisect_left
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------
# INPUT FILES
# ---------------------------------------------------------

TRANSCRIPT_FILE = Path("transcripts/chinese_transcript_clean.json")
DIARIZATION_FILE = Path("diarization/speaker_turns.json")
SCENE_FILE = Path("scenes/scene_timeline.csv")
AUDIO_FILE = Path("audio/source.wav")


# ---------------------------------------------------------
# OUTPUT FILES
# ---------------------------------------------------------

OUTPUT_DIR = Path("timeline")

JSON_OUTPUT = OUTPUT_DIR / "master_timeline.json"
CSV_OUTPUT = OUTPUT_DIR / "master_timeline.csv"
REPORT_OUTPUT = OUTPUT_DIR / "timeline_quality_report.json"


# ---------------------------------------------------------
# FILE LOADING
# ---------------------------------------------------------

def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_audio_duration():
    if not AUDIO_FILE.exists():
        raise FileNotFoundError(f"Audio file not found: {AUDIO_FILE}")

    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(AUDIO_FILE),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    return float(result.stdout.strip())


def load_scenes():
    if not SCENE_FILE.exists():
        raise FileNotFoundError(f"File not found: {SCENE_FILE}")

    scenes = []

    with open(
        SCENE_FILE,
        newline="",
        encoding="utf-8-sig"
    ) as f:

        reader = csv.DictReader(f)

        required_columns = {
            "scene_id",
            "start_seconds",
            "end_seconds",
        }

        missing = required_columns - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                f"Missing scene CSV columns: {sorted(missing)}"
            )

        for row in reader:
            start = float(row["start_seconds"])
            end = float(row["end_seconds"])

            if end <= start:
                continue

            scenes.append({
                "scene_id": str(row["scene_id"]),
                "start": start,
                "end": end,
            })

    return sorted(scenes, key=lambda x: x["start"])


def load_speaker_turns():
    data = load_json(DIARIZATION_FILE)

    turns = []

    for turn in data["turns"]:
        start = float(turn["start"])
        end = float(turn["end"])

        if end <= start:
            continue

        turns.append({
            "start": start,
            "end": end,
            "speaker_id": turn["speaker_id"],
        })

    return sorted(turns, key=lambda x: x["start"])


# ---------------------------------------------------------
# TIMELINE MATCHING
# ---------------------------------------------------------

def overlap_seconds(a_start, a_end, b_start, b_end):
    return max(
        0.0,
        min(a_end, b_end) - max(a_start, b_start)
    )


def find_overlaps(start, end, records, starts):
    """
    Find records overlapping the requested time interval.

    Records must be sorted by start time.
    """

    upper = bisect_left(starts, end)

    result = []

    for record in records[:upper]:

        if record["end"] <= start:
            continue

        overlap = overlap_seconds(
            start,
            end,
            record["start"],
            record["end"],
        )

        if overlap > 0:
            result.append((record, overlap))

    return result


def classify_match(ratio):

    if ratio >= 0.70:
        return "strong"

    if ratio >= 0.30:
        return "moderate"

    if ratio > 0:
        return "weak"

    return "none"


def calculate_ratio(overlap, duration):

    if duration <= 0:
        return 0.0

    return min(1.0, overlap / duration)


# ---------------------------------------------------------
# QUALITY REPORT INITIALIZATION
# ---------------------------------------------------------

def create_quality_report(audio_duration, transcript_count,
                          speaker_count, scene_count):

    return {
        "audio_duration_seconds": audio_duration,

        "transcript_segment_count": transcript_count,
        "speaker_turn_count": speaker_count,
        "scene_count": scene_count,

        "valid_timeline_segment_count": 0,

        "matched_speaker_segments": 0,
        "unmatched_speaker_segments": 0,

        "matched_scene_segments": 0,
        "unmatched_scene_segments": 0,

        "strong_speaker_matches": 0,
        "moderate_speaker_matches": 0,
        "weak_speaker_matches": 0,
        "none_speaker_matches": 0,

        "strong_scene_matches": 0,
        "moderate_scene_matches": 0,
        "weak_scene_matches": 0,
        "none_scene_matches": 0,

        "segments_outside_audio": [],
        "invalid_segments": [],

        "speaker_distribution": {},
        "scene_distribution_top_20": {},

        "timestamp_warning": (
            "Transcript segments extending beyond audio duration "
            "are flagged, not clipped or silently modified."
        ),
    }


# ---------------------------------------------------------
# MAIN TIMELINE BUILDER
# ---------------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading source files...")

    transcript_data = load_json(TRANSCRIPT_FILE)

    transcript_segments = transcript_data["segments"]

    speaker_turns = load_speaker_turns()

    scenes = load_scenes()

    audio_duration = get_audio_duration()

    print("Audio duration:", round(audio_duration, 3), "seconds")
    print("Transcript segments:", len(transcript_segments))
    print("Speaker turns:", len(speaker_turns))
    print("Scenes:", len(scenes))

    # Precompute start-time arrays for overlap lookup.
    speaker_starts = [
        turn["start"] for turn in speaker_turns
    ]

    scene_starts = [
        scene["start"] for scene in scenes
    ]

    quality = create_quality_report(
        audio_duration,
        len(transcript_segments),
        len(speaker_turns),
        len(scenes),
    )

    timeline = []

    speaker_distribution = Counter()
    scene_distribution = Counter()

    print("\nBuilding timeline...")

    # -----------------------------------------------------
    # PROCESS TRANSCRIPT SEGMENTS
    # -----------------------------------------------------

    for index, segment in enumerate(transcript_segments):

        start = float(segment["start"])
        end = float(segment["end"])

        duration = end - start

        segment_id = segment.get("id", index)

        # Validate timestamps.
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or duration <= 0
        ):

            quality["invalid_segments"].append({
                "segment_id": segment_id,
                "start": start,
                "end": end,
            })

            continue

        outside_audio = (
            start < 0 or end > audio_duration
        )

        if outside_audio:

            quality["segments_outside_audio"].append({
                "segment_id": segment_id,
                "start": start,
                "end": end,
                "audio_duration": audio_duration,
            })

        # -------------------------------------------------
        # MATCH SPEAKER
        # -------------------------------------------------

        speaker_overlaps = find_overlaps(
            start,
            end,
            speaker_turns,
            speaker_starts,
        )

        # Aggregate multiple turns belonging to the same speaker.
        speaker_totals = Counter()

        for turn, overlap in speaker_overlaps:

            speaker_totals[turn["speaker_id"]] += overlap

        best_speaker = None
        speaker_overlap = 0.0

        if speaker_totals:

            best_speaker, speaker_overlap = (
                speaker_totals.most_common(1)[0]
            )

        speaker_ratio = calculate_ratio(
            speaker_overlap,
            duration,
        )

        speaker_confidence = classify_match(
            speaker_ratio
        )

        if best_speaker:

            quality["matched_speaker_segments"] += 1

            speaker_distribution[best_speaker] += 1

        else:

            quality["unmatched_speaker_segments"] += 1

        # All four classifications are initialized.
        quality[
            f"{speaker_confidence}_speaker_matches"
        ] += 1

        # -------------------------------------------------
        # MATCH SCENE
        # -------------------------------------------------

        scene_overlaps = find_overlaps(
            start,
            end,
            scenes,
            scene_starts,
        )

        scene_overlaps.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        best_scene = None
        scene_overlap = 0.0

        if scene_overlaps:

            best_scene = scene_overlaps[0][0]["scene_id"]

            scene_overlap = scene_overlaps[0][1]

        scene_ratio = calculate_ratio(
            scene_overlap,
            duration,
        )

        scene_confidence = classify_match(
            scene_ratio
        )

        if best_scene is not None:

            quality["matched_scene_segments"] += 1

            scene_distribution[best_scene] += 1

        else:

            quality["unmatched_scene_segments"] += 1

        quality[
            f"{scene_confidence}_scene_matches"
        ] += 1

        # Include all scenes crossed by this dialogue segment.
        overlapping_scene_ids = [
            item[0]["scene_id"]
            for item in scene_overlaps
        ]

        # -------------------------------------------------
        # CREATE MASTER TIMELINE RECORD
        # -------------------------------------------------

        record = {
            "segment_id": segment_id,

            "start": start,
            "end": end,
            "duration": round(duration, 3),

            "chinese_text": segment.get(
                "text", ""
            ).strip(),

            "speaker_id": best_speaker,

            "speaker_overlap_seconds": round(
                speaker_overlap, 3
            ),

            "speaker_overlap_ratio": round(
                speaker_ratio, 4
            ),

            "speaker_confidence": speaker_confidence,

            "scene_id": best_scene,

            "scene_overlap_seconds": round(
                scene_overlap, 3
            ),

            "scene_overlap_ratio": round(
                scene_ratio, 4
            ),

            "scene_confidence": scene_confidence,

            "overlapping_scene_ids": overlapping_scene_ids,

            # Assigned during a later character-mapping stage.
            "character_id": None,

            "outside_audio": outside_audio,
        }

        timeline.append(record)

        if (index + 1) % 500 == 0:

            print(
                f"Processed {index + 1}/"
                f"{len(transcript_segments)} segments"
            )

    # -----------------------------------------------------
    # FINALIZE QUALITY REPORT
    # -----------------------------------------------------

    quality["valid_timeline_segment_count"] = len(timeline)

    quality["speaker_distribution"] = dict(
        speaker_distribution
    )

    quality["scene_distribution_top_20"] = dict(
        scene_distribution.most_common(20)
    )

    # -----------------------------------------------------
    # SAVE JSON
    # -----------------------------------------------------

    output_data = {
        "source_audio": str(AUDIO_FILE),

        "audio_duration_seconds": audio_duration,

        "source_transcript": str(TRANSCRIPT_FILE),

        "source_diarization": str(DIARIZATION_FILE),

        "source_scene_timeline": str(SCENE_FILE),

        "segment_count": len(timeline),

        "segments": timeline,
    }

    print("\nSaving JSON...")

    with open(
        JSON_OUTPUT,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output_data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # -----------------------------------------------------
    # SAVE CSV
    # -----------------------------------------------------

    csv_fields = [
        "segment_id",
        "start",
        "end",
        "duration",
        "chinese_text",

        "speaker_id",
        "speaker_overlap_seconds",
        "speaker_overlap_ratio",
        "speaker_confidence",

        "scene_id",
        "scene_overlap_seconds",
        "scene_overlap_ratio",
        "scene_confidence",

        "overlapping_scene_ids",

        "character_id",
        "outside_audio",
    ]

    print("Saving CSV...")

    with open(
        CSV_OUTPUT,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=csv_fields,
        )

        writer.writeheader()

        for record in timeline:

            row = record.copy()

            row["overlapping_scene_ids"] = "|".join(
                row["overlapping_scene_ids"]
            )

            writer.writerow(row)

    # -----------------------------------------------------
    # SAVE QUALITY REPORT
    # -----------------------------------------------------

    print("Saving quality report...")

    with open(
        REPORT_OUTPUT,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            quality,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # -----------------------------------------------------
    # FINAL SUMMARY
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("MASTER TIMELINE COMPLETED")
    print("=" * 60)

    print("\nSource statistics:")

    print("Audio duration:", round(audio_duration, 3))
    print("Transcript segments:", len(transcript_segments))
    print("Valid timeline segments:", len(timeline))
    print("Speaker turns:", len(speaker_turns))
    print("Scenes:", len(scenes))

    print("\nSpeaker matching:")

    print(
        "Matched:",
        quality["matched_speaker_segments"]
    )

    print(
        "Unmatched:",
        quality["unmatched_speaker_segments"]
    )

    print(
        "Strong:",
        quality["strong_speaker_matches"]
    )

    print(
        "Moderate:",
        quality["moderate_speaker_matches"]
    )

    print(
        "Weak:",
        quality["weak_speaker_matches"]
    )

    print(
        "None:",
        quality["none_speaker_matches"]
    )

    print("\nScene matching:")

    print(
        "Matched:",
        quality["matched_scene_segments"]
    )

    print(
        "Unmatched:",
        quality["unmatched_scene_segments"]
    )

    print(
        "Outside audio:",
        len(quality["segments_outside_audio"])
    )

    print(
        "Invalid segments:",
        len(quality["invalid_segments"])
    )

    print("\nOutput files:")

    print(JSON_OUTPUT)
    print(CSV_OUTPUT)
    print(REPORT_OUTPUT)

    print("\nSpeaker distribution:")

    for speaker, count in speaker_distribution.most_common():

        print(f"{speaker}: {count} dialogue segments")

    print("\nDone.")


if __name__ == "__main__":
    main()

