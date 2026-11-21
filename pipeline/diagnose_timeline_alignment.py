
import csv
import json
from pathlib import Path
from collections import defaultdict


# ---------------------------------------------------------
# INPUTS
# ---------------------------------------------------------

CLEAN_TRANSCRIPT = Path("transcripts/chinese_transcript_clean.json")
ORIGINAL_TRANSCRIPT = Path("transcripts/chinese_transcript.json")

TIMELINE_FILE = Path("timeline/master_timeline.json")
DIARIZATION_FILE = Path("diarization/speaker_turns.json")
QUALITY_FILE = Path("timeline/timeline_quality_report.json")

CHARACTER_MAPPING_FILE = Path(
    "characters/character_scene_mapping.csv"
)


# ---------------------------------------------------------
# OUTPUTS
# ---------------------------------------------------------

AUDIT_DIR = Path("timeline/audit")

DIAGNOSTIC_CSV = AUDIT_DIR / "alignment_diagnostics.csv"


# ---------------------------------------------------------
# LOADERS
# ---------------------------------------------------------

def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_character_mapping():
    mapping = defaultdict(set)

    if not CHARACTER_MAPPING_FILE.exists():
        print("WARNING: Character mapping file missing.")
        return mapping

    with open(
        CHARACTER_MAPPING_FILE,
        newline="",
        encoding="utf-8-sig"
    ) as f:

        for row in csv.DictReader(f):

            scene_id = str(row["scene_id"])
            character_id = row["character_id"]

            if character_id and character_id != "UNASSIGNED":
                mapping[scene_id].add(character_id)

    return mapping


def load_original_transcript():
    if not ORIGINAL_TRANSCRIPT.exists():
        print("WARNING: Original transcript not found.")
        return {}

    data = load_json(ORIGINAL_TRANSCRIPT)

    return {
        str(segment.get("id")): segment
        for segment in data.get("segments", [])
    }


# ---------------------------------------------------------
# SPEAKER TURN ANALYSIS
# ---------------------------------------------------------

def find_nearest_turns(start, end, turns):

    overlapping = []
    previous = None
    following = None

    for turn in turns:

        turn_start = turn["start"]
        turn_end = turn["end"]

        if turn_end > start and turn_start < end:
            overlapping.append(turn)

        if turn_end <= start:
            if previous is None or turn_end > previous["end"]:
                previous = turn

        if turn_start >= end:
            if following is None or turn_start < following["start"]:
                following = turn

    previous_gap = None
    following_gap = None

    if previous:
        previous_gap = round(start - previous["end"], 3)

    if following:
        following_gap = round(following["start"] - end, 3)

    return {
        "overlapping_turns": overlapping,
        "previous_turn": previous,
        "following_turn": following,
        "previous_gap_seconds": previous_gap,
        "following_gap_seconds": following_gap,
    }


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading files...")

    clean_data = load_json(CLEAN_TRANSCRIPT)
    timeline_data = load_json(TIMELINE_FILE)
    diarization_data = load_json(DIARIZATION_FILE)
    quality = load_json(QUALITY_FILE)

    original_segments = load_original_transcript()

    turns = sorted(
        diarization_data["turns"],
        key=lambda x: x["start"]
    )

    character_mapping = load_character_mapping()

    timeline_segments = timeline_data["segments"]

    timeline_by_id = {
        str(segment["segment_id"]): segment
        for segment in timeline_segments
    }

    clean_segments = clean_data["segments"]

    rows = []

    # -----------------------------------------------------
    # INVALID TRANSCRIPT SEGMENTS
    # -----------------------------------------------------

    print("\nINVALID TRANSCRIPT SEGMENTS")
    print("=" * 70)

    invalid_segments = quality.get("invalid_segments", [])

    for invalid in invalid_segments:

        segment_id = str(invalid["segment_id"])

        clean_segment = next(
            (
                item for item in clean_segments
                if str(item.get("id")) == segment_id
            ),
            None,
        )

        original_segment = original_segments.get(segment_id)

        start = float(invalid["start"])
        end = float(invalid["end"])

        # Use the available timestamp as the reference point.
        reference = min(start, end)

        nearby = find_nearest_turns(
            reference,
            reference + 0.001,
            turns,
        )

        print("\nSegment:", segment_id)
        print("Clean transcript:", clean_segment)
        print("Original transcript:", original_segment)

        print("Previous speaker turn:", nearby["previous_turn"])
        print("Next speaker turn:", nearby["following_turn"])

        rows.append({
            "issue_type": "invalid_transcript",
            "segment_id": segment_id,
            "start": start,
            "end": end,
            "duration": end - start,
            "chinese_text": (
                clean_segment.get("text", "")
                if clean_segment else ""
            ),
            "original_start": (
                original_segment.get("start")
                if original_segment else None
            ),
            "original_end": (
                original_segment.get("end")
                if original_segment else None
            ),
            "original_text": (
                original_segment.get("text", "")
                if original_segment else ""
            ),
            "speaker_id": "",
            "overlapping_speakers": "",
            "previous_speaker": (
                nearby["previous_turn"]["speaker_id"]
                if nearby["previous_turn"] else ""
            ),
            "previous_gap_seconds": nearby["previous_gap_seconds"],
            "next_speaker": (
                nearby["following_turn"]["speaker_id"]
                if nearby["following_turn"] else ""
            ),
            "next_gap_seconds": nearby["following_gap_seconds"],
            "scene_id": "",
            "visible_characters": "",
        })

    # -----------------------------------------------------
    # UNMATCHED AND WEAK SEGMENTS
    # -----------------------------------------------------

    print("\nUNMATCHED AND WEAK SPEAKER SEGMENTS")
    print("=" * 70)

    issue_counts = defaultdict(int)

    for segment in timeline_segments:

        confidence = segment["speaker_confidence"]

        if confidence not in ("none", "weak"):
            continue

        issue_type = (
            "unmatched_speaker"
            if confidence == "none"
            else "weak_speaker"
        )

        issue_counts[issue_type] += 1

        start = float(segment["start"])
        end = float(segment["end"])

        nearby = find_nearest_turns(start, end, turns)

        scene_ids = segment.get(
            "overlapping_scene_ids", []
        )

        visible_characters = set()

        for scene_id in scene_ids:
            visible_characters.update(
                character_mapping.get(str(scene_id), set())
            )

        overlapping_speakers = sorted({
            turn["speaker_id"]
            for turn in nearby["overlapping_turns"]
        })

        original_segment = original_segments.get(
            str(segment["segment_id"])
        )

        row = {
            "issue_type": issue_type,
            "segment_id": segment["segment_id"],
            "start": start,
            "end": end,
            "duration": segment["duration"],
            "chinese_text": segment["chinese_text"],
            "original_start": (
                original_segment.get("start")
                if original_segment else None
            ),
            "original_end": (
                original_segment.get("end")
                if original_segment else None
            ),
            "original_text": (
                original_segment.get("text", "")
                if original_segment else ""
            ),
            "speaker_id": segment.get("speaker_id") or "",
            "overlapping_speakers": "|".join(
                overlapping_speakers
            ),
            "previous_speaker": (
                nearby["previous_turn"]["speaker_id"]
                if nearby["previous_turn"] else ""
            ),
            "previous_gap_seconds": nearby["previous_gap_seconds"],
            "next_speaker": (
                nearby["following_turn"]["speaker_id"]
                if nearby["following_turn"] else ""
            ),
            "next_gap_seconds": nearby["following_gap_seconds"],
            "scene_id": segment.get("scene_id") or "",
            "visible_characters": "|".join(
                sorted(visible_characters)
            ),
        }

        rows.append(row)

    # -----------------------------------------------------
    # EXPORT
    # -----------------------------------------------------

    fields = [
        "issue_type",
        "segment_id",
        "start",
        "end",
        "duration",
        "chinese_text",
        "original_start",
        "original_end",
        "original_text",
        "speaker_id",
        "overlapping_speakers",
        "previous_speaker",
        "previous_gap_seconds",
        "next_speaker",
        "next_gap_seconds",
        "scene_id",
        "visible_characters",
    ]

    with open(
        DIAGNOSTIC_CSV,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(rows)

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("ALIGNMENT DIAGNOSTICS COMPLETED")
    print("=" * 70)

    print("Invalid transcript segments:", len(invalid_segments))

    print(
        "Unmatched speaker segments:",
        issue_counts["unmatched_speaker"]
    )

    print(
        "Weak speaker segments:",
        issue_counts["weak_speaker"]
    )

    print("Diagnostic records:", len(rows))

    print("\nOutput:")
    print(DIAGNOSTIC_CSV)


if __name__ == "__main__":
    main()

