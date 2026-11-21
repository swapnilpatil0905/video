
import csv
import json
from pathlib import Path
from collections import defaultdict


ROOT = Path(".")

TIMELINE_FILE = ROOT / "timeline/master_timeline.json"
SCENES_FILE = ROOT / "scenes/scene_timeline.csv"
CHARACTER_MAPPING_FILE = ROOT / "characters/character_scene_mapping.csv"

OUTPUT_DIR = ROOT / "characters"
OUTPUT_JSON = OUTPUT_DIR / "speaker_character_refined.json"
OUTPUT_CSV = OUTPUT_DIR / "speaker_character_refined.csv"


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_segment_list(data):
    """Find the timeline segment list without assuming one exact JSON layout."""
    if isinstance(data, list):
        if data and isinstance(data[0], dict):
            if any(k in data[0] for k in ("speaker_id", "speaker", "start")):
                return data

    if isinstance(data, dict):
        preferred_keys = [
            "segments",
            "timeline",
            "records",
            "items",
        ]

        for key in preferred_keys:
            value = data.get(key)
            if isinstance(value, list):
                if not value or isinstance(value[0], dict):
                    return value

        for value in data.values():
            if isinstance(value, (dict, list)):
                result = find_segment_list(value)
                if result is not None:
                    return result

    return None


def first_value(record, keys, default=None):
    for key in keys:
        value = record.get(key)
        if value is not None:
            return value
    return default


def parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_scenes():
    scenes = []

    with SCENES_FILE.open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            scene_id = int(float(row["scene_id"]))
            start = float(row["start_seconds"])
            end = float(row["end_seconds"])

            if end > start:
                scenes.append({
                    "scene_id": scene_id,
                    "start": start,
                    "end": end,
                })

    return sorted(scenes, key=lambda x: x["start"])


def load_character_mapping():
    mapping = defaultdict(set)

    with CHARACTER_MAPPING_FILE.open(
        "r", encoding="utf-8-sig"
    ) as f:
        for row in csv.DictReader(f):
            scene_id = int(float(row["scene_id"]))
            character_id = row["character_id"].strip()

            if character_id and character_id != "UNASSIGNED":
                mapping[scene_id].add(character_id)

    return mapping


def load_speech_segments():
    data = load_json(TIMELINE_FILE)
    records = find_segment_list(data)

    if records is None:
        raise RuntimeError(
            "Could not find the timeline segment list. "
            "Inspect timeline/master_timeline.json."
        )

    segments = []

    for record in records:
        if not isinstance(record, dict):
            continue

        start = parse_float(
            first_value(record, ["start", "start_seconds"])
        )
        end = parse_float(
            first_value(record, ["end", "end_seconds"])
        )

        speaker = first_value(
            record,
            ["speaker_id", "speaker", "speaker_label"]
        )

        if start is None or end is None:
            continue

        if end <= start or not speaker or str(speaker).lower() == "none":
            continue

        confidence = str(
            first_value(
                record,
                ["speaker_confidence", "confidence"],
                "unknown"
            )
        ).lower()

        segments.append({
            "start": start,
            "end": end,
            "speaker_id": str(speaker),
            "confidence": confidence,
        })

    return segments


def confidence_weight(confidence):
    if confidence == "strong":
        return 1.0
    if confidence == "moderate":
        return 0.7
    if confidence == "weak":
        return 0.35
    return 0.5


def overlap_seconds(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading timeline...")
    speech_segments = load_speech_segments()

    print("Loading scene boundaries...")
    scenes = load_scenes()

    print("Loading character-to-scene mapping...")
    character_mapping = load_character_mapping()

    print("\nLoaded:")
    print(f"  Speech segments: {len(speech_segments)}")
    print(f"  Scenes: {len(scenes)}")
    print(f"  Scenes with detected characters: {len(character_mapping)}")

    # Evidence is accumulated for each speaker-character pair.
    evidence = defaultdict(lambda: {
        "overlap_seconds": 0.0,
        "weighted_overlap_seconds": 0.0,
        "segment_ids": set(),
        "scene_ids": set(),
        "multi_character_seconds": 0.0,
        "single_character_seconds": 0.0,
        "confidence_seconds": defaultdict(float),
    })

    speaker_total_seconds = defaultdict(float)
    speaker_matched_seconds = defaultdict(float)

    # Use a moving scene index to avoid repeatedly scanning earlier scenes.
    scene_index = 0

    for segment_index, segment in enumerate(speech_segments):
        start = segment["start"]
        end = segment["end"]
        speaker = segment["speaker_id"]
        duration = end - start

        speaker_total_seconds[speaker] += duration

        while (
            scene_index < len(scenes)
            and scenes[scene_index]["end"] <= start
        ):
            scene_index += 1

        current_index = scene_index

        while current_index < len(scenes):
            scene = scenes[current_index]

            if scene["start"] >= end:
                break

            overlap = overlap_seconds(
                start, end, scene["start"], scene["end"]
            )

            if overlap > 0:
                scene_id = scene["scene_id"]
                characters = character_mapping.get(scene_id, set())

                if characters:
                    speaker_matched_seconds[speaker] += overlap

                    is_ambiguous = len(characters) > 1
                    weight = confidence_weight(segment["confidence"])

                    # Each character receives the same temporal evidence
                    # for this scene. Multi-character scenes are explicitly
                    # marked ambiguous and must not be treated as proof.
                    for character in characters:
                        key = (speaker, character)
                        item = evidence[key]

                        item["overlap_seconds"] += overlap
                        item["weighted_overlap_seconds"] += (
                            overlap * weight
                        )
                        item["segment_ids"].add(segment_index)
                        item["scene_ids"].add(scene_id)
                        item["confidence_seconds"][
                            segment["confidence"]
                        ] += overlap

                        if is_ambiguous:
                            item["multi_character_seconds"] += overlap
                        else:
                            item["single_character_seconds"] += overlap

            current_index += 1

    # Calculate the candidate ranking within each speaker.
    by_speaker = defaultdict(list)

    for (speaker, character), item in evidence.items():
        overlap = item["overlap_seconds"]
        total = speaker_total_seconds[speaker]

        share = (
            100.0 * overlap / total
            if total > 0 else 0.0
        )

        ambiguous_ratio = (
            item["multi_character_seconds"] / overlap
            if overlap > 0 else 0.0
        )

        single_ratio = (
            item["single_character_seconds"] / overlap
            if overlap > 0 else 0.0
        )

        weighted = item["weighted_overlap_seconds"]

        by_speaker[speaker].append({
            "speaker_id": speaker,
            "character_id": character,
            "overlap_seconds": round(overlap, 3),
            "weighted_overlap_seconds": round(weighted, 3),
            "speaker_total_seconds": round(total, 3),
            "speaker_overlap_percentage": round(share, 2),
            "unique_scenes": len(item["scene_ids"]),
            "unique_segments": len(item["segment_ids"]),
            "single_character_seconds": round(
                item["single_character_seconds"], 3
            ),
            "multi_character_seconds": round(
                item["multi_character_seconds"], 3
            ),
            "single_character_percentage": round(
                single_ratio * 100, 2
            ),
            "ambiguous_percentage": round(
                ambiguous_ratio * 100, 2
            ),
            "confidence_seconds": {
                k: round(v, 3)
                for k, v in item["confidence_seconds"].items()
            },
        })

    final_rows = []

    for speaker, candidates in by_speaker.items():
        candidates.sort(
            key=lambda x: (
                x["single_character_seconds"],
                x["weighted_overlap_seconds"],
                x["unique_scenes"],
            ),
            reverse=True,
        )

        for rank, candidate in enumerate(candidates, start=1):
            candidate["rank"] = rank

            # These are evidence labels, not identity verification.
            if (
                candidate["single_character_seconds"] >= 30
                and candidate["single_character_percentage"] >= 50
                and candidate["unique_scenes"] >= 10
            ):
                candidate["evidence_level"] = "stronger_candidate"
            elif (
                candidate["single_character_seconds"] >= 10
                and candidate["unique_scenes"] >= 5
            ):
                candidate["evidence_level"] = "moderate_candidate"
            else:
                candidate["evidence_level"] = "weak_candidate"

            final_rows.append(candidate)

    final_rows.sort(
        key=lambda x: (x["speaker_id"], x["rank"])
    )

    output = {
        "method": (
            "Temporal overlap between diarized speech intervals, "
            "scene boundaries, and representative-frame character detections"
        ),
        "warning": (
            "Scene-level visibility is not proof of speaking identity. "
            "Multi-character scenes remain ambiguous. "
            "Manual verification is required."
        ),
        "speech_segment_count": len(speech_segments),
        "scene_count": len(scenes),
        "character_scene_count": len(character_mapping),
        "speaker_count": len(speaker_total_seconds),
        "speakers": {},
    }

    for speaker in sorted(speaker_total_seconds):
        candidates = [
            row for row in final_rows
            if row["speaker_id"] == speaker
        ]

        output["speakers"][speaker] = {
            "total_speech_seconds": round(
                speaker_total_seconds[speaker], 3
            ),
            "matched_speech_seconds": round(
                speaker_matched_seconds[speaker], 3
            ),
            "candidates": candidates,
        }

    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    columns = [
        "speaker_id",
        "rank",
        "character_id",
        "evidence_level",
        "overlap_seconds",
        "weighted_overlap_seconds",
        "speaker_total_seconds",
        "speaker_overlap_percentage",
        "unique_scenes",
        "unique_segments",
        "single_character_seconds",
        "multi_character_seconds",
        "single_character_percentage",
        "ambiguous_percentage",
    ]

    with OUTPUT_CSV.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for row in final_rows:
            writer.writerow({
                key: row.get(key)
                for key in columns
            })

    print("\n" + "=" * 65)
    print("REFINED SPEAKER-CHARACTER EVIDENCE GENERATED")
    print("=" * 65)

    print(f"Speakers: {len(speaker_total_seconds)}")
    print(f"Candidate relationships: {len(final_rows)}")

    for speaker in sorted(by_speaker):
        print(f"\n{speaker}")

        candidates = sorted(
            by_speaker[speaker],
            key=lambda x: (
                x["single_character_seconds"],
                x["weighted_overlap_seconds"],
            ),
            reverse=True,
        )

        for candidate in candidates[:5]:
            print(
                f"  {candidate['character_id']}: "
                f"single={candidate['single_character_seconds']:.1f}s, "
                f"ambiguous={candidate['multi_character_seconds']:.1f}s, "
                f"scenes={candidate['unique_scenes']}, "
                f"share={candidate['speaker_overlap_percentage']:.1f}%, "
                f"evidence={candidate['evidence_level']}"
            )

    print("\nOutput files:")
    print(OUTPUT_JSON)
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()

