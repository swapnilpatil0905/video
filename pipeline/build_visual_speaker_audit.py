
import csv
import json
import subprocess
from pathlib import Path
from collections import defaultdict


ROOT = Path(".")
VIDEO = ROOT / "input/source.mp4"

TIMELINE = ROOT / "timeline/master_timeline.json"
SCENES = ROOT / "scenes/scene_timeline.csv"
CHARACTER_MAPPING = ROOT / "characters/character_scene_mapping.csv"
REFINED_MAPPING = ROOT / "characters/speaker_character_refined.json"

OUTPUT_DIR = ROOT / "verification"
CLIPS_DIR = OUTPUT_DIR / "clips"
MANIFEST = OUTPUT_DIR / "speaker_visual_audit_manifest.csv"
SUMMARY = OUTPUT_DIR / "speaker_visual_audit_summary.json"

MAX_CLIPS_PER_SPEAKER = 8
CLIP_DURATION = 6.0
MIN_SCENE_OVERLAP = 0.35
MIN_SEGMENT_DURATION = 0.5
MIN_SCENE_GAP = 15.0


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_segment_list(data):
    if isinstance(data, list):
        if data and isinstance(data[0], dict):
            if any(k in data[0] for k in ("speaker_id", "speaker", "start")):
                return data

    if isinstance(data, dict):
        for key in ("segments", "timeline", "records", "items"):
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
        if record.get(key) is not None:
            return record[key]
    return default


def get_duration():
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(VIDEO),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def load_scenes():
    scenes = []

    with SCENES.open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            scenes.append({
                "scene_id": int(float(row["scene_id"])),
                "start": float(row["start_seconds"]),
                "end": float(row["end_seconds"]),
            })

    return sorted(scenes, key=lambda x: x["start"])


def load_character_mapping():
    mapping = defaultdict(set)

    with CHARACTER_MAPPING.open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            scene_id = int(float(row["scene_id"]))
            character_id = row["character_id"].strip()

            if character_id and character_id != "UNASSIGNED":
                mapping[scene_id].add(character_id)

    return mapping


def load_top_candidates():
    data = load_json(REFINED_MAPPING)
    result = {}

    for speaker, speaker_data in data["speakers"].items():
        candidates = speaker_data.get("candidates", [])

        if not candidates:
            continue

        # The refined report ranks candidates by exclusive-scene
        # overlap first, followed by weighted temporal evidence.
        result[speaker] = candidates[0]["character_id"]

    return result


def load_segments():
    data = load_json(TIMELINE)
    records = find_segment_list(data)

    if records is None:
        raise RuntimeError("Could not find timeline segments.")

    segments = []

    for index, record in enumerate(records):
        start = first_value(record, ["start", "start_seconds"])
        end = first_value(record, ["end", "end_seconds"])
        speaker = first_value(
            record, ["speaker_id", "speaker", "speaker_label"]
        )

        try:
            start = float(start)
            end = float(end)
        except (TypeError, ValueError):
            continue

        if not speaker or str(speaker).lower() == "none":
            continue

        if end <= start or end - start < MIN_SEGMENT_DURATION:
            continue

        confidence = str(
            first_value(
                record,
                ["speaker_confidence", "confidence"],
                "unknown",
            )
        ).lower()

        # Keep strong and moderate assignments for the first audit.
        if confidence not in ("strong", "moderate"):
            continue

        segments.append({
            "index": index,
            "segment_id": first_value(
                record, ["id", "segment_id"], index
            ),
            "start": start,
            "end": end,
            "speaker": str(speaker),
            "confidence": confidence,
            "text": str(
                first_value(record, ["chinese_text", "text"], "")
            ),
        })

    return segments


def overlap(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def build_candidates(segments, scenes, character_mapping, top_candidates):
    by_speaker = defaultdict(list)

    for segment in segments:
        speaker = segment["speaker"]
        expected_character = top_candidates.get(speaker)

        if not expected_character:
            continue

        for scene in scenes:
            if scene["start"] >= segment["end"]:
                break

            if scene["end"] <= segment["start"]:
                continue

            seconds = overlap(
                segment["start"],
                segment["end"],
                scene["start"],
                scene["end"],
            )

            if seconds < MIN_SCENE_OVERLAP:
                continue

            visible_characters = character_mapping.get(
                scene["scene_id"], set()
            )

            # Only select scenes where exactly one character
            # was detected and it matches this speaker's top candidate.
            if visible_characters != {expected_character}:
                continue

            overlap_start = max(segment["start"], scene["start"])
            overlap_end = min(segment["end"], scene["end"])

            midpoint = (overlap_start + overlap_end) / 2

            by_speaker[speaker].append({
                "speaker": speaker,
                "character": expected_character,
                "scene_id": scene["scene_id"],
                "segment_id": segment["segment_id"],
                "segment_start": segment["start"],
                "segment_end": segment["end"],
                "overlap_seconds": seconds,
                "overlap_start": overlap_start,
                "overlap_end": overlap_end,
                "midpoint": midpoint,
                "confidence": segment["confidence"],
                "chinese_text": segment["text"],
            })

    return by_speaker


def choose_diverse_clips(candidates):
    selected = []

    # Prefer longer exclusive-character overlap.
    candidates = sorted(
        candidates,
        key=lambda x: x["overlap_seconds"],
        reverse=True,
    )

    used_scenes = set()
    used_times = []

    for item in candidates:
        if item["scene_id"] in used_scenes:
            continue

        if any(
            abs(item["midpoint"] - t) < MIN_SCENE_GAP
            for t in used_times
        ):
            continue

        selected.append(item)
        used_scenes.add(item["scene_id"])
        used_times.append(item["midpoint"])

        if len(selected) >= MAX_CLIPS_PER_SPEAKER:
            break

    return sorted(selected, key=lambda x: x["midpoint"])


def extract_clip(item, video_duration):
    midpoint = item["midpoint"]

    clip_start = max(0.0, midpoint - CLIP_DURATION / 2)
    clip_end = min(video_duration, clip_start + CLIP_DURATION)

    if clip_end - clip_start < 2.0:
        return None

    filename = (
        f"{item['speaker']}_{item['character']}_"
        f"scene_{item['scene_id']:05d}_"
        f"seg_{item['segment_id']}.mp4"
    )

    output_path = CLIPS_DIR / filename

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-ss", f"{clip_start:.3f}",
        "-i", str(VIDEO),
        "-t", f"{clip_end - clip_start:.3f}",
        "-map", "0:v:0",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        str(output_path),
    ]

    subprocess.run(command, check=True)

    return {
        "clip_file": str(output_path),
        "clip_start": round(clip_start, 3),
        "clip_end": round(clip_end, 3),
        "clip_duration": round(clip_end - clip_start, 3),
    }


def main():
    if not VIDEO.exists():
        raise FileNotFoundError(VIDEO)

    for path in (
        TIMELINE,
        SCENES,
        CHARACTER_MAPPING,
        REFINED_MAPPING,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading source video duration...")
    video_duration = get_duration()

    print("Loading timeline and candidate mappings...")
    segments = load_segments()
    top_candidates = load_top_candidates()

    print("Loading scene and character information...")
    scenes = load_scenes()
    character_mapping = load_character_mapping()

    print("\nTop candidate mapping:")
    for speaker, character in sorted(top_candidates.items()):
        print(f"  {speaker} -> {character}")

    print("\nFinding exclusive-character dialogue intervals...")
    candidates = build_candidates(
        segments,
        scenes,
        character_mapping,
        top_candidates,
    )

    manifest_rows = []
    summary = {}

    for speaker in sorted(top_candidates):
        selected = choose_diverse_clips(candidates.get(speaker, []))
        character = top_candidates[speaker]

        print(f"\n{speaker} -> {character}")
        print(f"  Eligible intervals: {len(candidates.get(speaker, []))}")
        print(f"  Selected clips: {len(selected)}")

        extracted = 0

        for item in selected:
            clip_info = extract_clip(item, video_duration)

            if clip_info is None:
                continue

            row = {
                "speaker_id": speaker,
                "candidate_character_id": character,
                "scene_id": item["scene_id"],
                "segment_id": item["segment_id"],
                "speaker_confidence": item["confidence"],
                "scene_overlap_seconds": round(
                    item["overlap_seconds"], 3
                ),
                "dialogue_start": item["segment_start"],
                "dialogue_end": item["segment_end"],
                "clip_start": clip_info["clip_start"],
                "clip_end": clip_info["clip_end"],
                "clip_duration": clip_info["clip_duration"],
                "chinese_transcript": item["chinese_text"],
                "clip_file": clip_info["clip_file"],
            }

            manifest_rows.append(row)
            extracted += 1

        summary[speaker] = {
            "candidate_character_id": character,
            "eligible_intervals": len(candidates.get(speaker, [])),
            "clips_extracted": extracted,
        }

    columns = [
        "speaker_id",
        "candidate_character_id",
        "scene_id",
        "segment_id",
        "speaker_confidence",
        "scene_overlap_seconds",
        "dialogue_start",
        "dialogue_end",
        "clip_start",
        "clip_end",
        "clip_duration",
        "chinese_transcript",
        "clip_file",
    ]

    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(manifest_rows)

    with SUMMARY.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 65)
    print("VISUAL SPEAKER AUDIT CLIPS GENERATED")
    print("=" * 65)

    print(f"Total clips: {len(manifest_rows)}")
    print(f"Clip directory: {CLIPS_DIR}")
    print(f"Manifest: {MANIFEST}")
    print(f"Summary: {SUMMARY}")


if __name__ == "__main__":
    main()

