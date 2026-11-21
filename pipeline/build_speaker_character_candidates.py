
import csv
import json
from pathlib import Path
from collections import defaultdict, Counter


TIMELINE_FILE = Path("timeline/master_timeline.json")
CHARACTER_MAPPING_FILE = Path("characters/character_scene_mapping.csv")
CLUSTER_SUMMARY_FILE = Path("faces/cluster_summary.csv")

OUTPUT_FILE = Path(
    "characters/speaker_character_candidates.json"
)

CSV_OUTPUT = Path(
    "characters/speaker_character_candidates.csv"
)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_character_scenes():
    scene_characters = defaultdict(set)

    with open(
        CHARACTER_MAPPING_FILE,
        newline="",
        encoding="utf-8-sig"
    ) as f:

        for row in csv.DictReader(f):

            character = row["character_id"]
            scene = str(row["scene_id"])

            if character and character != "UNASSIGNED":
                scene_characters[scene].add(character)

    return scene_characters


def load_cluster_sizes():
    sizes = {}

    with open(
        CLUSTER_SUMMARY_FILE,
        newline="",
        encoding="utf-8-sig"
    ) as f:

        for row in csv.DictReader(f):

            character = row["cluster_id"]

            if character == "UNASSIGNED":
                continue

            sizes[character] = int(row["face_count"])

    return sizes


def main():

    print("Loading timeline and character data...")

    timeline = load_json(TIMELINE_FILE)

    scene_characters = load_character_scenes()
    cluster_sizes = load_cluster_sizes()

    # speaker -> character -> accumulated evidence
    evidence = defaultdict(
        lambda: defaultdict(
            lambda: {
                "segment_count": 0,
                "dialogue_seconds": 0.0,
                "strong_segments": 0,
                "moderate_segments": 0,
                "weak_segments": 0,
                "scene_ids": set(),
            }
        )
    )

    # Track total usable dialogue per speaker.
    speaker_totals = Counter()

    for segment in timeline["segments"]:

        speaker = segment.get("speaker_id")

        if not speaker:
            continue

        duration = float(segment["duration"])

        if duration <= 0:
            continue

        if segment.get("outside_audio"):
            continue

        speaker_totals[speaker] += duration

        scene_ids = segment.get(
            "overlapping_scene_ids", []
        )

        if not scene_ids:
            continue

        # Only count each character once per transcript segment.
        visible_characters = set()

        for scene_id in scene_ids:
            visible_characters.update(
                scene_characters.get(str(scene_id), set())
            )

        for character in visible_characters:

            item = evidence[speaker][character]

            item["segment_count"] += 1
            item["dialogue_seconds"] += duration
            item["scene_ids"].update(
                str(scene_id) for scene_id in scene_ids
            )

            confidence = segment.get(
                "speaker_confidence", "none"
            )

            if confidence == "strong":
                item["strong_segments"] += 1

            elif confidence == "moderate":
                item["moderate_segments"] += 1

            elif confidence == "weak":
                item["weak_segments"] += 1

    # -----------------------------------------------------
    # RANK CANDIDATES
    # -----------------------------------------------------

    output = {
        "source_timeline": str(TIMELINE_FILE),
        "method": (
            "Scene co-occurrence between diarized speakers "
            "and visible face clusters. This does not prove "
            "that a character is speaking."
        ),
        "speaker_count": len(speaker_totals),
        "speakers": [],
    }

    csv_rows = []

    for speaker, characters in evidence.items():

        candidates = []

        for character, item in characters.items():

            scene_count = len(item["scene_ids"])

            dialogue_seconds = round(
                item["dialogue_seconds"], 3
            )

            # Ranking score is only a sorting heuristic.
            # It is NOT a probability of identity.
            score = (
                item["strong_segments"] * 3
                + item["moderate_segments"] * 2
                + item["weak_segments"]
            )

            candidate = {
                "character_id": character,
                "face_count": cluster_sizes.get(character, 0),
                "segment_count": item["segment_count"],
                "dialogue_seconds": dialogue_seconds,
                "scene_count": scene_count,
                "strong_segments": item["strong_segments"],
                "moderate_segments": item["moderate_segments"],
                "weak_segments": item["weak_segments"],
                "ranking_score": score,
                "scene_ids": sorted(item["scene_ids"]),
                "status": "candidate_only",
            }

            candidates.append(candidate)

            csv_rows.append({
                "speaker_id": speaker,
                "character_id": character,
                "face_count": candidate["face_count"],
                "segment_count": candidate["segment_count"],
                "dialogue_seconds": dialogue_seconds,
                "scene_count": scene_count,
                "strong_segments": item["strong_segments"],
                "moderate_segments": item["moderate_segments"],
                "weak_segments": item["weak_segments"],
                "ranking_score": score,
                "status": "candidate_only",
            })

        candidates.sort(
            key=lambda x: (
                x["ranking_score"],
                x["scene_count"],
                x["dialogue_seconds"],
            ),
            reverse=True,
        )

        output["speakers"].append({
            "speaker_id": speaker,
            "total_diarized_dialogue_seconds": round(
                speaker_totals[speaker], 3
            ),
            "candidate_count": len(candidates),
            "candidates": candidates[:20],
        })

    output["speakers"].sort(
        key=lambda x: x["speaker_id"]
    )

    # -----------------------------------------------------
    # SAVE JSON
    # -----------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # -----------------------------------------------------
    # SAVE CSV
    # -----------------------------------------------------

    fields = [
        "speaker_id",
        "character_id",
        "face_count",
        "segment_count",
        "dialogue_seconds",
        "scene_count",
        "strong_segments",
        "moderate_segments",
        "weak_segments",
        "ranking_score",
        "status",
    ]

    with open(
        CSV_OUTPUT,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(csv_rows)

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("SPEAKER-CHARACTER CANDIDATES GENERATED")
    print("=" * 65)

    print("Speakers:", len(output["speakers"]))
    print("Candidate relationships:", len(csv_rows))

    for speaker_data in output["speakers"]:

        print("\n", speaker_data["speaker_id"])

        for candidate in speaker_data["candidates"][:5]:

            print(
                f"  {candidate['character_id']}: "
                f"score={candidate['ranking_score']}, "
                f"scenes={candidate['scene_count']}, "
                f"segments={candidate['segment_count']}"
            )

    print("\nOutput files:")
    print(OUTPUT_FILE)
    print(CSV_OUTPUT)


if __name__ == "__main__":
    main()

