
import csv
import json
from pathlib import Path

FILES = [
    Path("timeline/audit/unmatched_speaker_segments.csv"),
    Path("timeline/audit/weak_speaker_segments.csv"),
    Path("characters/character_scene_mapping.csv"),
    Path("faces/clustered_faces.csv"),
    Path("faces/cluster_summary.csv"),
]

QUALITY_REPORT = Path("timeline/timeline_quality_report.json")


def inspect_csv(path):
    print("\n" + "=" * 75)
    print("FILE:", path)
    print("=" * 75)

    if not path.exists():
        print("MISSING")
        return

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        print("Columns:", reader.fieldnames)

        for index, row in enumerate(reader):
            if index >= 5:
                break
            print(f"Sample {index + 1}:", row)


def main():
    for path in FILES:
        inspect_csv(path)

    print("\n" + "=" * 75)
    print("TIMELINE QUALITY REPORT")
    print("=" * 75)

    if QUALITY_REPORT.exists():
        with open(QUALITY_REPORT, encoding="utf-8") as f:
            report = json.load(f)

        print("Invalid segments:")
        for item in report.get("invalid_segments", []):
            print(item)

        print("\nUnmatched speaker count:")
        print(report.get("unmatched_speaker_segments"))

        print("\nWeak speaker count:")
        print(report.get("weak_speaker_matches"))

    else:
        print("Quality report missing")


if __name__ == "__main__":
    main()

