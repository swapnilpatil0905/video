
import csv
import json
from pathlib import Path

FILES = {
    "Chinese transcript": Path("transcripts/chinese_transcript_clean.json"),
    "Diarization": Path("diarization/speaker_turns.json"),
    "Scene timeline": Path("scenes/scene_timeline.csv"),
}

for name, path in FILES.items():
    print("\n" + "=" * 65)
    print(name)
    print("=" * 65)

    if not path.exists():
        print("MISSING:", path)
        continue

    print("File:", path)
    print("Size:", round(path.stat().st_size / 1024, 2), "KB")

    if path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            print("Top-level keys:", list(data.keys()))

            for key, value in data.items():
                if isinstance(value, list):
                    print(f"Array: {key}, records: {len(value)}")
                    if value:
                        print("First record:", value[0])
                    break

        elif isinstance(data, list):
            print("Records:", len(data))
            if data:
                print("First record:", data[0])

    elif path.suffix == ".csv":
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            print("Columns:", reader.fieldnames)
            for row in list(reader)[:3]:
                print("Sample:", row)

