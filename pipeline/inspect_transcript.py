import json

path = "transcripts/chinese_transcript.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

segments = data["segments"]

print("Total segments:", len(segments))

if segments:
    print("First timestamp:", segments[0]["start"])
    print("Last timestamp:", segments[-1]["end"])
    print("Total duration (minutes):", round(segments[-1]["end"] / 60, 2))

print("\nFirst 20 segments:")
for s in segments[:20]:
    print(f'{s["start"]:8.2f} - {s["end"]:8.2f} | {s["text"]}')

print("\nLast 10 segments:")
for s in segments[-10:]:
    print(f'{s["start"]:8.2f} - {s["end"]:8.2f} | {s["text"]}')

print("\nInspection completed.")
