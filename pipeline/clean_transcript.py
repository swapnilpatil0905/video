import json
import re

INPUT = "transcripts/chinese_transcript.json"
OUTPUT = "transcripts/chinese_transcript_clean.json"

WATERMARKS = [
    "优优独播剧场",
    "YoYo Television Series Exclusive",
]

with open(INPUT, "r", encoding="utf-8") as f:
    data = json.load(f)

original_count = len(data["segments"])

cleaned_segments = []

for segment in data["segments"]:
    text = segment.get("text", "").strip()

    # Remove broadcaster watermark segments
    if any(marker.lower() in text.lower() for marker in WATERMARKS):
        continue

    cleaned_segments.append(segment)

data["segments"] = cleaned_segments

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("Original segments:", original_count)
print("Remaining segments:", len(cleaned_segments))
print("Removed segments:", original_count - len(cleaned_segments))
print("Saved:", OUTPUT)
