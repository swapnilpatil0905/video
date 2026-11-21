
import csv
import json
import os
import time
from pathlib import Path

from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook


# --------------------------------------------------
# Configuration
# --------------------------------------------------

AUDIO = Path("audio/source.wav")
OUTPUT_DIR = Path("diarization")

OUTPUT_CSV = OUTPUT_DIR / "speaker_turns.csv"
OUTPUT_JSON = OUTPUT_DIR / "speaker_turns.json"
SUMMARY_JSON = OUTPUT_DIR / "speaker_summary.json"

MODEL = "pyannote/speaker-diarization-community-1"

# Leave as None to let the model estimate speaker count.
# Do not force a speaker count without evidence.
MIN_SPEAKERS = None
MAX_SPEAKERS = None


# --------------------------------------------------
# Validation
# --------------------------------------------------

if not AUDIO.exists():
    raise FileNotFoundError(f"Audio file not found: {AUDIO}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

audio_size_mb = AUDIO.stat().st_size / (1024 * 1024)

print("=" * 60)
print("FULL VIDEO SPEAKER DIARIZATION")
print("=" * 60)
print(f"Audio: {AUDIO}")
print(f"Audio size: {audio_size_mb:.2f} MB")
print(f"Model: {MODEL}")


# --------------------------------------------------
# Load model
# --------------------------------------------------

print("\nLoading diarization model...")

pipeline = Pipeline.from_pretrained(MODEL)

if pipeline is None:
    raise RuntimeError("Failed to load diarization model.")

print("Model loaded successfully.")


# --------------------------------------------------
# Run diarization
# --------------------------------------------------

print("\nStarting full audio diarization...")
print("This may take considerable time.")

start_time = time.time()

pipeline_args = {}

if MIN_SPEAKERS is not None:
    pipeline_args["min_speakers"] = MIN_SPEAKERS

if MAX_SPEAKERS is not None:
    pipeline_args["max_speakers"] = MAX_SPEAKERS

with ProgressHook() as hook:
    output = pipeline(
        str(AUDIO),
        hook=hook,
        **pipeline_args
    )

elapsed_seconds = time.time() - start_time


# --------------------------------------------------
# Extract speaker turns
# --------------------------------------------------

turns = []

for turn, speaker in output.speaker_diarization:
    start = round(float(turn.start), 3)
    end = round(float(turn.end), 3)

    if end <= start:
        continue

    turns.append({
        "start": start,
        "end": end,
        "duration": round(end - start, 3),
        "speaker_id": str(speaker)
    })

turns.sort(key=lambda item: (item["start"], item["end"]))


# --------------------------------------------------
# Speaker statistics
# --------------------------------------------------

speaker_stats = {}

for item in turns:
    speaker = item["speaker_id"]

    if speaker not in speaker_stats:
        speaker_stats[speaker] = {
            "speaker_id": speaker,
            "turn_count": 0,
            "total_speech_seconds": 0.0,
            "first_seen": item["start"],
            "last_seen": item["end"]
        }

    stats = speaker_stats[speaker]

    stats["turn_count"] += 1
    stats["total_speech_seconds"] += item["duration"]
    stats["last_seen"] = max(stats["last_seen"], item["end"])

for stats in speaker_stats.values():
    stats["total_speech_seconds"] = round(
        stats["total_speech_seconds"], 3
    )

    stats["first_seen"] = round(stats["first_seen"], 3)
    stats["last_seen"] = round(stats["last_seen"], 3)


# --------------------------------------------------
# Save CSV
# --------------------------------------------------

csv_fields = [
    "start",
    "end",
    "duration",
    "speaker_id"
]

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=csv_fields)

    writer.writeheader()
    writer.writerows(turns)


# --------------------------------------------------
# Save JSON
# --------------------------------------------------

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(
        {
            "audio_file": str(AUDIO),
            "model": MODEL,
            "turn_count": len(turns),
            "speaker_count": len(speaker_stats),
            "turns": turns
        },
        f,
        ensure_ascii=False,
        indent=2
    )


# --------------------------------------------------
# Save speaker summary
# --------------------------------------------------

summary = {
    "audio_file": str(AUDIO),
    "model": MODEL,
    "processing_seconds": round(elapsed_seconds, 2),
    "turn_count": len(turns),
    "speaker_count": len(speaker_stats),
    "speakers": sorted(
        speaker_stats.values(),
        key=lambda item: item["total_speech_seconds"],
        reverse=True
    )
}

with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
    json.dump(
        summary,
        f,
        ensure_ascii=False,
        indent=2
    )


# --------------------------------------------------
# Final report
# --------------------------------------------------

print("\n" + "=" * 60)
print("DIARIZATION COMPLETED")
print("=" * 60)

print(f"Processing time: {elapsed_seconds / 60:.2f} minutes")
print(f"Speaker turns: {len(turns)}")
print(f"Unique speakers: {len(speaker_stats)}")

print("\nSpeaker statistics:")

for stats in summary["speakers"]:
    print(
        f"{stats['speaker_id']} | "
        f"Turns: {stats['turn_count']} | "
        f"Speech: {stats['total_speech_seconds']:.1f}s | "
        f"First: {stats['first_seen']:.1f}s | "
        f"Last: {stats['last_seen']:.1f}s"
    )

print("\nFirst 20 speaker turns:")

for item in turns[:20]:
    print(
        f"{item['start']:8.2f} - "
        f"{item['end']:8.2f} | "
        f"{item['speaker_id']}"
    )

print("\nOutput files:")
print(f"CSV:     {OUTPUT_CSV}")
print(f"JSON:    {OUTPUT_JSON}")
print(f"Summary: {SUMMARY_JSON}")
print("=" * 60)

