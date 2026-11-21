import csv
import os

from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

AUDIO = "audio/diarization_test.wav"
OUTPUT_DIR = "diarization"

MODEL = "pyannote/speaker-diarization-community-1"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading speaker diarization model...")

pipeline = Pipeline.from_pretrained(MODEL)

if pipeline is None:
    raise RuntimeError("Could not load diarization model.")

print("Model loaded.")
print("Processing:", AUDIO)

with ProgressHook() as hook:
    output = pipeline(AUDIO, hook=hook)

turns = []

for turn, speaker in output.speaker_diarization:
    turns.append({
        "start": round(turn.start, 3),
        "end": round(turn.end, 3),
        "duration": round(turn.end - turn.start, 3),
        "speaker_id": str(speaker)
    })

output_csv = os.path.join(
    OUTPUT_DIR,
    "speaker_turns_test.csv"
)

with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "start",
            "end",
            "duration",
            "speaker_id"
        ]
    )

    writer.writeheader()
    writer.writerows(turns)

speakers = sorted(set(t["speaker_id"] for t in turns))

print("\nDiarization completed.")
print("Audio duration:", 300, "seconds")
print("Speaker turns:", len(turns))
print("Unique speakers:", len(speakers))
print("Speakers:", speakers)
print("Output:", output_csv)

print("\nFirst 20 speaker turns:")

for turn in turns[:20]:
    print(
        f"{turn['start']:8.2f} - "
        f"{turn['end']:8.2f} | "
        f"{turn['speaker_id']}"
    )
