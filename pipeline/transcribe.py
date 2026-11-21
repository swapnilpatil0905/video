
import json
from pathlib import Path
import mlx_whisper

audio = "audio/source.wav"
output = Path("transcripts/chinese_transcript.json")

result = mlx_whisper.transcribe(
    audio,
    path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
    language="zh",
    condition_on_previous_text=False,
    verbose=True
)

output.parent.mkdir(parents=True, exist_ok=True)

with output.open("w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("Transcription completed.")
print("Segments:", len(result.get("segments", [])))
print("Saved:", output)

