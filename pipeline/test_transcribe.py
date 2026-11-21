
import mlx_whisper
import json

result = mlx_whisper.transcribe(
    "audio/test_2min.wav",
    path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
    language="zh",
    condition_on_previous_text=False,
    verbose=True
)

with open(
    "transcripts/test_2min.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

for segment in result["segments"]:
    print(
        f'{segment["start"]:.2f} - '
        f'{segment["end"]:.2f}: '
        f'{segment["text"]}'
    )

print("Test transcription completed.")

