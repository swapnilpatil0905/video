
import subprocess
from pathlib import Path

source = Path("input/source.mp4")
output = Path("audio/source.wav")

if not source.exists():
    raise FileNotFoundError(source)

output.parent.mkdir(parents=True, exist_ok=True)

subprocess.run([
    "ffmpeg", "-y",
    "-i", str(source),
    "-map", "0:a:0",
    "-vn",
    "-ac", "1",
    "-ar", "16000",
    "-c:a", "pcm_s16le",
    str(output)
], check=True)

print("Audio extracted:", output)

