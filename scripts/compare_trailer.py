"""Side-by-side review video: original footage (left) vs. the remade trailer (right), clip by clip.

Uses the trailer's own timeline (src_start/src_end of every clip), so each remade shot is shown next to
the exact source shot it replaced. Each clip is labelled with its id, character, identity score and
lip-sync status. Writes two files: one with the Hindi remake audio, one with the original audio.

    python3 scripts/compare_trailer.py input/source.mp4 <sample>/high/trailer_timeline.json \
        <sample>/high/trailer.mp4 ~/Downloads/trailer_review
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

def _has_filter(name: str) -> bool:
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    except OSError:
        return False
    return any(line.split()[1:2] == [name] for line in out.splitlines() if len(line.split()) > 1)


def _srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


FONT = next((f for f in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc",
                         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf") if Path(f).exists()), None)


def _text(s: str) -> str:
    return s.replace("\\", "\\\\").replace(":", "\\:").replace("'", "’").replace("%", "\\%")


def main() -> int:
    if len(sys.argv) != 5:
        print(__doc__)
        return 2
    source, timeline_path, trailer, out_dir = sys.argv[1:]
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    timeline = json.loads(Path(timeline_path).read_text(encoding="utf-8"))
    w, h, fps = 720, 1280, 24

    parts, labels = [], []
    for i, c in enumerate(timeline):
        a, b = float(c["src_start"]), float(c["src_end"])
        parts.append(f"[0:v]trim=start={a}:end={b},setpts=PTS-STARTPTS,scale={w}:{h},fps={fps},setsar=1[v{i}];")
        ident = c.get("identity_cos")
        info = (f"{c['clip_id']}  {c.get('character') or '-'}  id {ident:.2f}" if isinstance(ident, (int, float))
                else f"{c['clip_id']}  {c.get('character') or '-'}")
        info += f"  lipsync {c.get('lipsync', '-')}"
        labels.append((float(c["out_start"]), float(c["out_end"]), info, c.get("status", "")))
    n = len(timeline)
    graph = "".join(parts) + "".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[src];"
    graph += f"[1:v]scale={w}:{h},fps={fps},setsar=1[rem];[src][rem]hstack=inputs=2[sbs]"
    labels_srt = out / "compare_labels.srt"
    labels_srt.write_text("".join(
        f"{k + 1}\n{_srt_time(st)} --> {_srt_time(en)}\nLEFT original | RIGHT remake — {info}"
        f"{'  ⚠ low identity' if status != 'generated' else ''}\n\n"
        for k, (st, en, info, status) in enumerate(labels)), encoding="utf-8")
    burn = bool(FONT) and _has_filter("drawtext")
    if burn:
        draw = [f"drawtext=fontfile='{FONT}':text='ORIGINAL':x=20:y=20:fontsize=40:fontcolor=white:box=1:boxcolor=black@0.5",
                f"drawtext=fontfile='{FONT}':text='REMAKE':x={w + 20}:y=20:fontsize=40:fontcolor=white:box=1:boxcolor=black@0.5"]
        for start, end, info, status in labels:
            colour = "yellow" if status != "generated" else "white"
            draw.append(f"drawtext=fontfile='{FONT}':text='{_text(info)}':x=20:y=h-60:fontsize=26:fontcolor={colour}:"
                        f"box=1:boxcolor=black@0.6:enable='between(t,{start:.3f},{end:.3f})'")
        graph += "," + ",".join(draw)
    graph += "[out]" if burn else ""
    if not burn:
        graph = graph.replace("[sbs]", "[out]")

    remake_audio = out / "compare_side_by_side_hindi_audio.mp4"
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", source, "-i", trailer, "-i", str(labels_srt), "-filter_complex", graph,
           "-map", "[out]", "-map", "1:a", "-map", "2:s", "-c:v", "libx264", "-crf", "20", "-preset", "veryfast",
           "-c:a", "aac", "-b:a", "160k", "-c:s", "mov_text", "-metadata:s:s:0", "language=eng", "-shortest",
           str(remake_audio)]
    subprocess.run(cmd, check=True)
    # same picture, original Chinese audio (reuse the encoded video, re-cut the source audio only)
    original_audio = out / "compare_side_by_side_original_audio.mp4"
    agraph = "".join(f"[0:a]atrim=start={c['src_start']}:end={c['src_end']},asetpts=PTS-STARTPTS[a{i}];"
                     for i, c in enumerate(timeline)) + "".join(f"[a{i}]" for i in range(n)) + f"concat=n={n}:v=0:a=1[oa]"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", source, "-i", str(remake_audio), "-filter_complex", agraph,
                    "-map", "1:v", "-map", "[oa]", "-map", "1:s", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
                    "-c:s", "copy", "-shortest", str(original_audio)], check=True)
    print(remake_audio)
    print(original_audio)
    return 0


if __name__ == "__main__":
    sys.exit(main())
