"""Read-only report: how diarized speakers map to characters, and how much speech per character is visible
in the per-line active-speaker results. Uses only the Python standard library; writes nothing.

    python3 scripts/diagnose_speakers.py /workspace/video/projects/project_001
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def latest(project: Path, stage: str, name: str):
    files = sorted((project / "artifacts" / stage).glob(f"*/{name}"), key=lambda p: p.stat().st_mtime)
    if not files:
        sys.exit(f"missing {stage}/{name} under {project / 'artifacts'}")
    return json.loads(files[-1].read_text(encoding="utf-8"))


def main() -> None:
    project = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/video/projects/project_001")
    segments = latest(project, "alignment", "audio_segments.json")
    asd = latest(project, "active_speaker", "active_speaker.json")
    mapping = latest(project, "speaker_mapping", "speaker_mapping.json")
    profiles = {c["character_id"]: c for c in latest(project, "character_profile", "character_profiles.json")} \
        if list((project / "artifacts" / "character_profile").glob("*/character_profiles.json")) else {}

    dur = {s["segment_id"]: s["end_s"] - s["start_s"] for s in segments}
    print(f"segments: {len(segments)}, speech {sum(dur.values()) / 60:.1f} min, "
          f"no speaker id: {sum(not s.get('speaker_id') for s in segments)}")
    print("active-speaker decisions:", dict(Counter(d["decision"] for d in asd.values())))

    print("\n== speakers (voice clusters) → best character")
    for m in sorted(mapping, key=lambda m: -m["supporting_speech_s"]):
        votes = next((e["detail"] for e in m.get("evidence", []) if e.get("kind") == "asd"), {})
        others = sorted(((c["detail"]["character_id"], c["weight"]) for c in m.get("contradictions", [])), key=lambda x: -x[1])[:4]
        print(f"{m['speaker_id']}: {m['supporting_speech_s'] / 60:5.1f} min → {m['character_id']} [{m['status']}] "
              f"conf={m.get('confidence')} votes={votes.get('votes')}/{votes.get('total')} also: {others}")

    print("\n== per character: speech via speaker mapping vs. lines where they are clearly the visible speaker")
    via_map, via_asd, asd_any = defaultdict(float), defaultdict(float), defaultdict(float)
    spk_char = {m["speaker_id"]: m["character_id"] for m in mapping}
    for s in segments:
        cid = spk_char.get(s.get("speaker_id") or "")
        if cid:
            via_map[cid] += dur[s["segment_id"]]
        d = asd.get(s["segment_id"]) or {}
        who = d.get("speaking_character")
        if who and d.get("decision") == "asd_margin":
            via_asd[who] += dur[s["segment_id"]]
        if who:
            asd_any[who] += dur[s["segment_id"]]
    ids = sorted(set(via_map) | set(via_asd) | set(profiles), key=lambda c: -(profiles.get(c, {}).get("score") or 0))
    print(f"{'character':14} {'tier':20} {'faces':>6} {'map s':>7} {'asd clear s':>11} {'asd any s':>9}")
    for cid in ids:
        p = profiles.get(cid, {})
        print(f"{cid:14} {str(p.get('tier')):20} {p.get('face_count', ''):>6} {via_map[cid]:7.0f} {via_asd[cid]:11.0f} {asd_any[cid]:9.0f}")

    print("\n== top speakers: which characters the active-speaker check sees in their lines")
    by_spk = defaultdict(Counter)
    for s in segments:
        who = (asd.get(s["segment_id"]) or {}).get("speaking_character")
        if who and s.get("speaker_id"):
            by_spk[s["speaker_id"]][who] += round(dur[s["segment_id"]])
    for spk, c in sorted(by_spk.items(), key=lambda x: -sum(x[1].values()))[:9]:
        print(f"{spk}: {dict(c.most_common(6))} (seconds)")


if __name__ == "__main__":
    main()
