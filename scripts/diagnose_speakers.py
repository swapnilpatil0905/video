"""Read-only report: how diarized speakers map to characters, and how much speech per character is visible
in the per-line active-speaker results. Reads the artifacts the pipeline itself would use; writes nothing.

    /workspace/video/.venv-platform/bin/python scripts/diagnose_speakers.py /workspace/video/projects/project_001
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


_CTX = None


def _platform(project: Path):
    """The pipeline's own context, so we read exactly the artifacts the next run will use."""
    global _CTX
    if _CTX is None:
        import argparse

        from video_platform.cli import build_context

        _CTX = build_context(argparse.Namespace(project=str(project / "config.yaml"), overrides=[], email=None))
    return _CTX


def latest(project: Path, stage: str, name: str):
    try:
        ctx = _platform(project)
    except Exception as exc:  # noqa: BLE001 - plain python3 without the platform installed
        sys.exit(f"run this with the platform python (/workspace/video/.venv-platform/bin/python): {exc}")
    artifact_id = ctx.planner.expected_ids()[stage]
    record = ctx.index.find_valid(artifact_id)
    dirs = sorted(p.name for p in (project / "artifacts" / stage).glob("art_*"))
    print(f"[{stage}] using {artifact_id} ({'valid' if record else 'NOT BUILT'}); "
          f"{len(dirs)} folder(s) on disk{': ' + ', '.join(d[:12] for d in dirs) if len(dirs) > 1 else ''}")
    if record is None:
        return None
    ref = next((r for r in record.outputs if r.uri and r.uri.endswith("/" + name)), None)
    if ref is None:
        sys.exit(f"{stage}: no {name} in {artifact_id}")
    return json.loads(ctx.store.read_bytes(ctx.store.key_from_uri(ref.uri)))


def main() -> None:
    project = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/video/projects/project_001")
    segments = latest(project, "alignment", "audio_segments.json")
    asd = latest(project, "active_speaker", "active_speaker.json")
    mapping = latest(project, "speaker_mapping", "speaker_mapping.json")
    profiles = {c["character_id"]: c for c in latest(project, "character_profile", "character_profiles.json") or []}
    database = {c["character_id"]: c for c in latest(project, "character_database", "source_characters.json") or []}
    if segments is None or asd is None or mapping is None:
        sys.exit("an analysis stage is not built for the current settings")

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
    print(f"{'character':14} {'tier':20} {'db faces':>8} {'faces':>6} {'map s':>7} {'asd clear s':>11} {'asd any s':>9}")
    for cid in ids:
        p = profiles.get(cid, {})
        print(f"{cid:14} {str(p.get('tier')):20} {database.get(cid, {}).get('face_count', ''):>8} {p.get('face_count', ''):>6} {via_map[cid]:7.0f} {via_asd[cid]:11.0f} {asd_any[cid]:9.0f}")

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
