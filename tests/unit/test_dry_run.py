"""End-to-end dry run of every LLM / image / voice stage on a fabricated project (simulated models).

Catches code and data-shape errors in narrative → character_design → character_identity →
scene_analysis → localization → character_references → voice_design → sample_selection without a GPU.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import yaml

from video_platform.cli import build_context, build_parser
from video_platform.orchestration.dryrun import dry_run
from video_platform.schemas.common import ArtifactRecord, ArtifactRef

SIMULATED = {"narrative", "character_design", "character_identity", "scene_analysis", "localization",
             "character_references", "voice_design", "sample_selection"}


def fake_upstream(n_lines: int = 40) -> dict[str, dict[str, object]]:
    rng = np.random.default_rng(0)
    segs, t = [], 1.0
    for i in range(n_lines):
        dur = 1.5 + (i % 4) * 0.5
        segs.append({"segment_id": f"SEG_{i:06d}", "start_s": round(t, 2), "end_s": round(t + dur, 2),
                     "text": "你为什么来这里?", "speaker_id": f"SPK_0{i % 3}", "language": "zh"})
        t += dur + (8.0 if i % 9 == 8 else 0.6)  # a long pause every 9 lines → several scenes
    chars = ["SRC_CHAR_0001", "SRC_CHAR_0002", "SRC_CHAR_0003"]
    faces = [f"FACE_{i:06d}" for i in range(30)]
    emb = rng.normal(size=(30, 512)).astype("float32")
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    shots = [{"shot_id": f"SH_{i:05d}", "start_s": float(i * 6), "end_s": float(i * 6 + 6)} for i in range(int(t // 6) + 2)]
    return {
        "alignment": {"audio_segments.json": segs},
        "emotion": {"emotions.json": {s["segment_id"]: {"emotion": ["sad", "angry", "neutral", "happy"][i % 4]}
                                      for i, s in enumerate(segs)}},
        "speaker_mapping": {"speaker_mapping.json": [
            {"speaker_id": f"SPK_0{k}", "character_id": chars[k], "status": "mapped", "supporting_speech_s": 20.0,
             "score": 0.8} for k in range(3)]},
        "character_profile": {"character_profiles.json": [
            {"character_id": c, "tier": ["main", "main", "supporting"][k], "score": 1.0 - 0.2 * k,
             "gender": ["male", "female", "male"][k], "age_estimate": 30, "face_count": 10, "speech_s": 20.0,
             "shot_count": 5} for k, c in enumerate(chars)]},
        "face_embedding": {"embeddings.npz": {"face_ids": np.array(faces), "embeddings": emb}},
        "face_clustering": {"clusters.json": {f: f"CHAR_000{1 + i % 3}" for i, f in enumerate(faces)}},
        "shot_detection": {"shots.json": shots},
        "active_speaker": {"active_speaker.json": {s["segment_id"]: {"speaking_character": chars[i % 3]}
                                                   for i, s in enumerate(segs)}},
        "ingest": {"video_metadata.json": {"duration_s": t, "width": 720, "height": 1280, "fps": 30}},
    }


def test_dry_run_all_story_and_asset_stages(tmp_path, monkeypatch, repo_root):
    monkeypatch.setenv("VP_PLATFORM_ROOT", str(repo_root))
    monkeypatch.setenv("VP_CAST_DIR", str(tmp_path / "no_cast"))
    project = tmp_path / "projects" / "dry"
    (project / "input").mkdir(parents=True)
    (project / "input" / "source.mp4").write_bytes(b"fake video")
    cfg = yaml.safe_load((repo_root / "projects" / "project_001" / "config.yaml").read_text(encoding="utf-8"))
    cfg["project"]["id"] = "dry"
    cfg.setdefault("production", {}).setdefault("character_images", {})["cast_folders"] = {"enabled": False}
    # a consenting cast member from a solo photo, to exercise the blend path
    photos = []
    for i in range(4):  # several photos of one person, like the real cast folders
        photo = tmp_path / f"me_{i}.jpg"
        photo.write_bytes(b"jpeg" + bytes([i]))
        photos.append({"path": str(photo), "face": "auto"})
    cfg["production"]["character_images"]["cast"] = [{
        "name": "me", "character": "lead_male", "photos": photos,
        "consent": {"given_by": "self", "date": "2026-11-22"}}]
    (project / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    args = build_parser().parse_args(["--project", str(project / "config.yaml")])
    ctx = build_context(args)
    expected = ctx.planner.expected_ids()
    upstream = fake_upstream()
    ancestors: set[str] = set()

    def collect(name):
        for dep in ctx.graph.get(name).depends_on:
            if dep not in ancestors:
                ancestors.add(dep)
                collect(dep)
    for target in ("sample_selection", "character_references", "voice_design"):
        collect(target)
    for name in ctx.graph.order:
        if name not in ancestors or name in SIMULATED:
            continue
        outputs = []
        for fname, payload in upstream.get(name, {}).items():
            key = f"{name}/{expected[name]}/{fname}"
            if fname.endswith(".npz"):
                path = tmp_path / fname
                np.savez(path, **payload)
                uri = ctx.store.put_file(path, key)
            else:
                uri = ctx.store.put_bytes(json.dumps(payload, ensure_ascii=False).encode(), key)
            outputs.append(ArtifactRef(artifact_id=expected[name], uri=uri, role="output"))
        ctx.index.put(ArtifactRecord(project_id="dry", stage=name, stage_version=ctx.graph.get(name).version,
                                     artifact_id=expected[name], params_hash="x", outputs=outputs))

    result = dry_run(ctx)
    assert set(result["executed"]) == SIMULATED, result
    assert result["worker_calls"]["llm"] >= 7  # narrative x2, design, identity, scenes, writer, naturalness, review, rewrite
    # the real index and store were not touched
    assert ctx.index.find_valid(expected["localization"]) is None
    scratch = Path(result["scratch"])
    lines = json.loads((scratch / "artifacts" / "localization" / expected["localization"] / "localized_lines.json").read_text())
    assert len(lines) == 40 and all(ln["status"] == "ok" for ln in lines)
    assert any(ln["versions"] > 1 for ln in lines)  # rewrite / naturalness paths ran
    assert all(ln["direction"]["emotion"] for ln in lines)
    quality = json.loads((scratch / "artifacts" / "localization" / expected["localization"] / "screenplay_quality.json").read_text())
    assert quality["scenes"] and quality["overall"] is not None
    identity = json.loads((scratch / "artifacts" / "character_identity" / expected["character_identity"] /
                           "character_identity.json").read_text())
    assert len(identity["characters"]) == 3
    refs = json.loads((scratch / "artifacts" / "character_references" / expected["character_references"] /
                       "character_references.json").read_text())
    cast_refs = [r for r in refs.values() if r.get("cast")]
    assert cast_refs and 0.30 <= cast_refs[0]["cast"]["similarity"] <= 0.55  # blended lead chosen in range
    assert all(r["variants"] for r in refs.values()), {g: (r["variant_identity"], r["chosen"]) for g, r in refs.items()}
    assert (scratch / "artifacts" / "character_references" / expected["character_references"] / "contact_sheet.jpg").exists()
    shutil.rmtree(scratch, ignore_errors=True)
