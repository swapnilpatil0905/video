from pathlib import Path

from video_platform.config.loader import fingerprint_cast_photos, merge_global_cast


def test_global_cast_merged_with_absolute_paths(tmp_path: Path):
    cast_dir = tmp_path / "cast"
    cast_dir.mkdir()
    (cast_dir / "me.jpg").write_bytes(b"jpeg")
    (cast_dir / "cast.yaml").write_text(
        "cast:\n  - name: me\n    character: lead_male\n    photos: [{path: me.jpg, face: right}]\n"
        "    consent: {given_by: self, date: '2026-11-21'}\n", encoding="utf-8")
    merged = merge_global_cast({"production": {"character_images": {}}}, tmp_path, {})
    member = merged["production"]["character_images"]["cast"][0]
    assert member["name"] == "me" and Path(member["photos"][0]["path"]).is_absolute()
    fingerprint_cast_photos(merged, tmp_path)
    assert len(member["photos"][0]["sha256"]) == 64
    off = merge_global_cast({"production": {"character_images": {"use_global_cast": False}}}, tmp_path, {})
    assert "cast" not in off["production"]["character_images"]
