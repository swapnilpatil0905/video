import os

import pytest

from video_platform.schemas.common import ArtifactRecord
from video_platform.storage.index import ArtifactIndex
from video_platform.storage.store import LocalArtifactStore, S3ArtifactStore, normalize_key
from video_platform.core.errors import PolicyViolation


@pytest.mark.parametrize("bad", ["", "/abs", "../up", "a/../b", "a//b", "a\\b", "./a"])
def test_bad_keys_rejected(bad):
    with pytest.raises(ValueError):
        normalize_key(bad)


def test_local_store_hardlink_and_bytes(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    source = tmp_path / "src.bin"
    source.write_bytes(b"data")
    uri = store.put_file(source, "stage/art_1/legacy/src.bin", link=True)
    assert uri == "local://stage/art_1/legacy/src.bin"
    stored = store.path_for("stage/art_1/legacy/src.bin")
    assert os.stat(stored).st_ino == os.stat(source).st_ino  # hardlinked, no copy
    assert store.put_file(source, "stage/art_1/legacy/src.bin", link=True) == uri  # idempotent
    store.put_bytes(b"{}", "stage/art_1/out.json")
    assert store.read_bytes("stage/art_1/out.json") == b"{}"
    assert store.exists("stage/art_1/out.json")
    assert source.read_bytes() == b"data"  # original untouched


def test_s3_store_with_fake_client(tmp_path):
    calls = {}

    class FakeClient:
        def upload_file(self, filename, bucket, key):
            calls["upload"] = (bucket, key)

        def put_object(self, Bucket, Key, Body):
            calls["put"] = (Bucket, Key, Body)

    store = S3ArtifactStore(endpoint="https://media.example.org", bucket="vp", prefix="projects/p1",
                            client_factory=lambda *a, **k: FakeClient())
    (tmp_path / "f").write_bytes(b"x")
    assert store.put_file(tmp_path / "f", "s/a/f") == "s3://vp/projects/p1/s/a/f"
    assert calls["upload"] == ("vp", "projects/p1/s/a/f")
    with pytest.raises(PolicyViolation):
        S3ArtifactStore(endpoint="https://s3.corp.example", bucket="b", forbidden_hosts=["*.corp.example"],
                        client_factory=lambda *a, **k: FakeClient())


def _record(artifact_id="art_x", stage="asr", status="valid"):
    return ArtifactRecord(project_id="p", stage=stage, stage_version="1", artifact_id=artifact_id,
                          params_hash="h", status=status)


def test_index_roundtrip_and_summary(tmp_path):
    index = ArtifactIndex(tmp_path / "state" / "a.sqlite")
    index.put(_record())
    index.put(_record("art_y", status="quarantined"))
    assert index.find_valid("art_x") is not None
    assert index.find_valid("art_y") is None
    assert index.summary("p") == {"asr": {"valid": 1, "quarantined": 1}}
    index.set_status("art_x", "superseded")
    assert index.find_valid("art_x") is None
    assert len(index.list_records("p", "asr")) == 2


def test_file_hash_cache(tmp_path):
    index = ArtifactIndex(tmp_path / "a.sqlite")
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc")
    first = index.file_sha256(path)
    assert first == index.file_sha256(path)
    path.write_bytes(b"abcd")  # size changes → re-hash
    assert index.file_sha256(path) != first
