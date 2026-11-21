from video_platform.core.hashing import canonical_json, compute_artifact_id, params_hash, sha256_file


def test_sha256_file(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc")
    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_canonical_json_is_order_independent():
    assert canonical_json({"b": 1, "a": [1, 2]}) == canonical_json({"a": [1, 2], "b": 1})
    assert params_hash({"x": 1, "y": "z"}) == params_hash({"y": "z", "x": 1})


def test_artifact_id_deterministic_and_sensitive():
    base = dict(stage="asr", stage_version="1", input_ids=["a", "b"], params_digest="p", model_fingerprint="m@1")
    first = compute_artifact_id(**base)
    assert first == compute_artifact_id(**{**base, "input_ids": ["b", "a"]})
    assert first.startswith("art_") and len(first) == 44
    for change in ({"stage_version": "2"}, {"params_digest": "q"}, {"model_fingerprint": "m@2"}, {"input_ids": ["a"]}):
        assert compute_artifact_id(**{**base, **change}) != first
