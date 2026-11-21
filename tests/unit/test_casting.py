import numpy as np

from video_platform.storytelling import casting


def unit(v):
    v = np.asarray(v, dtype="float32")
    return (v / np.linalg.norm(v)).tolist()


def mix(a, b, t):
    """Unit vector at cosine ≈ t from a, towards b (a ⟂ b)."""
    return unit(np.asarray(a) * t + np.asarray(b) * np.sqrt(1 - t * t))


A, B, C = unit([1, 0, 0, 0]), unit([0, 1, 0, 0]), unit([0, 0, 1, 0])
DESIGNS = [{"gen_character_id": "G1", "source_character_id": "S1", "tier": "supporting", "gender": "male"},
           {"gen_character_id": "G2", "source_character_id": "S2", "tier": "main", "gender": "male"},
           {"gen_character_id": "G3", "source_character_id": "S3", "tier": "main", "gender": "female"}]


def test_resolve_leads_by_importance():
    cast = [{"name": "me", "character": "lead_male"}, {"name": "wife", "character": "lead_female"},
            {"name": "x", "character": "S1"}]
    t = casting.resolve_cast_targets(cast, DESIGNS)
    assert t["G2"]["name"] == "me" and t["G3"]["name"] == "wife" and t["G1"]["name"] == "x"


def test_blend_scoring_range():
    member = {"name": "me", "mode": "blend", "similarity": (0.30, 0.55)}
    base = {"det_score": 0.9, "face_ratio": 0.3}
    ok = casting.score_candidate({**base, "embedding": mix(A, B, 0.42)}, source=C, cast_emb=A, cast_member=member, other_cast={})
    same = casting.score_candidate({**base, "embedding": mix(A, B, 0.8)}, source=C, cast_emb=A, cast_member=member, other_cast={})
    far = casting.score_candidate({**base, "embedding": mix(A, B, 0.1)}, source=C, cast_emb=A, cast_member=member, other_cast={})
    assert ok["valid"] and not same["valid"] and not far["valid"]
    assert any("same person" in r for r in same["reasons"]) and any("too weak" in r for r in far["reasons"])
    assert casting.next_strength([far], member, "clear") == "strong"
    assert casting.next_strength([same], member, "clear") == "subtle"


def test_synthetic_must_not_look_like_cast_or_actor():
    base = {"det_score": 0.9, "face_ratio": 0.3}
    looks_like_me = casting.score_candidate({**base, "embedding": mix(A, B, 0.5)}, source=C, cast_emb=None,
                                            cast_member=None, other_cast={"me": A})
    looks_like_actor = casting.score_candidate({**base, "embedding": mix(C, B, 0.6)}, source=C, cast_emb=None,
                                               cast_member=None, other_cast={})
    fine = casting.score_candidate({**base, "embedding": B}, source=C, cast_emb=None, cast_member=None, other_cast={"me": A})
    assert not looks_like_me["valid"] and not looks_like_actor["valid"] and fine["valid"]
    assert casting.choose([looks_like_me, fine]) == 1 and casting.choose([looks_like_me, fine], pick=0) == 0
    assert casting.choose([looks_like_me]) is None


def test_prompts_never_contain_names():
    p = casting.face_prompt({"image_prompt": "Portrait of a man", "gender": "male", "age": 30},
                            {"full_name": "Arjun Reddy", "profession": "software engineer", "age": 31}, "Hyderabad", "real photo")
    assert "Arjun" not in p and "Hyderabad" in p and "software engineer" in p
    assert "NOT be the same person" in casting.blend_prompt(2, "blend")
    assert "exact face" in casting.blend_prompt(1, "as_is")


def test_auto_face_in_group_photo_and_best_photos():
    me, wife = A, B
    photos = {"solo": [{"embedding": mix(me, C, 0.9), "face_ratio": 0.3, "det_score": 0.9}],
              "couple": [{"embedding": wife, "face_ratio": 0.3, "det_score": 0.9},
                         {"embedding": mix(me, C, 0.85), "face_ratio": 0.25, "det_score": 0.9}]}
    picked = casting.resolve_auto_faces(photos)
    assert picked["couple"]["embedding"] == photos["couple"][1]["embedding"]
    # no solo photo: the face present in every photo wins
    both = {"p1": [{"embedding": wife, "face_ratio": 0.3, "det_score": 0.9}, {"embedding": me, "face_ratio": 0.3, "det_score": 0.9}],
            "p2": [{"embedding": mix(me, C, 0.9), "face_ratio": 0.3, "det_score": 0.9}, {"embedding": C, "face_ratio": 0.3, "det_score": 0.9}],
            "p3": [{"embedding": mix(me, B, 0.9), "face_ratio": 0.3, "det_score": 0.9}]}
    assert casting.resolve_auto_faces(both)["p1"]["embedding"] == me
    metrics = {"solo": {"embedding": me, "det_score": 0.8, "face_ratio": 0.2, "face_count": 1},
               "couple": {"embedding": me, "det_score": 0.99, "face_ratio": 0.4, "face_count": 2}}
    assert casting.best_photos(["couple", "solo"], metrics, 1) == ["solo"]


def test_appearance_in_prompts():
    look = {"build": "slim, fit build", "look": "good-looking", "blend_note": "slimmer face"}
    p = casting.face_prompt({"image_prompt": "Portrait", "gender": "female", "age": 28}, None, "Hyderabad", "real photo", look)
    assert "slim, fit build" in p and "good-looking" in p
    assert "slimmer face" in casting.blend_prompt(2, "blend", appearance=look)
