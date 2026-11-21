"""Realistic character faces: prompts, consenting-cast blends and candidate selection (no model calls here).

Flow per character (stage: character_references):
  1. prompt   = character design + Indian identity (age, region, profession, background) + realism rules
  2. N candidates from text-to-image (different seeds)
  3. cast characters only: each candidate is blended with the cast member's photos by a multi-image
     edit model into a NEW person ("blend"), or restyled while keeping their face ("as_is")
  4. every image is measured (ArcFace): face found, detection quality, face size, similarity to the
     original actor (must be low), to the cast member (blend: inside a range; as_is: high) and to the
     other cast members (must be low) — the best valid candidate wins, unless you pick one
  5. pose/expression sheet from the winner (identity-checked), plus a contact sheet for review
"""

from __future__ import annotations

from typing import Any

import numpy as np

TIER_ORDER = {"main": 0, "supporting": 1, "recurring_secondary": 2, "minor": 3}
MAX_SOURCE_SIMILARITY = 0.45     # vs the original actor
MAX_OTHER_CAST_SIMILARITY = 0.32  # a character must not look like a cast member it was not built from
MIN_DET_SCORE = 0.6
MIN_FACE_RATIO = 0.10


def resolve_cast_targets(cast: list[dict], designs: list[dict]) -> dict[str, dict]:
    """{gen_character_id: cast member}. lead_male / lead_female = most important character of that gender."""
    ordered = sorted(enumerate(designs), key=lambda x: (TIER_ORDER.get(x[1].get("tier"), 9), x[0]))
    out: dict[str, dict] = {}
    for member in cast:
        target = member["character"]
        if target in ("lead_male", "lead_female"):
            gender = target.split("_")[1]
            hit = next((d for _, d in ordered if str(d.get("gender", "")).lower().startswith(gender)
                        and d["gen_character_id"] not in out), None)
        else:
            hit = next((d for d in designs if target in (d["source_character_id"], d["gen_character_id"])), None)
        if hit is not None:
            out[hit["gen_character_id"]] = member
    return out


def face_prompt(design: dict, identity: dict | None, region: str, style_suffix: str,
                appearance: dict | None = None) -> str:
    """Design prompt + identity context (never the name: names can pull towards real people)."""
    parts = [design.get("image_prompt") or "Portrait of a fictional person"]
    who = identity or {}
    context = ", ".join(str(x) for x in (
        f"{who.get('age') or design.get('age')}-year-old {design.get('gender') or ''}".strip(),
        f"from {region}" if region else "", who.get("profession") or "", (who.get("background") or "")[:140])
        if x and x.strip())
    if context:
        parts.append(f"The person: {context}")
    if appearance:
        parts.append(f"Look: {appearance['look']}; body: {appearance['build']} (this overrides any other build description)")
        parts.append("A real person, not a celebrity")
    else:
        parts.append("An ordinary real person, not a celebrity or model")
    parts.append(style_suffix)
    return ". ".join(p.rstrip(". ") for p in parts if p) + "."


def blend_prompt(n_photos: int, mode: str, strength: str = "clear", glasses: bool = False,
                 appearance: dict | None = None) -> str:
    refs = "the first image" if n_photos == 1 else f"the first {n_photos} images"
    glasses_rule = "Glasses only if the last image has them." if not glasses else "Keep the glasses."
    if appearance:
        glasses_rule += (f" The new person is {appearance['look']}, with a {appearance['build']}"
                         + (f" ({appearance['blend_note']})" if appearance.get("blend_note") and mode != "as_is" else "") + ".")
    if mode == "as_is":
        return (f"Photorealistic portrait of the real person shown in {refs}: keep their exact face and identity. "
                "Use the pose, framing, clothing, hairstyle and lighting of the last image. "
                f"{glasses_rule} Natural skin texture, real photograph, no retouching, no beauty filter.")
    resemblance = {"subtle": "a subtle family resemblance (like a cousin)",
                   "clear": "a clear family resemblance (like a sibling)",
                   "strong": "a strong family resemblance (like a twin sibling, still a different person)"}[strength]
    return (f"Create ONE new, real-looking person. Keep the pose, framing, clothing, hairstyle and lighting of the "
            f"last image. Give the face {resemblance} to the person in {refs} — similar face shape, eyes, nose and "
            f"smile — but it must NOT be the same person. {glasses_rule} Natural skin texture with pores, slight "
            "asymmetry, real photograph, no retouching, no beauty filter.")


def resolve_auto_faces(photos: dict[str, list[dict]]) -> dict[str, dict | None]:
    """Pick the person's face in each photo: single-face photos anchor the identity; in group photos the
    face most similar to the anchor wins. Without an anchor, the face that best matches the faces in the
    person's other photos wins (the person is the one who appears in all of them)."""
    anchors = [faces[0]["embedding"] for faces in photos.values() if len(faces) == 1]
    anchor = mean_embedding(anchors)
    out: dict[str, dict | None] = {}
    for key, faces in photos.items():
        if not faces:
            out[key] = None
        elif len(faces) == 1:
            out[key] = faces[0]
        elif anchor is not None:
            out[key] = max(faces, key=lambda f: _cos(f["embedding"], anchor) or -1)
        else:
            others = [f["embedding"] for k, fs in photos.items() if k != key for f in fs]

            def support(face: dict) -> float:
                sims = sorted((_cos(face["embedding"], o) or -1 for o in others), reverse=True)
                return sum(sims[: max(1, len(photos) - 1)])
            out[key] = max(faces, key=support) if others else max(faces, key=lambda f: f["face_ratio"])
    return out


def best_photos(keys: list[str], metrics: dict[str, dict], n: int = 2) -> list[str]:
    """The n clearest photos of a person (detected face, large, confident) for the blend model."""
    usable = [k for k in keys if metrics.get(k, {}).get("embedding") is not None]
    usable.sort(key=lambda k: (int((metrics[k].get("face_count") or 1) > 1),  # solo photos first (no second person)
                               -(float(metrics[k].get("det_score") or 0) + float(metrics[k].get("face_ratio") or 0))))
    return usable[:n] or keys[:n]


def _cos(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    return float(np.dot(np.asarray(a, dtype="float32"), np.asarray(b, dtype="float32")))


def mean_embedding(vectors: list[Any]) -> list[float] | None:
    vs = [np.asarray(v, dtype="float32") for v in vectors if v is not None]
    if not vs:
        return None
    m = np.mean(vs, axis=0)
    return (m / (np.linalg.norm(m) + 1e-9)).tolist()


def score_candidate(m: dict[str, Any], *, source: Any, cast_emb: Any, cast_member: dict | None,
                    other_cast: dict[str, Any]) -> dict[str, Any]:
    """m = {embedding, det_score, face_ratio}. Returns {valid, score, reasons, metrics}."""
    emb = m.get("embedding")
    reasons = []
    metrics = {"det_score": round(float(m.get("det_score") or 0), 3), "face_ratio": round(float(m.get("face_ratio") or 0), 3)}
    if emb is None:
        return {"valid": False, "score": -9.0, "reasons": ["no face"], "metrics": metrics}
    if metrics["det_score"] < MIN_DET_SCORE:
        reasons.append(f"weak face detection {metrics['det_score']}")
    if metrics["face_ratio"] < MIN_FACE_RATIO:
        reasons.append(f"face too small {metrics['face_ratio']}")
    src = _cos(emb, source)
    metrics["source_similarity"] = round(src, 3) if src is not None else None
    if src is not None and src > MAX_SOURCE_SIMILARITY:
        reasons.append(f"too similar to the original actor ({src:.2f})")
    for name, other in other_cast.items():
        sim = _cos(emb, other)
        if sim is not None and sim > MAX_OTHER_CAST_SIMILARITY:
            reasons.append(f"looks like cast member {name} ({sim:.2f})")
    score = metrics["det_score"] + 0.3 * min(metrics["face_ratio"], 0.4) - 0.5 * max(0.0, src or 0.0)
    if cast_member is not None:
        lo, hi = cast_member.get("similarity", (0.30, 0.55))
        sim = _cos(emb, cast_emb)
        metrics["cast_similarity"] = round(sim, 3) if sim is not None else None
        if sim is None:
            reasons.append("cast reference has no face")
        elif cast_member.get("mode") == "as_is":
            if sim < max(lo, 0.55):
                reasons.append(f"not enough like {cast_member['name']} ({sim:.2f})")
            score += sim
        else:
            if sim < lo:
                reasons.append(f"resemblance to {cast_member['name']} too weak ({sim:.2f} < {lo})")
            elif sim > hi:
                reasons.append(f"too close to {cast_member['name']} — would be the same person ({sim:.2f} > {hi})")
            score -= 2.0 * abs(sim - (lo + hi) / 2)
    return {"valid": not reasons, "score": round(score, 4), "reasons": reasons, "metrics": metrics}


def next_strength(scored: list[dict], member: dict, current: str) -> str:
    """Adjust blend strength for the next attempt from the measured similarities."""
    lo, hi = member.get("similarity", (0.30, 0.55))
    sims = [s["metrics"].get("cast_similarity") for s in scored if s["metrics"].get("cast_similarity") is not None]
    if not sims:
        return current
    order = ["subtle", "clear", "strong"]
    i = order.index(current)
    if max(sims) < lo and i < 2:
        return order[i + 1]
    if min(sims) > hi and i > 0:
        return order[i - 1]
    return current


def choose(scored: list[dict], pick: int | None = None) -> int | None:
    """Index of the winner: a manual pick if it exists, else the best valid one."""
    if pick is not None and 0 <= pick < len(scored):
        return pick
    valid = [i for i, s in enumerate(scored) if s["valid"]]
    return max(valid, key=lambda i: scored[i]["score"]) if valid else None


def contact_sheet(rows: list[dict], out_path: str, thumb_h: int = 384) -> str | None:
    """Grid: one row per character, one tile per candidate, labelled with index, validity and key metrics."""
    try:
        import cv2
    except ImportError:
        return None
    lines = []
    for row in rows:
        tiles = []
        for i, c in enumerate(row["candidates"]):
            image = cv2.imread(c["path"]) if c.get("path") else None
            if image is None:
                image = np.full((thumb_h, int(thumb_h * 0.57), 3), 40, dtype=np.uint8)
            image = cv2.resize(image, (int(image.shape[1] * thumb_h / image.shape[0]), thumb_h))
            colour = (60, 200, 60) if i == row.get("chosen") else ((200, 200, 200) if c["valid"] else (60, 60, 220))
            cv2.rectangle(image, (0, 0), (image.shape[1] - 1, image.shape[0] - 1), colour, 6)
            m = c["metrics"]
            label = f"{row['id']} #{i}" + (" CHOSEN" if i == row.get("chosen") else "")
            detail = f"src {m.get('source_similarity')}" + (f" cast {m.get('cast_similarity')}" if "cast_similarity" in m else "")
            for k, text in enumerate((label, detail)):
                cv2.putText(image, text, (10, 28 + 26 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(image, text, (10, 28 + 26 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(image)
        if tiles:
            lines.append(np.concatenate(tiles, axis=1))
    if not lines:
        return None
    width = max(x.shape[1] for x in lines)
    lines = [np.pad(x, ((0, 8), (0, width - x.shape[1]), (0, 0))) for x in lines]
    cv2.imwrite(out_path, np.concatenate(lines, axis=0), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out_path
