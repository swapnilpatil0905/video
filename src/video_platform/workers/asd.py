"""Active speaker worker: which visible character is speaking in each segment. Runtime: core.

Method (heuristic, documented as such): sample frames inside the segment (clamped to its
shot so no cut is crossed), detect faces, assign each face to the nearest character
centroid, and measure lower-face (mouth/jaw) motion between consecutive frames. The
speaking face moves its lower face much more than listeners. Outputs scores per
character; the decision rules live in the speaker_mapping stage.
"""

from __future__ import annotations

from .common import device


def run(request: dict) -> dict:
    import cv2
    import numpy as np
    from insightface.app import FaceAnalysis

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device() == "cuda" else ["CPUExecutionProvider"]
    app = FaceAnalysis(name="buffalo_l", providers=providers, allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=0 if device() == "cuda" else -1, det_size=(640, 640))

    names = list(request["centroids"])
    centroids = np.array([request["centroids"][n] for n in names], dtype="float32") if names else np.zeros((0, 512))
    p = request["params"]
    capture = cv2.VideoCapture(request["video"])
    results = {}

    for seg in request["segments"]:
        start = max(seg["start"], seg["shot_start"]) + 0.04
        end = min(seg["end"], seg["shot_end"] - 0.04, start + p["max_segment_seconds"])
        if end <= start:
            results[seg["segment_id"]] = {"frames": 0, "characters": {}}
            continue
        times = np.linspace(start, end, p["frames_per_segment"])
        crops: dict[str, list] = {}
        seen: dict[str, int] = {}
        for t in times:
            capture.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
            ok, frame = capture.read()
            if not ok:
                continue
            frame_seen = set()
            for face in app.get(frame):
                x1, y1, x2, y2 = (int(v) for v in face.bbox)
                h = y2 - y1
                if h < p["min_face_px"] or len(centroids) == 0:
                    continue
                sims = centroids @ face.normed_embedding
                best = int(np.argmax(sims))
                if sims[best] < p["identity_threshold"]:
                    continue
                name = names[best]
                if name in frame_seen:
                    continue
                frame_seen.add(name)
                lower = frame[max(0, y1 + int(0.55 * h)) : max(0, y2), max(0, x1) : max(0, x2)]
                if lower.size == 0:
                    continue
                gray = cv2.cvtColor(cv2.resize(lower, (48, 32)), cv2.COLOR_BGR2GRAY).astype("float32")
                crops.setdefault(name, []).append((t, gray / (gray.mean() + 1e-3)))
            for name in frame_seen:
                seen[name] = seen.get(name, 0) + 1
        characters = {}
        step = (end - start) / max(1, p["frames_per_segment"] - 1)
        for name, series in crops.items():
            diffs = [
                float(np.mean(np.abs(b - a)))
                for (ta, a), (tb, b) in zip(series, series[1:])
                if tb - ta <= 2.5 * step  # consecutive samples only
            ]
            characters[name] = {
                "visible_ratio": seen.get(name, 0) / len(times),
                "motion": float(np.mean(diffs)) if diffs else None,
            }
        results[seg["segment_id"]] = {"frames": len(times), "characters": characters}
    capture.release()
    return {"results": results}
