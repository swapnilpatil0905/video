"""Active speaker detection with Light-ASD (trained on AVA-ActiveSpeaker). Runtime: asd.

For each speech segment (clamped to its shot): read contiguous frames at 25 fps, detect and
track faces (InsightFace), assign each track to a character by its mean identity embedding,
and score every track with Light-ASD (audio MFCC + 112x112 grey face crops). The per-track
speaking probability is returned as ``motion`` so the stage's decision rules work unchanged.

Uses the repository's own inference path (ASD.loadParameters, model.forward_*_frontend,
forward_audio_visual_backend, lossAV.forward), as in its Columbia_test.py. If the API does
not match, the response carries ``fatal`` and the stage falls back to the motion heuristic.
"""

from __future__ import annotations

import os
import sys
import traceback

from .common import device


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def run(request: dict) -> dict:
    repo = request["repo"]
    sys.path.insert(0, repo)
    try:
        import cv2
        import numpy as np
        import python_speech_features
        import soundfile as sf
        import torch
        from insightface.app import FaceAnalysis

        cwd = os.getcwd()
        os.chdir(repo)
        from ASD import ASD  # Light-ASD repository module

        asd = ASD()
        asd.loadParameters(os.path.join(repo, request["weights"]))
        asd.eval()
        os.chdir(cwd)
    except Exception:
        return {"fatal": traceback.format_exc()[-2500:], "results": {}}

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device() == "cuda" else ["CPUExecutionProvider"]
    app = FaceAnalysis(name="buffalo_l", providers=providers, allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=0 if device() == "cuda" else -1, det_size=(640, 640))

    names = list(request["centroids"])
    centroids = np.array([request["centroids"][n] for n in names], dtype="float32") if names else np.zeros((0, 512))
    p = request["params"]
    fps = int(p.get("asd_fps", 25))
    audio, rate = sf.read(request["audio"], dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    capture = cv2.VideoCapture(request["video"])
    results: dict = {}

    for seg in request["segments"]:
        start = max(seg["start"], seg["shot_start"]) + 0.04
        end = min(seg["end"], seg["shot_end"] - 0.04, start + float(p.get("asd_max_seconds", 2.5)))
        if end - start < 0.4:
            results[seg["segment_id"]] = {"frames": 0, "characters": {}, "method": "light_asd"}
            continue
        try:
            capture.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
            frames, next_t = [], start
            while True:
                t = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                ok, frame = capture.read()
                if not ok or t > end:
                    break
                if t + 1e-6 >= next_t:
                    frames.append(frame)
                    next_t += 1.0 / fps
            n = len(frames)
            if n < 10:
                results[seg["segment_id"]] = {"frames": n, "characters": {}, "method": "light_asd"}
                continue

            # detect + greedy IoU tracking
            tracks: list[dict] = []
            for index, frame in enumerate(frames):
                for face in app.get(frame):
                    box = [float(v) for v in face.bbox]
                    if box[3] - box[1] < p["min_face_px"]:
                        continue
                    best = max(tracks, key=lambda tr: _iou(tr["boxes"][-1], box) if tr["last"] == index - 1 else -1,
                               default=None)
                    if best is not None and best["last"] == index - 1 and _iou(best["boxes"][-1], box) > 0.4:
                        best["boxes"].append(box); best["idx"].append(index); best["last"] = index
                        best["emb"].append(face.normed_embedding)
                    else:
                        tracks.append({"boxes": [box], "idx": [index], "last": index, "emb": [face.normed_embedding]})

            mfcc = python_speech_features.mfcc(
                audio[int(start * rate): int((start + n / fps) * rate)], rate, numcep=13, winlen=0.025, winstep=0.010)
            characters: dict = {}
            for track in tracks:
                if len(track["idx"]) < max(10, 0.5 * n) or len(centroids) == 0:
                    continue
                emb = np.mean(track["emb"], axis=0)
                emb = emb / (np.linalg.norm(emb) + 1e-9)
                sims = centroids @ emb
                best_char = int(np.argmax(sims))
                if sims[best_char] < p["identity_threshold"]:
                    continue
                crops = []
                for index, box in zip(track["idx"], track["boxes"]):
                    frame = frames[index]
                    cx, cy, size = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, max(box[2] - box[0], box[3] - box[1]) * 0.7
                    x1, y1 = int(max(0, cx - size)), int(max(0, cy - size))
                    x2, y2 = int(min(frame.shape[1], cx + size)), int(min(frame.shape[0], cy + size))
                    grey = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                    crops.append(cv2.resize(grey, (224, 224))[56:168, 56:168])
                length = min(len(crops), mfcc.shape[0] // 4)
                if length < 10:
                    continue
                with torch.no_grad():
                    input_a = torch.FloatTensor(mfcc[: length * 4, :]).unsqueeze(0).to(device())
                    input_v = torch.FloatTensor(np.array(crops[:length])).unsqueeze(0).to(device())
                    embed_a = asd.model.forward_audio_frontend(input_a)
                    embed_v = asd.model.forward_visual_frontend(input_v)
                    out = asd.model.forward_audio_visual_backend(embed_a, embed_v)
                    score = asd.lossAV.forward(out, labels=None)
                score = np.asarray(score.detach().cpu().numpy() if hasattr(score, "detach") else score, dtype="float32")
                probability = float(np.mean(1.0 / (1.0 + np.exp(-score))))
                name = names[best_char]
                if name not in characters or characters[name]["motion"] < probability:
                    characters[name] = {"visible_ratio": len(track["idx"]) / n, "motion": probability}
            results[seg["segment_id"]] = {"frames": n, "characters": characters, "method": "light_asd"}
        except Exception as exc:  # one bad segment must not stop the batch
            results[seg["segment_id"]] = {"frames": 0, "characters": {}, "method": "light_asd",
                                          "error": f"{type(exc).__name__}: {exc}"[:300]}
    capture.release()
    return {"results": results}
