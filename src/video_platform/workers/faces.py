"""Face worker (OpenCV YuNet / InsightFace). Runtime: core.

Modes:
  detect   — images → face boxes + padded crops (YuNet, matching the legacy method)
  embed    — face crops → 512-d identity embeddings + gender/age (InsightFace buffalo_l)
  identity — arbitrary images (e.g. generated frames) → embedding of the largest face
"""

from __future__ import annotations

import os

from .common import device


def _providers() -> list[str]:
    return ["CUDAExecutionProvider", "CPUExecutionProvider"] if device() == "cuda" else ["CPUExecutionProvider"]


def _insightface(det_size: int):
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", providers=_providers())
    app.prepare(ctx_id=0 if device() == "cuda" else -1, det_size=(det_size, det_size))
    return app


def _largest(faces):
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])) if faces else None


def detect(request: dict) -> dict:
    import cv2
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(request.get("hf_repo", "opencv/face_detection_yunet"), request["model_file"])
    detector = cv2.FaceDetectorYN_create(
        model_path, "", (320, 320), request["confidence_threshold"], request["nms_threshold"], request["top_k"]
    )
    os.makedirs(request["crop_dir"], exist_ok=True)
    detections = []
    face_number = 0
    for image_item in request["images"]:
        image = cv2.imread(image_item["path"])
        if image is None:
            continue
        height, width = image.shape[:2]
        detector.setInputSize((width, height))
        _, faces = detector.detect(image)
        for face in faces if faces is not None else []:
            x, y, w, h = (float(v) for v in face[:4])
            confidence = float(face[-1])
            if confidence < request["confidence_threshold"]:
                continue
            cx, cy = x + w / 2, y + h / 2
            side = max(w, h) * (1 + 2 * request["padding"])
            x1, y1 = max(0, int(cx - side / 2)), max(0, int(cy - side / 2))
            x2, y2 = min(width, int(cx + side / 2)), min(height, int(cy + side / 2))
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            face_number += 1
            crop_path = os.path.join(request["crop_dir"], f"face_{face_number:06d}.jpg")
            cv2.imwrite(crop_path, cv2.resize(crop, (request["crop_size"],) * 2),
                        [cv2.IMWRITE_JPEG_QUALITY, request["jpeg_quality"]])
            detections.append({
                "face_number": face_number, "shot_id": image_item["shot_id"], "time_s": image_item.get("time_s"),
                "bbox": [x, y, w, h], "confidence": confidence, "crop": crop_path,
            })
    return {"detections": detections}


def detect_insightface(request: dict) -> dict:
    """SCRFD detection + ArcFace embedding + gender/age in one pass (buffalo_l), with padded crops."""
    import cv2
    import numpy as np

    app = _insightface(request.get("det_size", 640))
    os.makedirs(request["crop_dir"], exist_ok=True)
    detections, ids, vectors, attributes = [], [], [], {}
    number = 0
    for image_item in request["images"]:
        image = cv2.imread(image_item["path"])
        if image is None:
            continue
        height, width = image.shape[:2]
        for face in app.get(image):
            score = float(getattr(face, "det_score", 1.0))
            if score < request["confidence_threshold"]:
                continue
            x1, y1, x2, y2 = (float(v) for v in face.bbox)
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w / 2, y1 + h / 2
            side = max(w, h) * (1 + 2 * request["padding"])
            a, b = max(0, int(cx - side / 2)), max(0, int(cy - side / 2))
            c, d = min(width, int(cx + side / 2)), min(height, int(cy + side / 2))
            crop = image[b:d, a:c]
            if crop.size == 0:
                continue
            number += 1
            face_id = f"FACE_{number:06d}"
            path = os.path.join(request["crop_dir"], f"face_{number:06d}.jpg")
            cv2.imwrite(path, cv2.resize(crop, (request["crop_size"],) * 2), [cv2.IMWRITE_JPEG_QUALITY, request["jpeg_quality"]])
            detections.append({"face_number": number, "shot_id": image_item["shot_id"], "time_s": image_item.get("time_s"),
                               "bbox": [x1, y1, w, h], "confidence": score, "crop": path})
            ids.append(face_id)
            vectors.append(face.normed_embedding.astype("float32"))
            sex, age = getattr(face, "sex", None), getattr(face, "age", None)
            attributes[face_id] = {"gender": sex, "age": int(age) if age is not None else None}
    np.savez_compressed(request["out_npz"], face_ids=np.array(ids),
                        embeddings=np.stack(vectors) if vectors else np.zeros((0, 512), "float32"))
    return {"detections": detections, "attributes": attributes}


def embed(request: dict) -> dict:
    import cv2
    import numpy as np

    app = _insightface(request.get("det_size", 320))
    ids, vectors, attributes = [], [], {}
    for item in request["crops"]:
        image = cv2.imread(item["path"])
        if image is None:
            continue
        face = _largest(app.get(image))
        if face is None:
            continue
        ids.append(item["face_id"])
        vectors.append(face.normed_embedding.astype("float32"))
        sex = getattr(face, "sex", None)
        age = getattr(face, "age", None)
        attributes[item["face_id"]] = {"gender": sex, "age": int(age) if age is not None else None}
    np.savez_compressed(request["out_npz"], face_ids=np.array(ids), embeddings=np.stack(vectors) if vectors else np.zeros((0, 512), "float32"))
    return {"count": len(ids), "attributes": attributes}


def _pick_face(faces, which: str):
    """largest | left | right | only (largest, flagged if several)."""
    faces = [f for f in faces or []]
    if not faces:
        return None
    if which == "left":
        return min(faces, key=lambda f: (f.bbox[0] + f.bbox[2]) / 2)
    if which == "right":
        return max(faces, key=lambda f: (f.bbox[0] + f.bbox[2]) / 2)
    return _largest(faces)


def identity(request: dict) -> dict:
    import cv2

    app = _insightface(request.get("det_size", 640))
    out, ratio, score, count, all_faces, gender = {}, {}, {}, {}, {}, {}
    for item in request["images"]:
        image = cv2.imread(item["path"])
        faces = app.get(image) if image is not None else []
        face = _pick_face(faces, item.get("face", "largest"))
        count[item["id"]] = len(faces)
        if item.get("face") == "auto" and image is not None:  # every face: the stage decides which one is the person
            all_faces.setdefault(item["id"], [
                {"embedding": f.normed_embedding.astype("float32").tolist(), "det_score": float(f.det_score),
                 "face_ratio": float((f.bbox[3] - f.bbox[1]) / image.shape[0])} for f in faces])
        out[item["id"]] = face.normed_embedding.astype("float32").tolist() if face is not None else None
        ratio[item["id"]] = float((face.bbox[3] - face.bbox[1]) / image.shape[0]) if face is not None else 0.0
        score[item["id"]] = float(face.det_score) if face is not None else 0.0
        gender[item["id"]] = getattr(face, "sex", None) if face is not None else None  # "M" / "F" (genderage model)
    return {"embeddings": out, "face_ratio": ratio, "det_score": score, "face_count": count, "all_faces": all_faces,
            "gender": gender}


def run(request: dict) -> dict:
    return {"detect": detect, "detect_insightface": detect_insightface, "embed": embed,
            "identity": identity}[request["mode"]](request)
