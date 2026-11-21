"""Video upscaling worker (Real-ESRGAN via spandrel). Runtime: image. One model load for all clips."""

from __future__ import annotations

from .common import device


def run(request: dict) -> dict:
    import cv2
    import numpy as np
    import torch
    from spandrel import ModelLoader

    dev = "cuda" if device() == "cuda" and torch.cuda.is_available() else "cpu"
    model = ModelLoader().load_from_file(request["weights"]).to(dev).eval()
    half = dev == "cuda" and getattr(model, "supports_half", False)
    if half:
        model = model.half()
    results = {}
    batch_size = int(request.get("batch", 8))
    for job in request["jobs"]:
        try:
            capture = cv2.VideoCapture(job["input"])
            fps = capture.get(cv2.CAP_PROP_FPS) or 24
            writer = None
            frames: list = []

            def flush():
                nonlocal writer
                if not frames:
                    return
                tensor = torch.from_numpy(np.stack(frames)[..., ::-1].copy()).permute(0, 3, 1, 2).to(dev)
                tensor = (tensor.half() if half else tensor.float()) / 255.0
                with torch.no_grad():
                    out = model(tensor).clamp(0, 1)
                out = (out.float().permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype("uint8")[..., ::-1]
                if writer is None:
                    h, w = out.shape[1:3]
                    writer = cv2.VideoWriter(job["output"], cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                for frame in out:
                    writer.write(np.ascontiguousarray(frame))
                frames.clear()

            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
                if len(frames) >= batch_size:
                    flush()
            flush()
            capture.release()
            if writer is not None:
                writer.release()
            results[job["id"]] = {"ok": writer is not None}
        except Exception as exc:
            results[job["id"]] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:500]}
        finally:
            if dev == "cuda":
                torch.cuda.empty_cache()
    return {"results": results}
