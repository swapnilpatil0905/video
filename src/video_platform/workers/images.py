"""Character reference image worker (diffusers: Qwen-Image + Qwen-Image-Edit-2509). Runtime: image.

Modes (request["mode"]):
  candidates  Qwen-Image text-to-image: several master candidates per character (one per seed); then,
              for cast characters, Qwen-Image-Edit-2509 blends each candidate with the cast member's
              photo into a NEW person (multi-image edit). Every output path is given by the stage.
  variants    Qwen-Image-Edit: pose/expression sheet from the chosen master.
"""

from __future__ import annotations

from .common import Progress, device, free_gpu


def _load(repo: str):
    import torch
    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(repo, torch_dtype=torch.bfloat16)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
    if total_gb >= 79:
        pipe.to("cuda")
    elif total_gb > 0:
        pipe.enable_model_cpu_offload()
    return pipe


def _generator(seed: int):
    import torch

    dev = "cuda" if device() == "cuda" and torch.cuda.is_available() else "cpu"
    return torch.Generator(dev).manual_seed(int(seed))


def _edit(edit, images, prompt: str, request: dict, seed: int, negative: str = " "):
    """Qwen-Image-Edit-2509 takes a list of images; older pipelines take one (then only the first is used)."""
    kwargs = dict(prompt=prompt, negative_prompt=negative, num_inference_steps=request["edit_steps"],
                  true_cfg_scale=request["true_cfg_scale"], generator=_generator(seed))
    try:
        return edit(image=images if len(images) > 1 else images[0], **kwargs).images[0], len(images) > 1
    except (TypeError, ValueError):
        return edit(image=images[0], **kwargs).images[0], False


def candidates(request: dict) -> dict:
    from PIL import Image

    jobs = request["jobs"]  # [{id, prompt, negative, seed, out, blend: {photos, prompt, out, seed} | None}]
    results = {j["id"]: {"candidate": None, "blend": None, "errors": []} for j in jobs}
    todo = [j for j in jobs if not j.get("skip_candidate")]
    if todo:
        pipe = _load(request["model_repo"])
        progress = Progress("face-candidates", len(todo))
        for j in todo:
            progress.step(j["id"])
            try:
                image = pipe(prompt=j["prompt"], negative_prompt=j.get("negative", " "), width=request["width"],
                             height=request["height"], num_inference_steps=request["steps"],
                             true_cfg_scale=request["true_cfg_scale"], generator=_generator(j["seed"])).images[0]
                image.save(j["out"])
                results[j["id"]]["candidate"] = j["out"]
            except Exception as exc:  # noqa: BLE001
                results[j["id"]]["errors"].append(f"candidate: {type(exc).__name__}: {exc}"[:300])
        del pipe
        free_gpu()
    for j in jobs:
        if j.get("skip_candidate"):
            results[j["id"]]["candidate"] = j["out"]
    blends = [j for j in jobs if j.get("blend") and results[j["id"]]["candidate"]]
    if blends:
        edit = _load(request["edit_repo"])
        progress = Progress("face-blends", len(blends))
        for j in blends:
            progress.step(j["id"])
            b = j["blend"]
            try:
                photos = [Image.open(p).convert("RGB") for p in b["photos"]]
                base = Image.open(results[j["id"]]["candidate"]).convert("RGB")
                image, multi = _edit(edit, [*photos[:2], base], b["prompt"], request, b["seed"], j.get("negative", " "))
                image.save(b["out"])
                results[j["id"]]["blend"] = b["out"]
                if not multi:
                    results[j["id"]]["errors"].append("blend: edit pipeline took a single image (no multi-image blend)")
            except Exception as exc:  # noqa: BLE001
                results[j["id"]]["errors"].append(f"blend: {type(exc).__name__}: {exc}"[:300])
        del edit
        free_gpu()
    return {"results": results}


def variants(request: dict) -> dict:
    from PIL import Image

    jobs = request["jobs"]  # [{id, master, seed, variants: {name: {prompt, out}}}]
    results = {j["id"]: {"variants": {}, "errors": []} for j in jobs}
    edit = _load(request["edit_repo"])
    progress = Progress("face-variants", sum(len(j["variants"]) for j in jobs))
    for j in jobs:
        master = Image.open(j["master"]).convert("RGB")
        for name, v in j["variants"].items():
            progress.step(f"{j['id']}/{name}")
            try:
                image, _ = _edit(edit, [master], v["prompt"], request, j["seed"] + 1)
                image.save(v["out"])
                results[j["id"]]["variants"][name] = v["out"]
            except Exception as exc:  # noqa: BLE001
                results[j["id"]]["errors"].append(f"{name}: {type(exc).__name__}: {exc}"[:300])
    del edit
    free_gpu()
    return {"results": results}


def run(request: dict) -> dict:
    return {"candidates": candidates, "variants": variants}[request.get("mode", "candidates")](request)
