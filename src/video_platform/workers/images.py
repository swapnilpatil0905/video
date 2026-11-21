"""Character reference image worker (diffusers: Qwen-Image + Qwen-Image-Edit). Runtime: image."""

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


def run(request: dict) -> dict:
    import torch

    results = {c["id"]: {"master": None, "variants": {}, "errors": []} for c in request["characters"]}
    gen_device = "cuda" if device() == "cuda" and torch.cuda.is_available() else "cpu"

    pipe = _load(request["model_repo"])
    progress = Progress("images-master", len(request["characters"]))
    for c in request["characters"]:
        progress.step()
        try:
            generator = torch.Generator(gen_device).manual_seed(int(c["seed"]))
            image = pipe(
                prompt=c["prompt"], negative_prompt=c.get("negative", " "), width=request["width"],
                height=request["height"], num_inference_steps=request["steps"],
                true_cfg_scale=request["true_cfg_scale"], generator=generator,
            ).images[0]
            path = f"{c['out_dir']}/master.png"
            image.save(path)
            results[c["id"]]["master"] = path
        except Exception as exc:
            results[c["id"]]["errors"].append(f"master: {type(exc).__name__}: {exc}")
    del pipe
    free_gpu()

    if request.get("edit_repo"):
        from PIL import Image

        edit = _load(request["edit_repo"])
        for c in request["characters"]:
            master = results[c["id"]]["master"]
            if not master:
                continue
            for name, prompt in c["variants"].items():
                try:
                    generator = torch.Generator(gen_device).manual_seed(int(c["seed"]) + 1)
                    image = edit(
                        image=Image.open(master).convert("RGB"), prompt=prompt, negative_prompt=" ",
                        num_inference_steps=request["edit_steps"], true_cfg_scale=request["true_cfg_scale"],
                        generator=generator,
                    ).images[0]
                    path = f"{c['out_dir']}/{name}.png"
                    image.save(path)
                    results[c["id"]]["variants"][name] = path
                except Exception as exc:
                    results[c["id"]]["errors"].append(f"{name}: {type(exc).__name__}: {exc}")
        del edit
        free_gpu()
    return {"characters": results}
