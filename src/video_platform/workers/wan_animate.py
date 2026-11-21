"""Persistent Wan2.2-Animate worker: load the 14B model ONCE, render many clips. Runtime: wan.

Uses Wan2.2's Python API (the same calls generate.py makes). Keyword arguments are
filtered against the installed version's signatures so minor API differences don't
break it. If the API cannot be used at all, the response carries ``fatal`` and the
caller falls back to one CLI invocation per clip (slower, always works).

Request: {"repo", "ckpt_dir", "task", "frame_num", "offload_model": bool, "use_relighting_lora": bool,
          "jobs": [{"id", "src_root", "out", "seed"}]}
"""

from __future__ import annotations

import inspect
import os
import sys
import time
import traceback


def _accepts(fn, candidates: dict) -> dict:
    params = inspect.signature(fn).parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return candidates
    return {k: v for k, v in candidates.items() if k in params}


def merge_lora(model_obj, state_dict: dict, strength: float) -> int:
    """Merge a LoRA (kohya lora_down/up or peft lora_A/B, optional alpha) into the model's DiT weights."""
    import torch

    params = {}
    for value in vars(model_obj).values():
        if isinstance(value, torch.nn.Module):
            for name, param in value.named_parameters():
                params.setdefault(name, param)
    groups: dict[str, dict] = {}
    for key, tensor in state_dict.items():
        for suffix, role in ((".lora_down.weight", "down"), (".lora_up.weight", "up"), (".lora_A.weight", "down"),
                             (".lora_B.weight", "up"), (".alpha", "alpha")):
            if key.endswith(suffix):
                groups.setdefault(key[: -len(suffix)], {})[role] = tensor
    applied = 0
    for base, g in groups.items():
        if "down" not in g or "up" not in g:
            continue
        target = None
        for prefix in ("diffusion_model.", "model.diffusion_model.", "transformer.", "model.", ""):
            name = (base[len(prefix):] if base.startswith(prefix) else base) + ".weight"
            if name in params:
                target = params[name]
                break
        if target is None:
            continue
        down, up = g["down"].float(), g["up"].float()
        rank = down.shape[0]
        scale = strength * (float(g["alpha"]) / rank if "alpha" in g else 1.0)
        delta = (up.flatten(1) @ down.flatten(1)).reshape(target.shape)
        with torch.no_grad():
            target.add_((scale * delta).to(device=target.device, dtype=target.dtype))
        applied += 1
    return applied


def enable_sage_attention() -> str:
    """Route full (unpadded) attention calls through SageAttention; everything else keeps the original kernel."""
    import torch
    from sageattention import sageattn
    import wan.modules.attention as attention_module

    original = attention_module.flash_attention

    def patched(q, k, v, q_lens=None, k_lens=None, *args, **kwargs):
        try:
            if (q.shape[0] == 1 and q.dtype in (torch.float16, torch.bfloat16)
                    and (k_lens is None or int(k_lens.max()) == k.shape[1])
                    and (q_lens is None or int(q_lens.max()) == q.shape[1])):
                return sageattn(q, k, v, tensor_layout="NHD", is_causal=False)
        except Exception:
            pass
        return original(q, k, v, q_lens, k_lens, *args, **kwargs)

    patched_modules = 0
    for name, module in list(sys.modules.items()):
        if name.startswith("wan") and getattr(module, "flash_attention", None) is original:
            module.flash_attention = patched
            patched_modules += 1
    return f"sage attention patched into {patched_modules} modules"


def _save_function():
    try:
        from wan.utils.utils import save_video
        return save_video
    except ImportError:
        from wan.utils.utils import cache_video
        return cache_video


def run(request: dict) -> dict:
    repo = request["repo"]
    sys.path.insert(0, repo)
    os.chdir(repo)
    try:
        import torch
        import wan
        from wan.configs import WAN_CONFIGS

        cfg = WAN_CONFIGS[request.get("task", "animate-14B")]
        model_cls = getattr(wan, "WanAnimate")
        init = _accepts(model_cls.__init__, {
            "config": cfg, "checkpoint_dir": request["ckpt_dir"], "device_id": 0, "rank": 0,
            "t5_fsdp": False, "dit_fsdp": False, "use_sp": False, "t5_cpu": False, "init_on_cpu": False,
            "convert_model_dtype": True, "use_relighting_lora": bool(request.get("use_relighting_lora", True)),
        })
        started = time.time()
        model = model_cls(**init)
        load_seconds = time.time() - started
        save = _save_function()
    except Exception:
        return {"fatal": traceback.format_exc()[-3000:], "results": {}}

    notes = {}
    lora = request.get("lora")
    if lora:
        try:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file

            applied = merge_lora(model, load_file(hf_hub_download(lora["repo"], lora["file"])), float(lora["strength"]))
            notes["lora_tensors_merged"] = applied
            if applied == 0:
                raise RuntimeError("no LoRA tensors matched the model weights")
        except Exception as exc:
            return {"fatal": f"lora: {type(exc).__name__}: {exc}", "lora_failed": True, "results": {}}
    if request.get("attention") == "sage":
        try:
            notes["attention"] = enable_sage_attention()
        except Exception as exc:
            notes["attention"] = f"default (sage unavailable: {type(exc).__name__}: {exc})"

    results = {}
    for job in request["jobs"]:
        t0 = time.time()
        try:
            gen = _accepts(model.generate, {
                "src_root_path": job["src_root"], "replace_flag": True, "refert_num": 1,
                "clip_len": int(request.get("frame_num", 77)), "shift": getattr(cfg, "sample_shift", 5.0),
                "sample_solver": "dpm++", "sampling_steps": int(request.get("sampling_steps") or getattr(cfg, "sample_steps", 20)),
                "guide_scale": getattr(cfg, "sample_guide_scale", 1.0), "seed": int(job["seed"]),
                "offload_model": bool(request.get("offload_model", False)),
            })
            with torch.no_grad():
                video = model.generate(**gen)
            save(**_accepts(save, {"tensor": video[None], "save_file": job["out"],
                                   "fps": getattr(cfg, "sample_fps", 30), "nrow": 1, "normalize": True,
                                   "value_range": (-1, 1)}))
            results[job["id"]] = {"ok": os.path.exists(job["out"]), "seconds": round(time.time() - t0, 2)}
        except Exception as exc:
            results[job["id"]] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:1500],
                                  "seconds": round(time.time() - t0, 2)}
            if isinstance(exc, TypeError) and not any(r.get("ok") for r in results.values()):
                return {"fatal": traceback.format_exc()[-3000:], "results": results}  # API mismatch → CLI
        finally:
            torch.cuda.empty_cache()
    return {"results": results, "load_seconds": round(load_seconds, 2), "notes": notes}
