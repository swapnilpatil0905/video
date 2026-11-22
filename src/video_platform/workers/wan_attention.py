"""Make Wan2.2 run without flash-attn. Runtime: wan.

Wan's transformer calls ``wan.modules.attention.flash_attention`` directly, and that function
*asserts* flash-attn is installed (only the separate ``attention()`` helper falls back). flash-attn
often cannot be built for a new GPU/toolkit, so without this every clip fails on that assert.

``install_fallback()`` swaps in a PyTorch scaled-dot-product version with the same signature that
keeps the key padding mask (``k_lens``, e.g. the padded text context), so results match the
original apart from kernel rounding. It does nothing when flash-attn is available.

Also runnable as a launcher for Wan's own scripts:
    python -m video_platform.workers.wan_attention generate.py --task animate-14B …
"""

from __future__ import annotations

import runpy
import sys


def sdpa_flash_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.0, softmax_scale=None, q_scale=None,
                         causal=False, window_size=(-1, -1), deterministic=False, dtype=None, version=None):
    """Drop-in for Wan's flash_attention: q/k/v are [B, L, heads, C]; returns [B, Lq, heads, C]."""
    import torch
    import torch.nn.functional as F

    out_dtype = q.dtype
    work = dtype if dtype in (torch.float16, torch.bfloat16) else (q.dtype if q.dtype in (torch.float16, torch.bfloat16)
                                                                  else torch.bfloat16)
    if q_scale is not None:
        q = q * q_scale
    q, k, v = (t.transpose(1, 2).to(work) for t in (q, k, v))  # -> [B, heads, L, C]
    mask = None
    if k_lens is not None and not causal:
        lens = torch.as_tensor(k_lens, device=k.device).reshape(-1)
        if int(lens.min()) < k.size(2):
            mask = (torch.arange(k.size(2), device=k.device)[None, :] < lens[:, None])[:, None, None, :]
    kwargs = {"scale": softmax_scale} if softmax_scale is not None else {}
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p, is_causal=causal, **kwargs)
    return out.transpose(1, 2).contiguous().to(out_dtype)


def install_fallback() -> str:
    """Patch every loaded wan module's flash_attention when flash-attn is missing. Import ``wan`` first."""
    import wan.modules.attention as attention_module

    if getattr(attention_module, "FLASH_ATTN_2_AVAILABLE", False) or getattr(attention_module, "FLASH_ATTN_3_AVAILABLE", False):
        return "flash-attn available"
    original = attention_module.flash_attention
    patched = 0
    for name, module in list(sys.modules.items()):
        if name.startswith("wan") and getattr(module, "flash_attention", None) is original:
            module.flash_attention = sdpa_flash_attention
            patched += 1
    return f"flash-attn missing: PyTorch attention patched into {patched} modules"


def install_imread_fallback() -> str:
    """cv2.imread can return None for valid PNGs inside the Wan process ("libpng error: bad parameters to
    zlib": two zlib builds loaded). When it does, read the file with Pillow and return what OpenCV would."""
    import os

    import cv2
    import numpy as np

    original = getattr(cv2.imread, "_vp_original", cv2.imread)

    def imread(path, flags=cv2.IMREAD_COLOR):
        image = original(path, flags)
        if image is not None or not os.path.isfile(str(path)):
            return image
        from PIL import Image

        with Image.open(path) as im:
            if flags == cv2.IMREAD_GRAYSCALE:
                return np.array(im.convert("L"))
            if flags == cv2.IMREAD_UNCHANGED and im.mode in ("RGBA", "LA"):
                rgba = np.array(im.convert("RGBA"))
                return rgba[..., [2, 1, 0, 3]].copy()
            return np.array(im.convert("RGB"))[..., ::-1].copy()

    imread._vp_original = original
    cv2.imread = imread
    return "cv2.imread falls back to Pillow when OpenCV cannot decode a file"


def main() -> int:
    """Launcher: patch, then run a Wan script (first argument) as __main__ with the remaining arguments."""
    if len(sys.argv) < 2:
        print("usage: python -m video_platform.workers.wan_attention <script.py> [args…]", file=sys.stderr)
        return 2
    script = sys.argv[1]
    sys.path.insert(0, ".")
    import wan  # noqa: F401  (loads the modules to patch; cwd is the Wan repo)

    print(install_fallback(), file=sys.stderr, flush=True)
    print(install_imread_fallback(), file=sys.stderr, flush=True)
    sys.argv = [script, *sys.argv[2:]]
    runpy.run_path(script, run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
