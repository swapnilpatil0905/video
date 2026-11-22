"""PyTorch stand-in for Wan's flash_attention: same shapes, and padded keys are ignored like flash-attn's k_lens."""

import pytest

torch = pytest.importorskip("torch")

from video_platform.workers.wan_attention import sdpa_flash_attention  # noqa: E402


def test_shapes_dtype_and_key_padding_mask():
    torch.manual_seed(0)
    q = torch.randn(1, 6, 2, 8)
    k = torch.randn(1, 5, 2, 8)
    v = torch.randn(1, 5, 2, 8)
    out = sdpa_flash_attention(q, k, v, k_lens=torch.tensor([3]), dtype=torch.float32)
    assert out.shape == q.shape and out.dtype == q.dtype
    # padded keys (positions 3, 4) must not matter: same result with them changed
    k2, v2 = k.clone(), v.clone()
    k2[:, 3:], v2[:, 3:] = 100.0, -100.0
    out2 = sdpa_flash_attention(q, k2, v2, k_lens=torch.tensor([3]), dtype=torch.float32)
    assert torch.allclose(out, out2, atol=1e-5)
    # and it equals attention over only the first 3 keys
    ref = sdpa_flash_attention(q, k[:, :3], v[:, :3], dtype=torch.float32)
    assert torch.allclose(out, ref, atol=1e-5)


def test_flash_attn_func_stand_in_matches_layout():
    from video_platform.workers.wan_attention import flash_attn_func

    q = torch.randn(2, 7, 3, 8)
    k = torch.randn(2, 5, 3, 8)
    out = flash_attn_func(q, k, k)
    assert out.shape == (2, 7, 3, 8)
