"""Spatiotemporal rotary positions for the from-scratch policy DiTs.

Head dimension is split across time, height, and width. Every channel is
rotated. This is independent of the V-JEPA encoder RoPE, which has to match
the pretrained checkpoint (including its frequency layout).
"""

from __future__ import annotations

import torch


def axis_dims(head_dim: int) -> tuple[int, int, int]:
    if head_dim % 2 != 0 or head_dim < 6:
        raise ValueError(f"head_dim must be even and >= 6 for 3D RoPE, got {head_dim}")
    pairs = head_dim // 2
    base, rem = divmod(pairs, 3)
    counts = [base, base, base]
    for i in range(rem):
        counts[i] += 1
    return counts[0] * 2, counts[1] * 2, counts[2] * 2


def rope_cos_sin(coords: torch.Tensor, head_dim: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """``coords`` is integer ``[S, 3]`` = (time, y, x). Returns ``[S, head_dim]``."""
    dims = axis_dims(head_dim)
    cos_parts = []
    sin_parts = []
    for axis, dim in enumerate(dims):
        half = dim // 2
        freq = 1.0 / (base ** (torch.arange(half, device=coords.device, dtype=torch.float32) / half))
        ang = coords[:, axis].float().unsqueeze(-1) * freq.unsqueeze(0)
        cos_parts.append(torch.repeat_interleave(ang.cos(), 2, dim=-1))
        sin_parts.append(torch.repeat_interleave(ang.sin(), 2, dim=-1))
    return torch.cat(cos_parts, dim=-1), torch.cat(sin_parts, dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """``x`` is ``[B, S, H, D]``; ``cos/sin`` are ``[S, D]``."""
    cos = cos[None, :, None, :].to(dtype=x.dtype)
    sin = sin[None, :, None, :].to(dtype=x.dtype)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
    return x * cos + rotated * sin


def grid_coords(n_time: int, height: int, width: int, t0: int, device: torch.device) -> torch.Tensor:
    """Flattened ``(t, y, x)`` in the same order as ``tensor.flatten(time, y, x)``."""
    ts = torch.arange(t0, t0 + n_time, device=device)
    ys = torch.arange(height, device=device)
    xs = torch.arange(width, device=device)
    grid = torch.stack(torch.meshgrid(ts, ys, xs, indexing="ij"), dim=-1)
    return grid.reshape(-1, 3)
