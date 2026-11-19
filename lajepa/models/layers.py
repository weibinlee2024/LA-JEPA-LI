"""AdaLN-zero blocks shared by the World and Action DiTs.

MLP expansion 4 uses a gated MLP (up and gate both map to ``4 * width``).
With cross-attention and the input/output projections, that is the parameter
count measured for the S policy (0.094B on LIBERO).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lajepa.models.masking import attend


def timestep_embedding(tau: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embedding of a continuous timestep in ``[0, 1]``."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=tau.device, dtype=torch.float32) / half
    )
    args = tau.float().unsqueeze(-1) * freqs.unsqueeze(0) * 1000.0
    emb = torch.cat([args.cos(), args.sin()], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimeEmbedder(nn.Module):
    def __init__(self, hidden_dim: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        return self.mlp(timestep_embedding(tau, self.freq_dim))


class AdaLNZero(nn.Module):
    def __init__(self, hidden_dim: int, chunks: int = 6):
        super().__init__()
        self.chunks = chunks
        self.proj = nn.Linear(hidden_dim, chunks * hidden_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, t_emb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.proj(t_emb).chunk(self.chunks, dim=-1)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim {hidden_dim} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, s, _ = x.shape
        qkv = self.qkv(x).view(b, s, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        return q, k, v

    def combine(self, attn_out: torch.Tensor) -> torch.Tensor:
        b, s, h, d = attn_out.shape
        return self.out_proj(attn_out.reshape(b, s, h * d))


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim {hidden_dim} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.kv = nn.Linear(hidden_dim, hidden_dim * 2)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, cond_keep: torch.Tensor | None) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.q(x).view(b, s, self.num_heads, self.head_dim)
        kv = self.kv(cond).view(b, cond.shape[1], 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(dim=2)
        out = attend(q, k, v, cond_keep)
        return self.out_proj(out.reshape(b, s, -1))


class GatedMLP(nn.Module):
    def __init__(self, hidden_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        inner = int(hidden_dim * mlp_ratio)
        self.up = nn.Linear(hidden_dim, inner)
        self.gate = nn.Linear(hidden_dim, inner)
        self.down = nn.Linear(inner, hidden_dim)
        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.down.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class FinalHead(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.proj = nn.Linear(hidden_dim, out_dim)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        shift, scale = self.modulation(t_emb).chunk(2, dim=-1)
        return self.proj(modulate(self.norm(x), shift, scale))


def init_linear_(module: nn.Module) -> None:
    for submodule in module.modules():
        if isinstance(submodule, nn.Linear):
            nn.init.xavier_uniform_(submodule.weight)
            if submodule.bias is not None:
                nn.init.zeros_(submodule.bias)
