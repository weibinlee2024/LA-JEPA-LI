"""Action DiT over the noised action chunk.

Each layer's queries read the World DiT's observation keys and values from
the same depth, concatenated with the action stream's own keys and values.
Future keys are not in that concatenation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lajepa.models.layers import (
    AdaLNZero,
    CrossAttention,
    FinalHead,
    GatedMLP,
    SelfAttention,
    TimeEmbedder,
    modulate,
)
from lajepa.models.masking import attend_action
from lajepa.models.rope import apply_rope, grid_coords, rope_cos_sin


class ActionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(hidden_dim, num_heads)
        self.norm_cross = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.cross = CrossAttention(hidden_dim, num_heads)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = GatedMLP(hidden_dim, mlp_ratio)
        self.ada = AdaLNZero(hidden_dim, chunks=6)

    def forward(
        self,
        h_act: torch.Tensor,
        k_obs: torch.Tensor,
        v_obs: torch.Tensor,
        cond: torch.Tensor,
        cond_keep: torch.Tensor | None,
        t_emb: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        obs_keep: torch.Tensor | None,
        act_keep: torch.Tensor | None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ada(t_emb)
        q, k, v = self.attn.project(modulate(self.norm1(h_act), shift_msa, scale_msa))
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        mixed = attend_action(q, k_obs, v_obs, k, v, obs_keep, act_keep)
        h_act = h_act + gate_msa.unsqueeze(1) * self.attn.combine(mixed)
        h_act = h_act + self.cross(self.norm_cross(h_act), cond, cond_keep)
        h_act = h_act + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(h_act), shift_mlp, scale_mlp))
        return h_act


class ActionDiT(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        action_dim: int,
        horizon: int,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        self.horizon = horizon
        self.action_in = nn.Linear(action_dim, hidden_dim)
        self.head = FinalHead(hidden_dim, action_dim)
        self.time = TimeEmbedder(hidden_dim)
        self.blocks = nn.ModuleList([ActionBlock(hidden_dim, num_heads, mlp_ratio) for _ in range(depth)])
        coords = grid_coords(horizon, 1, 1, 0, torch.device("cpu"))
        cos, sin = rope_cos_sin(coords, self.head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def embed(self, noisy_actions: torch.Tensor) -> torch.Tensor:
        return self.action_in(noisy_actions)

    def rope(self, device: torch.device, dtype: torch.dtype):
        return self.rope_cos.to(device=device, dtype=dtype), self.rope_sin.to(device=device, dtype=dtype)
