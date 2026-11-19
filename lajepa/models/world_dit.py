"""World DiT: observation tokens and noised future-feature tokens.

Token reduction is pluggable. Mean pooling (LIBERO) keeps the V-JEPA channel
count, so it can also be applied when the feature store is written. Patch
grouping (RoboTwin) concatenates each neighborhood and therefore needs the
full-resolution store; the wider per-factor projections are what move S/B/L
from 0.10/0.34/1.14B on LIBERO to 0.11/0.37/1.17B on RoboTwin.
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
from lajepa.models.masking import attend_future, attend_obs
from lajepa.models.rope import apply_rope, grid_coords


def reduce_grid(features: torch.Tensor, mode: str, factor: int) -> torch.Tensor:
    """``features`` is ``[B, T, H, W, D]``."""
    if mode == "identity" or factor == 1:
        return features
    b, t, h, w, d = features.shape
    if h % factor != 0 or w % factor != 0:
        raise ValueError(f"grid {(h, w)} is not divisible by reduction factor {factor}")
    view = features.view(b, t, h // factor, factor, w // factor, factor, d)
    if mode == "mean_pool":
        return view.mean(dim=(3, 5))
    if mode == "patch_group":
        grouped = view.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
        return grouped.view(b, t, h // factor, w // factor, factor * factor * d)
    raise ValueError(f"unknown token reduction {mode!r}")


def input_dim(mode: str, feature_dim: int, factor: int) -> int:
    if mode in ("identity", "mean_pool"):
        return feature_dim
    if mode == "patch_group":
        return feature_dim * factor * factor
    raise ValueError(f"unknown token reduction {mode!r}")


class WorldBlock(nn.Module):
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
        h_obs: torch.Tensor,
        h_fut: torch.Tensor,
        cond: torch.Tensor,
        cond_keep: torch.Tensor | None,
        t_emb: torch.Tensor,
        obs_cos: torch.Tensor,
        obs_sin: torch.Tensor,
        fut_cos: torch.Tensor,
        fut_sin: torch.Tensor,
        obs_keep: torch.Tensor | None,
        fut_keep: torch.Tensor | None,
    ):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ada(t_emb)

        def project(h, cos, sin):
            q, k, v = self.attn.project(modulate(self.norm1(h), shift_msa, scale_msa))
            return apply_rope(q, cos, sin), apply_rope(k, cos, sin), v

        q_obs, k_obs, v_obs = project(h_obs, obs_cos, obs_sin)
        h_obs = h_obs + gate_msa.unsqueeze(1) * self.attn.combine(attend_obs(q_obs, k_obs, v_obs, obs_keep))
        if h_fut.shape[1] > 0:
            q_fut, k_fut, v_fut = project(h_fut, fut_cos, fut_sin)
            mixed = attend_future(q_fut, k_obs, v_obs, k_fut, v_fut, obs_keep, fut_keep)
            h_fut = h_fut + gate_msa.unsqueeze(1) * self.attn.combine(mixed)
        h_obs = h_obs + self.cross(self.norm_cross(h_obs), cond, cond_keep)
        h_obs = h_obs + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(h_obs), shift_mlp, scale_mlp))
        if h_fut.shape[1] > 0:
            h_fut = h_fut + self.cross(self.norm_cross(h_fut), cond, cond_keep)
            h_fut = h_fut + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(h_fut), shift_mlp, scale_mlp))
        if obs_keep is not None:
            h_obs = h_obs * obs_keep.unsqueeze(-1).to(dtype=h_obs.dtype)
        if fut_keep is not None and h_fut.shape[1] > 0:
            h_fut = h_fut * fut_keep.unsqueeze(-1).to(dtype=h_fut.dtype)
        return h_obs, h_fut, k_obs, v_obs


class CameraSlice:
    def __init__(self, start: int, end: int, channels: int, keep: torch.Tensor, flat: torch.Tensor):
        self.start = start
        self.end = end
        self.channels = channels
        self.keep = keep
        self.flat = flat


class WorldDiT(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        feature_dim: int,
        num_cameras: int,
        factors: list[int],
        reduction: str,
    ):
        super().__init__()
        if len(factors) != num_cameras:
            raise ValueError("one reduction factor is required per camera")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.feature_dim = feature_dim
        self.factors = [int(f) for f in factors]
        self.reduction = reduction
        dims = sorted({input_dim(reduction, feature_dim, factor) for factor in self.factors})
        self.in_proj = nn.ModuleDict({str(dim): nn.Linear(dim, hidden_dim) for dim in dims})
        self.out_proj = nn.ModuleDict({str(dim): FinalHead(hidden_dim, dim) for dim in dims})
        self.cam_embed = nn.Parameter(torch.zeros(num_cameras, hidden_dim))
        self.obs_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.fut_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.cam_embed, std=0.02)
        nn.init.normal_(self.obs_type, std=0.02)
        nn.init.normal_(self.fut_type, std=0.02)
        self.time = TimeEmbedder(hidden_dim)
        self.blocks = nn.ModuleList([WorldBlock(hidden_dim, num_heads, mlp_ratio) for _ in range(depth)])

    def pack(
        self,
        features: torch.Tensor,
        t0: int,
        cam_keep: torch.Tensor | None,
        grid_mask: torch.Tensor | None,
    ):
        """Reduce ``[B, C, T, H, W, D]`` without projecting it.

        Flow matching noising happens on these reduced features. ``project``
        maps them into the transformer afterwards.
        """
        keeps = []
        coords = []
        slices: list[CameraSlice] = []
        cursor = 0
        for camera, factor in enumerate(self.factors):
            reduced = reduce_grid(features[:, camera], self.reduction, factor)
            b, t, h, w, channels = reduced.shape
            flat = reduced.reshape(b, t * h * w, channels)
            if grid_mask is None:
                valid = features.new_ones(b, t, dtype=torch.bool)
            else:
                valid = grid_mask[:, camera]
            if cam_keep is not None:
                valid = valid & cam_keep[:, camera].unsqueeze(1)
            keep = valid[:, :, None, None].expand(b, t, h, w).reshape(b, t * h * w)
            keeps.append(keep)
            coords.append(grid_coords(t, h, w, t0, features.device))
            slices.append(CameraSlice(cursor, cursor + flat.shape[1], channels, keep, flat))
            cursor += flat.shape[1]
        cos, sin = self._rope(torch.cat(coords, dim=0))
        return slices, torch.cat(keeps, dim=1), cos, sin

    def project(self, slices: list[CameraSlice], type_embed: torch.Tensor) -> torch.Tensor:
        tokens = []
        for camera, item in enumerate(slices):
            hidden = self.in_proj[str(item.channels)](item.flat)
            tokens.append(hidden + self.cam_embed[camera] + type_embed)
        return torch.cat(tokens, dim=1)

    def predict(self, hidden: torch.Tensor, t_emb: torch.Tensor, slices: list[CameraSlice]) -> list[torch.Tensor]:
        outputs = []
        for item in slices:
            head = self.out_proj[str(item.channels)]
            outputs.append(head(hidden[:, item.start : item.end], t_emb))
        return outputs

    def _rope(self, coords: torch.Tensor):
        from lajepa.models.rope import rope_cos_sin

        return rope_cos_sin(coords, self.head_dim)
