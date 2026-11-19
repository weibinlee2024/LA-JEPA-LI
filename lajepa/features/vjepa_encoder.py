"""Frozen V-JEPA 2.1 ViT-L encoder.

The distilled ViT-L used by La-JEPA is the ~304.7M video transformer: width
1024, depth 24, 16 heads, patch 16, tubelet 2, 3-axis RoPE, a separate
tubelet-1 image stem, image/video modality embeddings, and four hierarchical
LayerNorms at blocks 5, 11, 17 and 23. La-JEPA reads the last of those
normalizations (1024-d tokens), not the predictor and not the 4096-d
concatenation.

RoPE follows the pretrained V-JEPA layout, including the duplicated-frequency
rotation the public checkpoint was trained with. Spatial positions are
rescaled by ``(rope_grid - 1) / (patches - 1)`` with ``rope_grid = 16`` so a
256 input sits on the native grid and a 384 checkpoint can still be evaluated
at 256.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HIERARCHICAL_LAYERS = (5, 11, 17, 23)
VIT_L_PARAMS = 304_680_960
# Distilled ViT-L released with V-JEPA 2.1. Training and feature precompute
# both read the EMA encoder inside this file. Override with VJEPA_CHECKPOINT.
DEFAULT_VJEPA_CHECKPOINT = Path(
    os.environ.get("VJEPA_CHECKPOINT", "checkpoints/vjepa2_1_vitl_dist_vitG_384.pt")
)
_CHECKPOINT_SOURCES = ("ema_encoder", "target_encoder", "encoder", "model", "state_dict")


def _vjepa_rotate(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """V-JEPA rotary helper, including the pretrained frequency duplication."""
    _, _, _, dim = x.shape
    omega = torch.arange(dim // 2, device=x.device, dtype=torch.float32)
    omega = omega / (dim / 2.0)
    omega = 1.0 / (10000.0 ** omega)
    if pos.ndim == 1:
        pos = pos.view(1, 1, -1)
    freq = pos.float().unsqueeze(-1) * omega.view(1, 1, 1, -1)
    emb_sin = freq.sin().repeat(1, 1, 1, 2)
    emb_cos = freq.cos().repeat(1, 1, 1, 2)
    pairs = x.unflatten(-1, (-1, 2))
    first, second = pairs.unbind(dim=-1)
    rotated = torch.stack((-second, first), dim=-1).flatten(-2)
    return x * emb_cos.to(dtype=x.dtype) + rotated * emb_sin.to(dtype=x.dtype)


class PatchEmbed3d(nn.Module):
    def __init__(self, tubelet: int, patch: int, embed_dim: int):
        super().__init__()
        self.tubelet = tubelet
        self.patch = patch
        self.proj = nn.Conv3d(
            3,
            embed_dim,
            kernel_size=(tubelet, patch, patch),
            stride=(tubelet, patch, patch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class EncoderAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, rope_grid: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.rope_grid = rope_grid
        axis = int(2 * ((self.head_dim // 3) // 2))
        self.d_dim = axis
        self.h_dim = axis
        self.w_dim = axis

    def forward(self, x: torch.Tensor, t_tokens: int, height: int, width: int, keep: torch.Tensor | None):
        b, n, _ = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self._rope(q, k, t_tokens, height, width)
        mask = None
        if keep is not None:
            mask = keep[:, None, None, :]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.proj(out)

    def _rope(self, q, k, t_tokens, height, width):
        device = q.device
        ids = torch.arange(t_tokens * height * width, device=device)
        frame = torch.div(ids, height * width, rounding_mode="floor")
        rem = ids - frame * height * width
        y = torch.div(rem, width, rounding_mode="floor")
        x = rem - y * width
        if height > 1:
            y = y.float() * ((self.rope_grid - 1) / (height - 1))
        else:
            y = y.float()
        if width > 1:
            x = x.float() * ((self.rope_grid - 1) / (width - 1))
        else:
            x = x.float()
        frame = frame.float()
        q, k = self._rotate_axis(q, k, frame, 0, self.d_dim)
        q, k = self._rotate_axis(q, k, y, self.d_dim, self.d_dim + self.h_dim)
        q, k = self._rotate_axis(q, k, x, self.d_dim + self.h_dim, self.d_dim + self.h_dim + self.w_dim)
        return q, k

    @staticmethod
    def _rotate_axis(q, k, pos, start, end):
        if start == end:
            return q, k
        q = q.clone()
        k = k.clone()
        q[..., start:end] = _vjepa_rotate(q[..., start:end], pos)
        k[..., start:end] = _vjepa_rotate(k[..., start:end], pos)
        return q, k


class EncoderMLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, rope_grid: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = EncoderAttention(dim, num_heads, rope_grid)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = EncoderMLP(dim, mlp_ratio)

    def forward(self, x, t_tokens, height, width, keep):
        x = x + self.attn(self.norm1(x), t_tokens, height, width, keep)
        x = x + self.mlp(self.norm2(x))
        if keep is not None:
            x = x * keep.unsqueeze(-1).to(dtype=x.dtype)
        return x


class VJEPAEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        patch_size: int = 16,
        tubelet_size: int = 2,
        rope_grid: int = 16,
        hierarchical: bool | None = None,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.depth = depth
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.patch_embed = PatchEmbed3d(tubelet_size, patch_size, embed_dim)
        self.image_patch_embed = PatchEmbed3d(1, patch_size, embed_dim)
        self.modality_embed = nn.Parameter(torch.zeros(2, embed_dim))
        self.blocks = nn.ModuleList(
            [EncoderBlock(embed_dim, num_heads, mlp_ratio, rope_grid) for _ in range(depth)]
        )
        self.hierarchical = depth == 24 if hierarchical is None else hierarchical
        if self.hierarchical:
            if depth < 24:
                raise ValueError("hierarchical V-JEPA norms require the 24-block ViT-L")
            self.norms_block = nn.ModuleList([nn.LayerNorm(embed_dim, eps=1e-6) for _ in range(4)])
            self.out_layers = {layer: i for i, layer in enumerate(HIERARCHICAL_LAYERS)}
        else:
            self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
        self.register_buffer("pixel_mean", mean, persistent=False)
        self.register_buffer("pixel_std", std, persistent=False)

    @classmethod
    def vit_l(cls, rope_grid: int = 16) -> "VJEPAEncoder":
        return cls(rope_grid=rope_grid, hierarchical=True)

    @classmethod
    def tiny(cls, **overrides) -> "VJEPAEncoder":
        cfg = dict(embed_dim=32, depth=2, num_heads=4, patch_size=16, tubelet_size=2, hierarchical=False)
        cfg.update(overrides)
        return cls(**cfg)

    def forward(self, images: torch.Tensor, frame_valid: torch.Tensor | None = None) -> torch.Tensor:
        """Encode ``[B, 3, T, H, W]`` pixels in ``[0, 1]`` to ``[B, T_out, H_out, W_out, D]``."""
        if images.ndim != 5:
            raise ValueError(f"expected [B, 3, T, H, W], got {tuple(images.shape)}")
        images = (images - self.pixel_mean) / self.pixel_std
        _, _, frames, height, width = images.shape
        if frames == 1:
            tokens = self.image_patch_embed(images)
            modality = 0
        else:
            if frames % self.tubelet_size != 0:
                raise ValueError(f"frame count {frames} is not divisible by tubelet {self.tubelet_size}")
            tokens = self.patch_embed(images)
            modality = 1
        b, dim, t_out, h_out, w_out = tokens.shape
        x = tokens.flatten(2).transpose(1, 2).contiguous()
        x = x + self.modality_embed[modality]
        keep = self._token_keep(frame_valid, frames, t_out, h_out, w_out, b, x.device)
        collected = []
        for index, block in enumerate(self.blocks):
            x = block(x, t_out, h_out, w_out, keep)
            if self.hierarchical and index in self.out_layers:
                collected.append(self.norms_block[self.out_layers[index]](x))
        out = collected[-1] if self.hierarchical else self.norm(x)
        return out.view(b, t_out, h_out, w_out, dim)

    def _token_keep(self, frame_valid, frames, t_out, h_out, w_out, batch, device):
        if frame_valid is None:
            return None
        if frames == 1:
            valid_t = frame_valid
        else:
            valid_t = frame_valid[:, 0::2] & frame_valid[:, 1::2]
        if valid_t.shape[-1] != t_out:
            raise ValueError(f"frame_valid implies {valid_t.shape[-1]} tubelets, encoder produced {t_out}")
        keep = valid_t[:, :, None, None].expand(batch, t_out, h_out, w_out).reshape(batch, -1)
        return keep

    def load_checkpoint(self, path: str | Path, strict: bool = False) -> tuple[list[str], list[str]]:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        state, source = prepare_vjepa_state(payload)
        missing, unexpected = self.load_state_dict(state, strict=strict)
        missing, unexpected = list(missing), list(unexpected)
        if missing or unexpected:
            raise RuntimeError(
                f"V-JEPA checkpoint {source} does not match this encoder: "
                f"missing={missing[:8]} unexpected={unexpected[:8]}"
            )
        return missing, unexpected


def prepare_vjepa_state(payload: dict) -> tuple[dict[str, torch.Tensor], str]:
    """Return encoder tensors and which checkpoint entry they came from.

    The public distilled checkpoint stores the frozen target under
    ``ema_encoder``. Its image stem is named ``patch_embed_img`` and its
    modality vectors are two separate parameters, which this encoder keeps
    as ``image_patch_embed`` and one ``modality_embed`` of shape ``[2, D]``.
    """
    source = "state_dict"
    tensors = payload
    if isinstance(payload, dict):
        for key in _CHECKPOINT_SOURCES:
            value = payload.get(key)
            if isinstance(value, dict) and any(torch.is_tensor(item) for item in value.values()):
                tensors = value
                source = key
                break
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in tensors.items():
        if not torch.is_tensor(value):
            continue
        name = key
        for prefix in ("module.backbone.", "backbone.", "module.", "encoder."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
                break
        if name.startswith("patch_embed_img."):
            name = "image_patch_embed." + name[len("patch_embed_img.") :]
        cleaned[name] = value
    image = cleaned.pop("img_mod_embed", None)
    video = cleaned.pop("video_mod_embed", None)
    if image is not None and video is not None:
        cleaned["modality_embed"] = torch.stack([image.reshape(-1), video.reshape(-1)], dim=0)
    return cleaned, source


def count_parameters(module: nn.Module) -> int:
    return sum(param.numel() for param in module.parameters())


def apply_encoder_lora(encoder: VJEPAEncoder, last_n_blocks: int, rank: int = 16, alpha: int = 32, dropout: float = 0.05) -> VJEPAEncoder:
    """Adapt fused QKV, attention output, and both MLP projections with LoRA.

    Uses ``peft`` when it is installed. The fallback wraps the same Linear
    modules and names the factors ``lora_A`` / ``lora_B`` so the optimizer
    group (weight decay 0, its own LR) is unchanged.
    """
    if last_n_blocks <= 0:
        encoder.requires_grad_(False)
        return encoder
    depth = len(encoder.blocks)
    if last_n_blocks > depth:
        raise ValueError(f"last_n_blocks {last_n_blocks} exceeds depth {depth}")
    start = depth - last_n_blocks
    pattern = rf"^blocks\.({'|'.join(str(i) for i in range(start, depth))})\.(attn\.qkv|attn\.proj|mlp\.fc1|mlp\.fc2)$"
    encoder.requires_grad_(False)
    try:
        from peft import LoraConfig, inject_adapter_in_model

        config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=pattern,
            bias="none",
        )
        inject_adapter_in_model(config, encoder)
    except ImportError:
        _inject_lora_fallback(encoder, start, rank, alpha, dropout)
    for name, param in encoder.named_parameters():
        param.requires_grad = "lora_" in name
    return encoder


class _LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: int, dropout: float):
        super().__init__()
        self.base = base
        self.scaling = alpha / rank
        self.drop = nn.Dropout(dropout)
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_B(self.lora_A(self.drop(x))) * self.scaling


def _inject_lora_fallback(encoder: nn.Module, start_block: int, rank: int, alpha: int, dropout: float) -> None:
    import re

    pattern = re.compile(rf"^blocks\.(\d+)\.(attn\.qkv|attn\.proj|mlp\.fc1|mlp\.fc2)$")
    for name, module in list(encoder.named_modules()):
        match = pattern.match(name)
        if match is None or int(match.group(1)) < start_block:
            continue
        if not isinstance(module, nn.Linear):
            continue
        parent_name, _, child = name.rpartition(".")
        parent = encoder.get_submodule(parent_name)
        setattr(parent, child, _LoRALinear(module, rank, alpha, dropout))
