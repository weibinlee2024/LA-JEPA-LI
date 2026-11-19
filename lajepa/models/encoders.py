"""Observation encoder, frozen FLAN-T5, and the shared condition projection.

The condition sequence is built once and read by both DiTs through their own
cross-attention. Proprioception is one learned token. Text tokens come from a
frozen FLAN-T5-large encoder, or from a precomputed cache of the same vectors.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lajepa.features.vjepa_encoder import VJEPAEncoder, apply_encoder_lora


class ObservationEncoder(nn.Module):
    def __init__(self, backbone: VJEPAEncoder, lora_last_n: int = 0, lora_rank: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05):
        super().__init__()
        self.backbone = backbone
        if lora_last_n > 0:
            apply_encoder_lora(backbone, lora_last_n, lora_rank, lora_alpha, lora_dropout)
        else:
            backbone.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """``images`` is ``[B, cameras, time, 3, H, W]`` in ``[0, 1]``."""
        b, cameras, time, channels, height, width = images.shape
        flat = images.reshape(b * cameras, time, channels, height, width).permute(0, 2, 1, 3, 4).contiguous()
        features = self.backbone(flat)
        return features.reshape(b, cameras, *features.shape[1:])


class FrozenT5(nn.Module):
    def __init__(self, model_name: str = "google/flan-t5-large", max_length: int = 128):
        super().__init__()
        from transformers import AutoTokenizer, T5EncoderModel

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        self.encoder.eval()
        self.encoder.requires_grad_(False)
        self.max_length = max_length
        self.output_dim = int(self.encoder.config.d_model)

    def train(self, mode: bool = True):
        # The language encoder stays frozen and in eval, including dropout.
        return super().train(False)

    @torch.no_grad()
    def forward(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(self.encoder.parameters()).device
        tokens = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        hidden = self.encoder(**tokens).last_hidden_state
        return hidden, tokens["attention_mask"].bool()


def encode_instructions(batch: dict, encoder: FrozenT5) -> dict:
    """Fill ``text_embeds`` from frozen FLAN-T5 when the batch only has strings.

    A precomputed cache of the same vectors is left untouched. The paper
    conditions both DiTs on the frozen language encoder, so training cannot
    enter the forward with raw instruction text.
    """
    if "text_embeds" in batch:
        return batch
    instructions = batch.get("instruction")
    if not isinstance(instructions, (list, tuple)) or len(instructions) == 0:
        raise KeyError("batch needs text_embeds or instruction strings")
    hidden, mask = encoder([str(text) for text in instructions])
    if int(hidden.shape[-1]) != int(encoder.output_dim):
        raise RuntimeError(f"FLAN-T5 width {hidden.shape[-1]} != declared {encoder.output_dim}")
    batch["text_embeds"] = hidden
    batch["text_mask"] = mask
    return batch


class ConditionProjector(nn.Module):
    def __init__(self, text_dim: int, proprio_dim: int, hidden_dim: int):
        super().__init__()
        self.text = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.proprio = nn.Sequential(
            nn.Linear(proprio_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        text: torch.Tensor,
        text_mask: torch.Tensor,
        proprio: torch.Tensor,
        text_keep: torch.Tensor | None,
        proprio_keep: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_tokens = self.text(text)
        proprio_token = self.proprio(proprio).unsqueeze(1)
        cond = torch.cat([text_tokens, proprio_token], dim=1)
        keep_text = text_mask
        if text_keep is not None:
            keep_text = keep_text & text_keep.unsqueeze(1)
        keep_proprio = torch.ones(text.shape[0], 1, dtype=torch.bool, device=text.device)
        if proprio_keep is not None:
            keep_proprio = proprio_keep.unsqueeze(1)
        return cond, torch.cat([keep_text, keep_proprio], dim=1)
