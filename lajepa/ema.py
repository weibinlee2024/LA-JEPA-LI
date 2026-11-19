"""EMA of trainable weights, decay 0.9999.

Frozen encoder weights are left untouched. LoRA weights are included, which
is the "EMA encoder" used at evaluation when the observation encoder adapts.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self.shadow: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().float().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        one_minus = 1.0 - self.decay
        for name, param in model.named_parameters():
            if name not in self.shadow:
                continue
            shadow = self.shadow[name]
            shadow.mul_(self.decay).add_(param.detach().float(), alpha=one_minus)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {key: value.clone() for key, value in self.shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.shadow = {key: value.detach().float().clone() for key, value in state.items()}

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.copy_(self.shadow[name].to(dtype=param.dtype, device=param.device))
