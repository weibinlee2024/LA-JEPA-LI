"""Learning-rate schedules from Appendix B.

Policy: 5% linear warmup, cosine decay down to 1% of the peak, then a
constant tail over the last 10% of updates.

LoRA: the same warmup, cosine decay down to 5% of its peak, and the same
final tail at 1% of its peak.
"""

from __future__ import annotations

import math


def _cosine(progress: float, floor: float) -> float:
    # progress 0 → 1, progress 1 → floor
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return floor + (1.0 - floor) * cosine


def lr_scale(
    step: int,
    total_steps: int,
    warmup_ratio: float = 0.05,
    tail_ratio: float = 0.10,
    cosine_floor: float = 0.01,
    tail_floor: float = 0.01,
) -> float:
    """Multiplicative scale in ``[0, 1]`` relative to the group's peak LR."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    step = max(0, min(int(step), total_steps))
    warmup = int(round(total_steps * warmup_ratio))
    tail = int(round(total_steps * tail_ratio))
    tail_start = total_steps - tail
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    if step >= tail_start:
        return tail_floor
    span = max(1, tail_start - warmup)
    progress = (step - warmup) / span
    return _cosine(progress, cosine_floor)


def policy_lr_scale(step: int, total_steps: int) -> float:
    return lr_scale(step, total_steps, cosine_floor=0.01, tail_floor=0.01)


def lora_lr_scale(step: int, total_steps: int) -> float:
    return lr_scale(step, total_steps, cosine_floor=0.05, tail_floor=0.01)
