"""Joint flow matching for future features and actions.

The paper's interpolant runs from noise at τ=0 to data at τ=1::

    x_τ = (1 - τ) ε + τ x
    v*  = x - ε

World and action timesteps are sampled independently. The velocity field is
integrated with Euler steps from 0 to 1 at inference.
"""

from __future__ import annotations

import torch


def sample_tau(batch: int, device: torch.device, generator: torch.Generator | None = None) -> torch.Tensor:
    return torch.rand(batch, device=device, generator=generator)


def sample_noise(
    shape: tuple[int, ...],
    device: torch.device,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return torch.randn(shape, device=device, generator=generator, dtype=dtype)


def interpolate(data: torch.Tensor, noise: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """Broadcast τ over every dimension after the batch axis."""
    while tau.ndim < data.ndim:
        tau = tau.unsqueeze(-1)
    return (1.0 - tau) * noise + tau * data


def velocity_target(data: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    return data - noise


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Mean squared error averaged over supervised scalar elements.

    ``mask`` is bool ``[B, S]`` (True = supervise). An all-false mask yields a
    zero loss that stays connected to ``pred`` so DDP still sees the tensor.
    """
    err = (pred - target).pow(2)
    if mask is None:
        return err.mean()
    weight = mask.to(dtype=err.dtype).unsqueeze(-1)
    denom = weight.sum() * pred.shape[-1]
    if float(denom) == 0.0:
        return pred.sum() * 0.0
    return (err * weight).sum() / denom


def euler_step(x: torch.Tensor, velocity: torch.Tensor, dt: float) -> torch.Tensor:
    return x + dt * velocity
