"""Percentile action/state normalization and V-JEPA feature moments.

Action and proprioception use per-channel 1st/99th percentiles mapped onto
``[-1, 1]``. RoboTwin additionally clamps the normalized values to ±5, and
maps gripper channels from ``[0, 1]`` with ``2x - 1`` instead of the
percentile. Feature mean/std are stored with the checkpoint and are not
recomputed at evaluation. Motion targets keep their own statistics, and
LIBERO pooled futures can carry a per-camera scalar variance correction.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class PercentileNormalizer:
    def __init__(
        self,
        low: np.ndarray,
        high: np.ndarray,
        gripper_indices: list[int] | None = None,
        clamp: float | None = None,
    ):
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)
        self.gripper_indices = [int(index) for index in (gripper_indices or [])]
        self.clamp = None if clamp is None else float(clamp)

    def normalize(self, values: np.ndarray | torch.Tensor):
        array, restore = _as_numpy(values)
        span = np.maximum(self.high - self.low, 1e-6)
        out = 2.0 * (array - self.low) / span - 1.0
        for index in self.gripper_indices:
            out[..., index] = array[..., index] * 2.0 - 1.0
        if self.clamp is not None:
            out = np.clip(out, -self.clamp, self.clamp)
        return restore(out)

    def unnormalize(self, values: np.ndarray | torch.Tensor):
        array, restore = _as_numpy(values)
        if self.clamp is not None:
            array = np.clip(array, -self.clamp, self.clamp)
        span = np.maximum(self.high - self.low, 1e-6)
        out = (array + 1.0) / 2.0 * span + self.low
        for index in self.gripper_indices:
            out[..., index] = (array[..., index] + 1.0) / 2.0
        return restore(out)

    def state_dict(self) -> dict:
        return {
            "low": self.low.tolist(),
            "high": self.high.tolist(),
            "gripper_indices": self.gripper_indices,
            "clamp": self.clamp,
        }

    @classmethod
    def from_state(cls, state: dict) -> "PercentileNormalizer":
        return cls(
            low=np.asarray(state["low"], dtype=np.float32),
            high=np.asarray(state["high"], dtype=np.float32),
            gripper_indices=list(state.get("gripper_indices") or []),
            clamp=state.get("clamp"),
        )


def fit_percentile(values: np.ndarray, gripper_indices: list[int] | None = None, clamp: float | None = None) -> PercentileNormalizer:
    data = np.asarray(values, dtype=np.float32).reshape(-1, values.shape[-1])
    low = np.percentile(data, 1, axis=0).astype(np.float32)
    high = np.percentile(data, 99, axis=0).astype(np.float32)
    # Percentiles collapse on a constant channel; keep a unit span so the map is defined.
    high = np.where(np.abs(high - low) < 1e-6, low + 1.0, high)
    return PercentileNormalizer(low, high, gripper_indices, clamp)


class ChannelMoments:
    def __init__(self, dim: int):
        self.sum = np.zeros(dim, dtype=np.float64)
        self.sumsq = np.zeros(dim, dtype=np.float64)
        self.count = 0

    def update(self, values: np.ndarray) -> None:
        flat = np.asarray(values, dtype=np.float64).reshape(-1, values.shape[-1])
        self.sum += flat.sum(axis=0)
        self.sumsq += np.square(flat).sum(axis=0)
        self.count += int(flat.shape[0])

    def mean_std(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count == 0:
            raise RuntimeError("no samples were accumulated")
        mean = self.sum / self.count
        var = np.maximum(self.sumsq / self.count - np.square(mean), 1e-6)
        return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def variance_scales(normalized_per_camera: list[np.ndarray]) -> np.ndarray:
    """Per-camera scalar that restores unit variance after the global z-score.

    ``normalized_per_camera`` holds up to 64 training windows of pooled future
    features. A scale of 1 leaves the camera unchanged.
    """
    scales = []
    for sample in normalized_per_camera:
        std = float(np.asarray(sample, dtype=np.float64).std())
        scales.append(1.0 if std < 1e-6 else 1.0 / std)
    return np.asarray(scales, dtype=np.float32)


def save_stats(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_stats(path: str | Path) -> dict:
    # Stats are written by this repo and contain NumPy arrays. PyTorch 2.6
    # refuses those unless weights_only is turned off.
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def _as_numpy(values):
    if isinstance(values, torch.Tensor):
        device, dtype = values.device, values.dtype
        return values.detach().float().cpu().numpy(), lambda array: torch.as_tensor(array, device=device, dtype=dtype)
    return np.asarray(values, dtype=np.float32), lambda array: array
