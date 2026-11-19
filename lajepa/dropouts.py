"""Camera, proprioception, and text condition dropout.

Camera dropout is the classifier-free pattern, but inference does not apply
guidance. Eligible cameras are dropped independently with probability ``p``.
If that would remove every camera, one eligible camera is kept at random.
On LIBERO both views are eligible, so the expected number of missing views
is 0.19 rather than 0.20. On RoboTwin only the wrists are eligible, so the
head view is always present.
"""

from __future__ import annotations

import torch


def camera_keep_mask(
    batch_size: int,
    num_cameras: int,
    prob: float,
    eligible: list[int] | None = None,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return ``[B, C]`` bool, True where the camera is kept."""
    if num_cameras <= 0:
        raise ValueError("num_cameras must be positive")
    if not 0.0 <= prob <= 1.0:
        raise ValueError(f"camera dropout prob must be in [0, 1], got {prob}")
    keep = torch.ones(batch_size, num_cameras, dtype=torch.bool, device=device)
    if prob == 0.0 or not eligible:
        return keep
    eligible = [int(i) for i in eligible]
    for index in eligible:
        if index < 0 or index >= num_cameras:
            raise ValueError(f"eligible camera {index} outside 0..{num_cameras - 1}")
    drop = torch.rand(batch_size, len(eligible), generator=generator, device=device) < prob
    for column, index in enumerate(eligible):
        keep[:, index] = ~drop[:, column]
    none_left = ~keep.any(dim=1)
    if none_left.any():
        choice = torch.randint(len(eligible), (int(none_left.sum()),), generator=generator, device=device)
        rows = none_left.nonzero(as_tuple=False).flatten()
        for row, pick in zip(rows.tolist(), choice.tolist()):
            keep[row, eligible[pick]] = True
    return keep


def bernoulli_keep(
    batch_size: int,
    prob: float,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """True = keep the condition. ``prob`` is the probability of dropping it."""
    if prob <= 0.0:
        return torch.ones(batch_size, dtype=torch.bool, device=device)
    if prob >= 1.0:
        return torch.zeros(batch_size, dtype=torch.bool, device=device)
    return torch.rand(batch_size, generator=generator, device=device) >= prob


def apply_condition_dropout(
    batch: dict,
    camera_prob: float,
    text_prob: float,
    proprio_prob: float,
    eligible_cameras: list[int] | None,
    generator: torch.Generator | None = None,
) -> dict:
    """Attach ``cam_keep``, ``text_keep``, and ``proprio_keep`` masks to ``batch``.

    Dropped future grids are removed from the loss by the model, which ANDs
    ``cam_keep`` into the future-token mask. This function does not delete
    tensors, so the batch stays rectangular.
    """
    if "actions" in batch:
        batch_size = int(batch["actions"].shape[0])
        device = batch["actions"].device
    elif "fut_features" in batch and batch["fut_features"] is not None:
        batch_size = int(batch["fut_features"].shape[0])
        device = batch["fut_features"].device
    else:
        raise KeyError("batch needs 'actions' or 'fut_features' to infer the batch size")
    num_cameras = int(batch.get("num_cameras", _infer_cameras(batch)))
    batch["cam_keep"] = camera_keep_mask(
        batch_size, num_cameras, camera_prob, eligible_cameras, generator, device
    )
    batch["text_keep"] = bernoulli_keep(batch_size, text_prob, generator, device)
    batch["proprio_keep"] = bernoulli_keep(batch_size, proprio_prob, generator, device)
    return batch


def _infer_cameras(batch: dict) -> int:
    if batch.get("obs_images") is not None:
        return int(batch["obs_images"].shape[1])
    if batch.get("obs_features") is not None:
        return int(batch["obs_features"].shape[1])
    if batch.get("fut_features") is not None:
        return int(batch["fut_features"].shape[1])
    raise KeyError("cannot infer num_cameras")
