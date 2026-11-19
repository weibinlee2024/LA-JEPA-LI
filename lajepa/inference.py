"""Deployment policy.

Observation keys and values are computed once at ``τ_ctx = 0``. Each Euler
step runs only the Action DiT. The returned chunk is in the robot's action
space; callers execute the first ``execute_steps`` commands and then replan
(10 on LIBERO, 24 on RoboTwin).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from lajepa.data.normalize import PercentileNormalizer
from lajepa.models.lajepa import LaJEPA
from lajepa.train import build_model


class LaJEPAPolicy:
    def __init__(
        self,
        model: LaJEPA,
        action_normalizer: PercentileNormalizer,
        state_normalizer: PercentileNormalizer,
        steps: int = 10,
        execute_steps: int = 10,
        tau_ctx: float = 0.0,
        noise_scale: float = 1.0,
        clamp: float | None = None,
        device: str = "cpu",
    ):
        self.model = model.to(device).eval()
        self.action_normalizer = action_normalizer
        self.state_normalizer = state_normalizer
        self.steps = steps
        self.execute_steps = execute_steps
        self.tau_ctx = tau_ctx
        self.noise_scale = noise_scale
        self.clamp = clamp
        self.device = torch.device(device)
        self._queue: list[np.ndarray] = []

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "cuda", use_ema: bool = True, **overrides) -> "LaJEPAPolicy":
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(payload["config"])
        model = build_model(cfg)
        state = payload["model"]
        model.load_state_dict(state)
        if use_ema and payload.get("ema"):
            for name, param in model.named_parameters():
                if name in payload["ema"]:
                    param.copy_(payload["ema"][name].to(dtype=param.dtype))
        stats = payload["stats"]
        if "features" in stats:
            model.feat_norm.load_stats(stats["features"])
        policy = cls(
            model=model,
            action_normalizer=PercentileNormalizer.from_state(stats["action"]),
            state_normalizer=PercentileNormalizer.from_state(stats["state"]),
            steps=int(cfg.recipe.num_inference_steps),
            execute_steps=int(cfg.data.get("execute_steps", cfg.recipe.execute_steps)),
            tau_ctx=float(cfg.recipe.tau_ctx),
            noise_scale=float(cfg.recipe.noise_scale),
            clamp=cfg.data.get("normalized_clamp"),
            device=device if torch.cuda.is_available() or device == "cpu" else "cpu",
        )
        for key, value in overrides.items():
            setattr(policy, key, value)
        return policy

    def reset(self) -> None:
        self._queue.clear()

    def act(self, obs_images: torch.Tensor, proprio: np.ndarray, text_embeds: torch.Tensor, text_mask: torch.Tensor | None = None) -> np.ndarray:
        """Return the next robot action, replanning when the queue is empty.

        ``obs_images`` is ``[cameras, time, 3, H, W]`` in ``[0, 1]``.
        """
        if not self._queue:
            chunk = self.predict_chunk(obs_images, proprio, text_embeds, text_mask)
            self._queue.extend(chunk[: self.execute_steps])
        return self._queue.pop(0)

    @torch.no_grad()
    def predict_chunk(self, obs_images, proprio, text_embeds, text_mask=None) -> np.ndarray:
        images = obs_images.unsqueeze(0).to(self.device)
        state = self.state_normalizer.normalize(np.asarray(proprio, dtype=np.float32))
        if text_mask is None:
            text_mask = torch.ones(text_embeds.shape[0], dtype=torch.bool)
        batch = {
            "obs_images": images,
            "proprio": torch.as_tensor(state, device=self.device).unsqueeze(0),
            "text_embeds": text_embeds.unsqueeze(0).to(self.device),
            "text_mask": text_mask.unsqueeze(0).to(self.device),
        }
        normalized = self.model.sample_actions(
            batch,
            steps=self.steps,
            tau_ctx=self.tau_ctx,
            noise_scale=self.noise_scale,
            clamp=None if self.clamp is None else float(self.clamp),
        )
        return np.asarray(self.action_normalizer.unnormalize(normalized[0].float().cpu()), dtype=np.float32)
