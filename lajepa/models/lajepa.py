"""La-JEPA policy: World DiT and Action DiT with tripartite attention.

Training receives clean observation features, noised future features, and a
noised action chunk, and regresses both velocity fields. Inference caches
observation keys and values at ``τ_ctx = 0`` and integrates only the action
field. Future values never affect the action forward pass.
"""

from __future__ import annotations

from torch.utils.checkpoint import checkpoint

import torch
import torch.nn as nn

from lajepa.flow_matching import euler_step, interpolate, masked_mse, sample_noise, sample_tau, velocity_target
from lajepa.models.action_dit import ActionDiT
from lajepa.models.encoders import ConditionProjector, ObservationEncoder
from lajepa.models.world_dit import CameraSlice, WorldDiT


def _sliced_mse(preds: list[torch.Tensor], targets: list[torch.Tensor], slices: list[CameraSlice]) -> torch.Tensor:
    numerator = preds[0].new_zeros(())
    denominator = preds[0].new_zeros(())
    for pred, target, item in zip(preds, targets, slices):
        error = (pred - target).pow(2)
        weight = item.keep.to(dtype=error.dtype).unsqueeze(-1)
        numerator = numerator + (error * weight).sum()
        denominator = denominator + weight.sum() * pred.shape[-1]
    if float(denominator.detach()) == 0.0:
        return preds[0].sum() * 0.0
    return numerator / denominator


class FeatureNorm(nn.Module):
    """Per-channel z-score. Future tokens also get a per-camera scalar."""

    def __init__(self, feature_dim: int, num_cameras: int):
        super().__init__()
        self.register_buffer("obs_mean", torch.zeros(feature_dim))
        self.register_buffer("obs_std", torch.ones(feature_dim))
        self.register_buffer("fut_mean", torch.zeros(feature_dim))
        self.register_buffer("fut_std", torch.ones(feature_dim))
        self.register_buffer("fut_cam_scale", torch.ones(num_cameras))

    def load_stats(self, stats: dict) -> None:
        self._copy("obs_mean", stats["obs_mean"])
        self._copy("obs_std", stats["obs_std"])
        self._copy("fut_mean", stats["fut_mean"])
        self._copy("fut_std", stats["fut_std"])
        if "fut_cam_scale" in stats:
            self._copy("fut_cam_scale", stats["fut_cam_scale"])

    def _copy(self, name: str, value) -> None:
        tensor = torch.as_tensor(value, dtype=torch.float32)
        getattr(self, name).copy_(tensor)

    def norm_obs(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.obs_mean) / self.obs_std.clamp_min(1e-6)

    def norm_fut(self, features: torch.Tensor) -> torch.Tensor:
        scaled = (features - self.fut_mean) / self.fut_std.clamp_min(1e-6)
        view = self.fut_cam_scale.view(1, -1, *([1] * (features.ndim - 2)))
        return scaled * view


class LaJEPA(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        feature_dim: int = 1024,
        text_dim: int = 1024,
        action_dim: int = 7,
        proprio_dim: int = 8,
        horizon: int = 32,
        num_cameras: int = 2,
        reduction: str = "mean_pool",
        factors: list[int] | None = None,
        predict_future: bool = True,
        future_loss_weight: float = 1.0,
        action_loss_weight: float = 1.0,
        gradient_checkpointing: bool = False,
        obs_encoder: ObservationEncoder | None = None,
    ):
        super().__init__()
        if factors is None:
            factors = [2] * num_cameras
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.action_dim = action_dim
        self.horizon = horizon
        self.predict_future = predict_future
        self.future_loss_weight = future_loss_weight
        self.action_loss_weight = action_loss_weight
        self.gradient_checkpointing = gradient_checkpointing
        self.feat_norm = FeatureNorm(feature_dim, num_cameras)
        self.condition = ConditionProjector(text_dim, proprio_dim, hidden_dim)
        self.world = WorldDiT(
            hidden_dim, depth, num_heads, mlp_ratio, feature_dim, num_cameras, factors, reduction
        )
        self.action = ActionDiT(hidden_dim, depth, num_heads, mlp_ratio, action_dim, horizon)
        if len(self.world.blocks) != len(self.action.blocks):
            raise ValueError("World and Action DiTs must have the same depth")
        self.obs_encoder = obs_encoder

    def trainable_count(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def forward(self, batch: dict, tau_f: torch.Tensor | None = None, tau_a: torch.Tensor | None = None) -> dict:
        obs = self._normalize_obs(self._observation_features(batch))
        batch_size = obs.shape[0]
        device = obs.device
        if tau_a is None:
            tau_a = sample_tau(batch_size, device)
        cond, cond_keep = self._condition(batch)
        cam_keep = batch.get("cam_keep")
        obs_slices, obs_keep, obs_cos, obs_sin = self.world.pack(obs, t0=0, cam_keep=cam_keep, grid_mask=None)
        h_obs = self.world.project(obs_slices, self.world.obs_type)

        future_targets: list[torch.Tensor] = []
        future_slices: list[CameraSlice] = []
        if self.predict_future:
            if tau_f is None:
                tau_f = sample_tau(batch_size, device)
            future = self.feat_norm.norm_fut(batch["fut_features"])
            future_slices, fut_keep, fut_cos, fut_sin = self.world.pack(
                future, t0=1, cam_keep=cam_keep, grid_mask=batch.get("fut_grid_mask")
            )
            for item in future_slices:
                noise = sample_noise(item.flat.shape, device, dtype=item.flat.dtype)
                future_targets.append(velocity_target(item.flat, noise))
                item.flat = interpolate(item.flat, noise, tau_f)
            h_fut = self.world.project(future_slices, self.world.fut_type)
        else:
            tau_f = obs.new_zeros(batch_size)
            h_fut, fut_keep, fut_cos, fut_sin = self._empty_future(h_obs)
            future_slices = []

        actions = batch["actions"]
        action_noise = sample_noise(actions.shape, device, dtype=actions.dtype)
        action_target = velocity_target(actions, action_noise)
        noisy_actions = interpolate(actions, action_noise, tau_a)
        h_act = self.action.embed(noisy_actions)
        act_keep = batch.get("action_mask")
        if act_keep is None:
            act_keep = torch.ones(actions.shape[:2], dtype=torch.bool, device=device)

        t_world = self.world.time(tau_f)
        t_action = self.action.time(tau_a)
        h_obs, h_fut, h_act = self._layers(
            h_obs, h_fut, h_act, cond, cond_keep, t_world, t_action,
            obs_cos, obs_sin, fut_cos, fut_sin, obs_keep, fut_keep, act_keep,
        )
        action_pred = self.action.head(h_act, t_action)
        action_loss = masked_mse(action_pred, action_target, act_keep)
        if self.predict_future and future_slices:
            future_pred = self.world.predict(h_fut, t_world, future_slices)
            future_loss = _sliced_mse(future_pred, future_targets, future_slices)
        else:
            future_loss = action_pred.sum() * 0.0
        loss = self.action_loss_weight * action_loss + self.future_loss_weight * future_loss
        return {
            "loss": loss,
            "action_loss": action_loss.detach(),
            "future_loss": future_loss.detach(),
        }

    @torch.no_grad()
    def sample_actions(
        self,
        batch: dict,
        steps: int = 10,
        tau_ctx: float = 0.0,
        noise_scale: float = 1.0,
        clamp: float | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Euler integration of the action field. Future tokens are absent."""
        cache, cond, cond_keep, obs_keep = self.cache_observation(batch, tau_ctx)
        batch_size = obs_keep.shape[0]
        device = obs_keep.device
        actions = torch.randn(
            batch_size, self.horizon, self.action_dim, device=device, generator=generator
        ) * noise_scale
        dt = 1.0 / steps
        for index in range(steps):
            tau = torch.full((batch_size,), index * dt, device=device)
            velocity = self.action_velocity(actions, tau, cache, cond, cond_keep, obs_keep, batch.get("action_mask"))
            actions = euler_step(actions, velocity, dt)
        if clamp is not None:
            actions = actions.clamp(-clamp, clamp)
        return actions

    def cache_observation(self, batch: dict, tau_ctx: float = 0.0):
        obs = self._normalize_obs(self._observation_features(batch))
        cond, cond_keep = self._condition(batch)
        cam_keep = batch.get("cam_keep")
        slices, obs_keep, obs_cos, obs_sin = self.world.pack(obs, t0=0, cam_keep=cam_keep, grid_mask=None)
        h_obs = self.world.project(slices, self.world.obs_type)
        h_fut, fut_keep, fut_cos, fut_sin = self._empty_future(h_obs)
        tau = obs.new_full((obs.shape[0],), float(tau_ctx))
        t_world = self.world.time(tau)
        cache = []
        for block in self.world.blocks:
            h_obs, h_fut, k_obs, v_obs = block(
                h_obs, h_fut, cond, cond_keep, t_world, obs_cos, obs_sin, fut_cos, fut_sin, obs_keep, fut_keep
            )
            cache.append((k_obs, v_obs))
        return cache, cond, cond_keep, obs_keep

    def action_velocity(self, noisy_actions, tau, cache, cond, cond_keep, obs_keep, action_mask):
        h_act = self.action.embed(noisy_actions)
        t_action = self.action.time(tau)
        cos, sin = self.action.rope(noisy_actions.device, h_act.dtype)
        act_keep = action_mask
        if act_keep is None:
            act_keep = torch.ones(noisy_actions.shape[:2], dtype=torch.bool, device=noisy_actions.device)
        for block, (k_obs, v_obs) in zip(self.action.blocks, cache):
            h_act = block(h_act, k_obs, v_obs, cond, cond_keep, t_action, cos, sin, obs_keep, act_keep)
        return self.action.head(h_act, t_action)

    def _layers(
        self, h_obs, h_fut, h_act, cond, cond_keep, t_world, t_action,
        obs_cos, obs_sin, fut_cos, fut_sin, obs_keep, fut_keep, act_keep,
    ):
        act_cos, act_sin = self.action.rope(h_act.device, h_act.dtype)
        for index, (world_block, action_block) in enumerate(zip(self.world.blocks, self.action.blocks)):
            if self.gradient_checkpointing and self.training:
                h_obs, h_fut, h_act = checkpoint(
                    self._checkpoint_layer,
                    h_obs, h_fut, h_act, cond, cond_keep, t_world, t_action,
                    obs_cos, obs_sin, fut_cos, fut_sin, act_cos, act_sin,
                    obs_keep, fut_keep, act_keep, torch.tensor(index, device=h_obs.device),
                    use_reentrant=False,
                )
            else:
                h_obs, h_fut, k_obs, v_obs = world_block(
                    h_obs, h_fut, cond, cond_keep, t_world, obs_cos, obs_sin, fut_cos, fut_sin, obs_keep, fut_keep
                )
                h_act = action_block(
                    h_act, k_obs, v_obs, cond, cond_keep, t_action, act_cos, act_sin, obs_keep, act_keep
                )
        return h_obs, h_fut, h_act

    def _checkpoint_layer(
        self, h_obs, h_fut, h_act, cond, cond_keep, t_world, t_action,
        obs_cos, obs_sin, fut_cos, fut_sin, act_cos, act_sin, obs_keep, fut_keep, act_keep, index,
    ):
        layer = int(index.item())
        h_obs, h_fut, k_obs, v_obs = self.world.blocks[layer](
            h_obs, h_fut, cond, cond_keep, t_world, obs_cos, obs_sin, fut_cos, fut_sin, obs_keep, fut_keep
        )
        h_act = self.action.blocks[layer](
            h_act, k_obs, v_obs, cond, cond_keep, t_action, act_cos, act_sin, obs_keep, act_keep
        )
        return h_obs, h_fut, h_act

    def _observation_features(self, batch: dict) -> torch.Tensor:
        images = batch.get("obs_images")
        if images is not None:
            if self.obs_encoder is None:
                raise RuntimeError("obs_images were passed but the model has no observation encoder")
            return self.obs_encoder(images)
        features = batch.get("obs_features")
        if features is None:
            raise KeyError("batch needs obs_images or obs_features")
        return features

    def _normalize_obs(self, features: torch.Tensor) -> torch.Tensor:
        return self.feat_norm.norm_obs(features)

    def _condition(self, batch: dict):
        if "text_embeds" not in batch:
            raise KeyError("batch needs text_embeds; encode instructions before the forward")
        return self.condition(
            batch["text_embeds"],
            batch["text_mask"],
            batch["proprio"],
            batch.get("text_keep"),
            batch.get("proprio_keep"),
        )

    def _empty_future(self, h_obs: torch.Tensor):
        b, _, hidden = h_obs.shape
        empty = h_obs.new_zeros(b, 0, hidden)
        cos = h_obs.new_zeros(0, self.world.head_dim)
        keep = torch.zeros(b, 0, dtype=torch.bool, device=h_obs.device)
        return empty, keep, cos, cos
