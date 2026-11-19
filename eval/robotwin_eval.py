"""RoboTwin 2.0 policy entry.

The simulator calls ``get_model``, ``eval``, and ``reset_model``. Each replan
encodes the head and both wrist cameras at 256 and executes 24 actions before
asking for a new observation. Instructions are the episode's own RoboTwin
text. Normalized actions are clamped to ±5 before they are mapped back, which
is the RoboTwin training clamp.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import torch

from lajepa.inference import LaJEPAPolicy

CAMERA_KEYS = (
    ("head_camera", "cam_high"),
    ("left_camera", "cam_left_wrist"),
    ("right_camera", "cam_right_wrist"),
)


def _rgb(observation: dict, key: str) -> np.ndarray:
    camera = observation["observation"][key]
    return np.asarray(camera["rgb"])


def _to_tensor(image: np.ndarray, size: int = 256) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return torch.nn.functional.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)[0]


class RoboTwinLaJEPA:
    def __init__(self, policy: LaJEPAPolicy, image_size: int = 256):
        self.policy = policy
        self.image_size = image_size
        self.pending: deque[np.ndarray] = deque()
        self.prev: dict[str, torch.Tensor] | None = None

    def reset(self) -> None:
        self.pending.clear()
        self.prev = None
        self.policy.reset()

    def _clip(self, observation: dict) -> torch.Tensor:
        current = {name: _to_tensor(_rgb(observation, key), self.image_size) for key, name in CAMERA_KEYS}
        if self.prev is None:
            self.prev = current
        frames = []
        for _, name in CAMERA_KEYS:
            frames.append(torch.stack([self.prev[name], current[name]], dim=0))
        self.prev = current
        return torch.stack(frames, dim=0)

    def infer(self, observation: dict, instruction: str) -> None:
        clip = self._clip(observation)
        state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        text, mask = _encode(self.policy, instruction)
        chunk = self.policy.predict_chunk(clip, state, text, mask)
        for action in chunk[: self.policy.execute_steps]:
            self.pending.append(np.asarray(action, dtype=np.float32))

    def step(self, task_env, observation: dict | None) -> None:
        if not self.pending:
            if observation is None:
                raise RuntimeError("an observation is required when the action queue is empty")
            instruction = task_env.get_instruction()
            self.infer(observation, instruction)
        action = self.pending.popleft()
        task_env.take_action(action, action_type="qpos")


def _encode(policy: LaJEPAPolicy, instruction: str):
    from lajepa.models.encoders import FrozenT5

    encoder = getattr(policy, "_t5", None)
    if encoder is None:
        encoder = FrozenT5().to(policy.device)
        policy._t5 = encoder
    hidden, mask = encoder([instruction])
    return hidden[0].cpu(), mask[0].cpu()


def get_model(usr_args: dict[str, Any]) -> RoboTwinLaJEPA:
    checkpoint = usr_args.get("ckpt_setting") or usr_args.get("checkpoint")
    device = str(usr_args.get("device", "cuda"))
    policy = LaJEPAPolicy.from_checkpoint(checkpoint, device=device, execute_steps=int(usr_args.get("replan_steps", 24)))
    policy.clamp = 5.0
    return RoboTwinLaJEPA(policy)


def eval(task_env, model: RoboTwinLaJEPA, observation):
    model.step(task_env, observation)


def reset_model(model: RoboTwinLaJEPA) -> None:
    model.reset()
