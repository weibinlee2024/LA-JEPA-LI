"""Training windows: 2 observation frames, 16 future frames, 32 actions.

Augmentation runs only while the observation images are being returned.
Future features are read from the store, which was encoded from unmodified
frames. End-of-episode futures and actions stay in the batch and are masked.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

try:
    import h5py
except ImportError:  # hdf5 episodes are optional until the LIBERO export is used
    h5py = None
from torch.utils.data import Dataset

from lajepa.augment import ObservationAugmentor
from lajepa.data.normalize import PercentileNormalizer
from lajepa.features.indices import (
    action_indices,
    action_valid,
    future_grid_valid,
    observation_frame_indices,
)
from lajepa.features.store import FeatureStore


class EpisodeWindowDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        normalizer_action: PercentileNormalizer,
        normalizer_state: PercentileNormalizer,
        feature_store: str | Path | None = None,
        cameras: list[str] | None = None,
        image_size: int = 256,
        horizon: int = 32,
        temporal_encoding: str = "tube_motion",
        augment: bool = False,
        vertical_fov_deg: float = 45.0,
        return_images: bool = True,
        text_embeds: dict[str, np.ndarray] | None = None,
        text_dim: int = 1024,
    ):
        self.manifest_path = Path(manifest)
        payload = json.loads(self.manifest_path.read_text())
        self.episodes = payload["episodes"]
        self.root = self.manifest_path.parent
        self.action_norm = normalizer_action
        self.state_norm = normalizer_state
        self.store = FeatureStore.open(feature_store) if feature_store else None
        self.cameras = list(cameras or [])
        self.image_size = image_size
        self.horizon = horizon
        self.temporal_encoding = temporal_encoding
        self.augmentor = ObservationAugmentor(vertical_fov_deg=vertical_fov_deg) if augment else None
        self.return_images = return_images
        self.text_embeds = text_embeds or {}
        self.text_dim = text_dim
        self.index: list[tuple[int, int]] = []
        for episode_index, episode in enumerate(self.episodes):
            for t in range(int(episode["num_frames"])):
                self.index.append((episode_index, t))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict:
        episode_index, t = self.index[item]
        episode = self.episodes[episode_index]
        arrays = load_episode_arrays(self.root, episode)
        images = arrays["images"]  # [T, C, H, W, 3] uint8
        actions = np.asarray(arrays["actions"], dtype=np.float32)
        states = np.asarray(arrays["states"], dtype=np.float32)
        length = int(images.shape[0])
        encoding = "static" if self.temporal_encoding == "static" else "tube_motion"
        obs_idx = observation_frame_indices(t, encoding)
        # images: [T, C, H, W, 3] -> obs: [C, time, H, W, 3]
        obs_frames = []
        for index in obs_idx:
            cameras = [_resize(images[index, camera], self.image_size) for camera in range(images.shape[1])]
            obs_frames.append(np.stack(cameras, axis=0))
        obs = np.stack(obs_frames, axis=1)
        obs = torch.from_numpy(obs).permute(0, 1, 4, 2, 3).contiguous().float() / 255.0
        if self.augmentor is not None:
            obs = self.augmentor(obs)
        action_idx = action_indices(t, self.horizon)
        valid_actions = action_valid(t, length, self.horizon)
        action = np.zeros((self.horizon, actions.shape[-1]), dtype=np.float32)
        for step, (index, valid) in enumerate(zip(action_idx, valid_actions)):
            if valid:
                action[step] = actions[index]
        state = states[t]
        sample = {
            "obs_images": obs,
            "actions": torch.from_numpy(self.action_norm.normalize(action)),
            "action_mask": torch.tensor(valid_actions, dtype=torch.bool),
            "proprio": torch.from_numpy(self.state_norm.normalize(state)),
            "instruction": str(episode.get("instruction", "")),
            "fut_grid_mask": _camera_grid_mask(
                future_grid_valid(t, length, self.temporal_encoding), images.shape[1]
            ),
        }
        if self.store is not None:
            episode_id = episode["id"]
            futures = []
            obs_features = []
            camera_names = self.cameras or list(self.store.meta["cameras"])
            for camera_index, camera in enumerate(camera_names):
                futures.append(self.store.read_future(episode_id, camera, t))
                obs_features.append(self.store.read_obs(episode_id, camera, t))
            sample["fut_features"] = torch.from_numpy(np.stack(futures, axis=0))
            sample["obs_features"] = torch.from_numpy(np.stack(obs_features, axis=0))
            sample["fut_grid_mask"] = _camera_grid_mask(
                self.store.read_future_mask(episode_id, t).astype(bool).tolist(), len(camera_names)
            )
        text = self.text_embeds.get(sample["instruction"])
        if text is not None:
            sample["text_embeds"] = torch.from_numpy(np.asarray(text, dtype=np.float32))
            sample["text_mask"] = torch.ones(text.shape[0], dtype=torch.bool)
        if not self.return_images:
            sample.pop("obs_images")
        return sample


_EPISODE_CACHE: dict[tuple, dict] = {}
_EPISODE_ORDER: list[tuple] = []
_EPISODE_CACHE_LIMIT = 256


def load_episode_arrays(root: Path, episode: dict) -> dict:
    """Load ``images`` ``[T, cameras, H, W, 3]``, ``actions`` and ``states``.

    LIBERO HDF5 stores Robosuite frames upside down. They are flipped here so
    training and the closed-loop evaluator see the same orientation. ``states``
    is the 8-D proprioception used by the paper: end-effector position, axis-angle
    orientation, and the two gripper fingers.
    """
    path = Path(episode["path"])
    if not path.is_absolute():
        path = Path(root) / path
    demo = str(episode.get("demo", ""))
    key = (str(path), demo)
    cached = _EPISODE_CACHE.get(key)
    if cached is not None:
        return cached
    if path.suffix in {".hdf5", ".h5"}:
        loaded = _load_libero_hdf5(path, demo)
    else:
        arrays = np.load(path)
        loaded = {
            "images": arrays["images"],
            "actions": np.asarray(arrays["actions"], dtype=np.float32),
            "states": np.asarray(arrays["states"], dtype=np.float32),
        }
    _EPISODE_CACHE[key] = loaded
    _EPISODE_ORDER.append(key)
    while len(_EPISODE_ORDER) > _EPISODE_CACHE_LIMIT:
        _EPISODE_CACHE.pop(_EPISODE_ORDER.pop(0), None)
    return loaded


def _load_libero_hdf5(path: Path, demo: str) -> dict:
    if h5py is None:
        raise ImportError("h5py is required to read LIBERO demonstrations")
    with h5py.File(path, "r") as handle:
        group = handle["data"][demo]
        obs = group["obs"]
        agent = np.ascontiguousarray(obs["agentview_rgb"][()][:, ::-1, ::-1])
        wrist = np.ascontiguousarray(obs["eye_in_hand_rgb"][()][:, ::-1, ::-1])
        images = np.stack([agent, wrist], axis=1)
        actions = np.asarray(group["actions"], dtype=np.float32)
        state = np.concatenate(
            [
                np.asarray(obs["ee_pos"], dtype=np.float32),
                np.asarray(obs["ee_ori"], dtype=np.float32),
                np.asarray(obs["gripper_states"], dtype=np.float32),
            ],
            axis=-1,
        )
    return {"images": images, "actions": actions, "states": state}


def collate_windows(samples: list[dict]) -> dict:
    batch: dict = {}
    tensor_keys = [
        "obs_images",
        "obs_features",
        "fut_features",
        "fut_grid_mask",
        "actions",
        "action_mask",
        "proprio",
    ]
    for key in tensor_keys:
        if key in samples[0]:
            batch[key] = torch.stack([sample[key] for sample in samples], dim=0)
    if "text_embeds" in samples[0]:
        length = max(int(sample["text_embeds"].shape[0]) for sample in samples)
        dim = int(samples[0]["text_embeds"].shape[-1])
        embeds = torch.zeros(len(samples), length, dim)
        mask = torch.zeros(len(samples), length, dtype=torch.bool)
        for row, sample in enumerate(samples):
            n = sample["text_embeds"].shape[0]
            embeds[row, :n] = sample["text_embeds"]
            mask[row, :n] = sample["text_mask"]
        batch["text_embeds"] = embeds
        batch["text_mask"] = mask
    else:
        batch["instruction"] = [sample["instruction"] for sample in samples]
    return batch


def _camera_grid_mask(temporal_valid: list[bool], num_cameras: int) -> torch.Tensor:
    mask = torch.tensor(temporal_valid, dtype=torch.bool)
    return mask.unsqueeze(0).expand(num_cameras, -1).contiguous()


def _resize(image: np.ndarray, size: int) -> np.ndarray:
    """``image`` is ``[H, W, 3]`` uint8. Bilinear resize to ``size x size``."""
    if image.shape[0] == size and image.shape[1] == size:
        return image
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    resized = torch.nn.functional.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
    return resized[0].permute(1, 2, 0).round().clamp(0, 255).byte().numpy()


def crosses_joint_wrap(joints: np.ndarray, limit: float = np.pi) -> bool:
    """True when a planned joint path jumps across ±π.

    A continuous joint angle changes by at most π between control steps; a
    larger step is the wrap the revised RoboTwin pipeline excludes.
    """
    if joints.shape[0] < 2:
        return False
    delta = np.diff(np.asarray(joints, dtype=np.float64), axis=0)
    return bool(np.any(np.abs(delta) > limit))
