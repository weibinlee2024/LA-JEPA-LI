"""Fit action/state percentiles and feature moments, then save them with the run.

The file written here is what training and evaluation reload. Do not recompute
these statistics on the evaluation split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lajepa.data.dataset import load_episode_arrays
from lajepa.data.normalize import ChannelMoments, fit_percentile, save_stats, variance_scales
from lajepa.features.store import FeatureStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--feature-store", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gripper-indices", default="")
    parser.add_argument("--clamp", type=float, default=None)
    parser.add_argument("--variance-windows", type=int, default=64)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    root = Path(args.manifest).parent
    actions, states = [], []
    for episode in manifest["episodes"]:
        arrays = load_episode_arrays(root, episode)
        actions.append(np.asarray(arrays["actions"], dtype=np.float32))
        states.append(np.asarray(arrays["states"], dtype=np.float32))
    gripper = [int(part) for part in args.gripper_indices.split(",") if part != ""]
    action_norm = fit_percentile(np.concatenate(actions, axis=0), gripper, args.clamp)
    state_norm = fit_percentile(np.concatenate(states, axis=0), gripper, args.clamp)
    store = FeatureStore.open(args.feature_store)
    dim = int(store.meta["feature_dim"])
    obs_moments, fut_moments = ChannelMoments(dim), ChannelMoments(dim)
    cameras = store.meta["cameras"]
    windows = {camera: [] for camera in cameras}
    for episode_id in store.episode_ids():
        length = store.length(episode_id)
        for t in range(length):
            for camera in cameras:
                obs_moments.update(store.read_obs(episode_id, camera, t))
                future = store.read_future(episode_id, camera, t)
                fut_moments.update(future)
                if len(windows[camera]) < args.variance_windows:
                    windows[camera].append(future)
    obs_mean, obs_std = obs_moments.mean_std()
    fut_mean, fut_std = fut_moments.mean_std()
    per_camera = []
    for camera in cameras:
        stacked = np.concatenate([(sample - fut_mean) / np.maximum(fut_std, 1e-6) for sample in windows[camera]], axis=0)
        per_camera.append(stacked)
    scales = variance_scales(per_camera)
    save_stats(
        args.output,
        {
            "action": action_norm.state_dict(),
            "state": state_norm.state_dict(),
            "features": {
                "obs_mean": obs_mean,
                "obs_std": obs_std,
                "fut_mean": fut_mean,
                "fut_std": fut_std,
                "fut_cam_scale": scales,
            },
        },
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
