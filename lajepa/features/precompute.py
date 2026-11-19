"""Encode every demonstration once and write the feature store.

Future targets are the expensive part: a tube-motion window is 16 frames per
camera, eight times the observation clip. They are frozen, so training never
recomputes them. Observation features are stored too, for the stored-feature
ablation and for normalization statistics; the final recipe still re-encodes
observations online.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from lajepa.data.dataset import _resize, load_episode_arrays
from lajepa.features.indices import clip_frames_for_encoding, future_grid_valid, observation_frame_indices
from lajepa.features.store import FeatureStore
from lajepa.features.vjepa_encoder import DEFAULT_VJEPA_CHECKPOINT, VJEPAEncoder
from lajepa.models.world_dit import reduce_grid


def precompute_manifest(
    manifest: str | Path,
    output: str | Path,
    checkpoint: str | None = None,
    image_size: int = 256,
    temporal_encoding: str = "tube_motion",
    pool_factor: int | None = None,
    batch_windows: int = 4,
    device: str = "cuda",
    overwrite: bool = False,
    append: bool = False,
    min_free_gb: float = 40.0,
) -> FeatureStore:
    manifest = Path(manifest)
    payload = json.loads(manifest.read_text())
    episodes = payload["episodes"]
    root = manifest.parent
    encoder = VJEPAEncoder.vit_l()
    checkpoint = checkpoint or str(DEFAULT_VJEPA_CHECKPOINT)
    encoder.load_checkpoint(checkpoint)
    print(f"loaded frozen V-JEPA encoder from {checkpoint}")
    encoder.eval()
    device_t = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    encoder.to(device_t)
    first = load_episode_arrays(root, episodes[0])
    cameras = int(first["images"].shape[1])
    camera_names = payload.get("cameras") or [f"cam{i}" for i in range(cameras)]
    meta = {
        "cameras": camera_names,
        "image_size": image_size,
        "temporal_encoding": temporal_encoding,
        "feature_dim": encoder.embed_dim,
        "pool_factor": pool_factor,
        "reduction": "mean_pool" if pool_factor else "full",
        "dtype": "float16",
    }
    output = Path(output)
    if append and (output / "index.json").exists():
        store = FeatureStore.open(output, writable=True)
        _check_store_meta(store, meta)
        print(f"appending to {output} ({len(store.episode_ids())} episodes already stored)", flush=True)
    else:
        store = FeatureStore.create(output, meta, overwrite=overwrite)
    for index, episode in enumerate(episodes, start=1):
        episode_id = episode["id"]
        if episode_id in store:
            continue
        free = shutil.disk_usage(output).free
        if free < min_free_gb * 1024**3:
            store.save()
            print(
                f"stopped before {episode_id}: {free / 1024**3:.1f} GiB free, "
                f"keeping {min_free_gb:.0f} GiB. stored {len(store.episode_ids())} episodes",
                flush=True,
            )
            return store
        print(f"episode {index}/{len(episodes)} {episode_id}", flush=True)
        arrays = load_episode_arrays(root, episode)
        images = arrays["images"]
        _write_episode(store, encoder, episode_id, images, camera_names, image_size, temporal_encoding, pool_factor, batch_windows, device_t)
        store.save()
    store.save()
    print(f"finished {len(store.episode_ids())} episodes", flush=True)
    return store


def _check_store_meta(store: FeatureStore, meta: dict) -> None:
    for key in ("cameras", "image_size", "temporal_encoding", "feature_dim", "pool_factor", "reduction", "dtype"):
        if store.meta.get(key) != meta.get(key):
            raise ValueError(f"store {key} is {store.meta.get(key)}, this run expects {meta.get(key)}")


def _write_episode(store, encoder, episode_id, images, camera_names, image_size, encoding, pool_factor, batch_windows, device):
    length, n_cameras = images.shape[0], images.shape[1]
    obs_out = {name: [] for name in camera_names}
    fut_out = {name: [] for name in camera_names}
    masks = []
    for start in range(0, length, batch_windows):
        times = list(range(start, min(length, start + batch_windows)))
        masks.extend(future_grid_valid(t, length, encoding) for t in times)
        for camera, name in enumerate(camera_names):
            obs_batch, obs_valid = _gather(images, camera, times, image_size, encoding, kind="obs")
            obs_feat = _encode(encoder, obs_batch, obs_valid, device, pool_factor)
            fut_feat = _encode_future(encoder, images, camera, times, image_size, encoding, device, pool_factor)
            obs_out[name].append(obs_feat)
            fut_out[name].append(fut_feat)
    obs = {name: np.concatenate(parts, axis=0) for name, parts in obs_out.items()}
    future = {name: np.concatenate(parts, axis=0) for name, parts in fut_out.items()}
    store.add_episode(episode_id, obs, future, np.asarray(masks, dtype=np.uint8))


def _gather(images, camera, times, image_size, encoding, kind):
    clips = []
    valids = []
    for t in times:
        if kind == "obs":
            mode = "static" if encoding == "static" else "tube_motion"
            indices = observation_frame_indices(t, mode)
            frames = [_resize(images[min(index, images.shape[0] - 1), camera], image_size) for index in indices]
            valid = [index < images.shape[0] for index in indices]
        else:
            pieces = clip_frames_for_encoding(t, encoding)
            # Joint tube-motion is one clip. Tube/static are several clips; the
            # caller encodes those separately, so this path only builds the
            # tube-motion clip here and leaves the others to _encode_future.
            indices = pieces[0]
            frames = [_resize(images[min(index, images.shape[0] - 1), camera], image_size) for index in indices]
            valid = [index < images.shape[0] for index in indices]
        clip = np.stack(frames, axis=0)  # [T, H, W, 3]
        clips.append(clip)
        valids.append(valid)
    return np.stack(clips, axis=0), np.asarray(valids, dtype=bool)


def _encode(encoder, clips, valid, device, pool_factor):
    tensor = torch.from_numpy(clips).permute(0, 4, 1, 2, 3).contiguous().float() / 255.0
    tensor = tensor.to(device)
    frame_valid = torch.from_numpy(valid).to(device)
    with torch.no_grad():
        features = encoder(tensor, frame_valid=frame_valid)
    features = _maybe_pool(features, pool_factor)
    return features.detach().float().cpu().numpy()


def _encode_future(encoder, images, camera, times, image_size, encoding, device, pool_factor):
    """Encode future clips so the stored tensor is ``[B, 8, H, W, D]``.

    Tube-motion is one joint 16-frame forward. Tube and static encode each of
    the eight targets on its own, so those grids do not attend across targets.
    """
    if encoding == "tube_motion":
        clips, valid = _gather(images, camera, times, image_size, encoding, kind="future")
        return _encode(encoder, clips, valid, device, pool_factor)
    grids = []
    for target in range(8):
        clips, valids = [], []
        for t in times:
            indices = clip_frames_for_encoding(t, encoding)[target]
            frames = [_resize(images[min(index, images.shape[0] - 1), camera], image_size) for index in indices]
            clips.append(np.stack(frames, axis=0))
            valids.append([index < images.shape[0] for index in indices])
        grids.append(_encode(encoder, np.stack(clips), np.asarray(valids, dtype=bool), device, pool_factor))
    return np.concatenate(grids, axis=1)


def _maybe_pool(features: torch.Tensor, pool_factor: int | None) -> torch.Tensor:
    if not pool_factor:
        return features
    return reduce_grid(features, "mean_pool", int(pool_factor))


def verify_indices(t: int = 0) -> None:
    frames = future_frame_indices_safe(t)
    if frames != [t + 2 * (i + 1) for i in range(16)]:
        raise AssertionError(frames)
    if frames[-1] != t + 32 or len(frames) != 16:
        raise AssertionError("tube-motion horizon is not 32 frames")
    if len(frames) // 2 != 8:
        raise AssertionError("tube-motion must produce 8 temporal grids")


def future_frame_indices_safe(t: int) -> list[int]:
    from lajepa.features.indices import future_frame_indices

    return future_frame_indices(t, "tube_motion")


def verify_store_matches_encoder(store: FeatureStore, encoder: VJEPAEncoder, images: np.ndarray, episode_id: str, t: int, camera: int, image_size: int, device: str) -> float:
    """Max absolute error between a stored future grid and a fresh encode."""
    name = store.meta["cameras"][camera]
    stored = store.read_future(episode_id, name, t)
    encoding = store.meta["temporal_encoding"]
    fresh = _encode_future(
        encoder, images, camera, [t], image_size, encoding, torch.device(device), store.meta.get("pool_factor")
    )[0]
    return float(np.max(np.abs(stored - fresh)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute La-JEPA V-JEPA features")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default=str(DEFAULT_VJEPA_CHECKPOINT))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--temporal-encoding", default="tube_motion", choices=["tube_motion", "tube", "static"])
    parser.add_argument("--pool-factor", type=int, default=0)
    parser.add_argument("--batch-windows", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--min-free-gb", type=float, default=40.0)
    args = parser.parse_args()
    verify_indices()
    precompute_manifest(
        args.manifest,
        args.output,
        checkpoint=args.checkpoint,
        image_size=args.image_size,
        temporal_encoding=args.temporal_encoding,
        pool_factor=args.pool_factor or None,
        batch_windows=args.batch_windows,
        device=args.device,
        overwrite=args.overwrite,
        append=args.append,
        min_free_gb=args.min_free_gb,
    )


if __name__ == "__main__":
    main()
