"""Sharded memmap feature store.

Each episode stores, per camera, observation tokens and future tokens plus a
``uint8`` mask over the 8 future temporal grids. Arrays are packed into
fixed-capacity shards so a RoboTwin-scale library stays a handful of files
instead of one file per episode. Two granularities share the format: full
V-JEPA grids (RoboTwin patch grouping) and spatially pooled grids (LIBERO).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

STORE_VERSION = 1


class FeatureStore:
    def __init__(self, root: str | Path, meta: dict, index: dict, writable: bool = False):
        self.root = Path(root)
        self.meta = meta
        self.index = index
        self.writable = writable
        self._maps: dict[tuple[int, str], np.memmap] = {}

    @classmethod
    def create(cls, root: str | Path, meta: dict, overwrite: bool = False) -> "FeatureStore":
        root = Path(root)
        if root.exists() and any(root.iterdir()) and not overwrite:
            raise FileExistsError(f"feature store already exists: {root}")
        root.mkdir(parents=True, exist_ok=True)
        meta = dict(meta)
        meta["version"] = STORE_VERSION
        meta.setdefault("dtype", "float16")
        index = {"episodes": {}, "shards": {"float16": [], "uint8": []}}
        store = cls(root, meta, index, writable=True)
        store.save()
        return store

    @classmethod
    def open(cls, root: str | Path, writable: bool = False) -> "FeatureStore":
        root = Path(root)
        meta = json.loads((root / "meta.json").read_text())
        index = json.loads((root / "index.json").read_text())
        if int(meta.get("version", 0)) != STORE_VERSION:
            raise ValueError(f"unsupported feature store version {meta.get('version')}")
        return cls(root, meta, index, writable=writable)

    def save(self) -> None:
        (self.root / "meta.json").write_text(json.dumps(self.meta, indent=2))
        (self.root / "index.json").write_text(json.dumps(self.index))
        for memmap in self._maps.values():
            memmap.flush()

    def __contains__(self, episode_id: str) -> bool:
        return episode_id in self.index["episodes"]

    def episode_ids(self) -> list[str]:
        return list(self.index["episodes"].keys())

    def add_episode(
        self,
        episode_id: str,
        obs: dict[str, np.ndarray],
        future: dict[str, np.ndarray],
        future_mask: np.ndarray,
    ) -> None:
        """Write one episode.

        ``obs[cam]`` is ``[T, T_obs, H, W, D]``, ``future[cam]`` is
        ``[T, 8, H, W, D]``, and ``future_mask`` is ``[T, 8]`` with 1 = valid.
        """
        if not self.writable:
            raise RuntimeError("store is opened read-only")
        if episode_id in self.index["episodes"]:
            raise KeyError(f"episode {episode_id} is already in the store")
        cameras = list(self.meta["cameras"])
        if set(obs) != set(cameras) or set(future) != set(cameras):
            raise ValueError(f"expected cameras {cameras}, got obs {sorted(obs)} future {sorted(future)}")
        record: dict = {"cameras": {}}
        length = None
        for camera in cameras:
            obs_array = np.ascontiguousarray(obs[camera], dtype=np.float16)
            fut_array = np.ascontiguousarray(future[camera], dtype=np.float16)
            if length is None:
                length = int(obs_array.shape[0])
            if obs_array.shape[0] != length or fut_array.shape[0] != length:
                raise ValueError(f"camera {camera} length does not match the episode")
            record["cameras"][camera] = {
                "obs": self._append("float16", obs_array),
                "future": self._append("float16", fut_array),
            }
        mask = np.ascontiguousarray(future_mask.astype(np.uint8))
        if mask.shape != (length, 8):
            raise ValueError(f"future_mask must be {(length, 8)}, got {mask.shape}")
        record["length"] = length
        record["future_mask"] = self._append("uint8", mask)
        self.index["episodes"][episode_id] = record

    def read_obs(self, episode_id: str, camera: str, t: int) -> np.ndarray:
        spec = self.index["episodes"][episode_id]["cameras"][camera]["obs"]
        array = self._read(spec)
        return np.array(array[t], dtype=np.float32)

    def read_future(self, episode_id: str, camera: str, t: int) -> np.ndarray:
        spec = self.index["episodes"][episode_id]["cameras"][camera]["future"]
        array = self._read(spec)
        return np.array(array[t], dtype=np.float32)

    def read_future_mask(self, episode_id: str, t: int) -> np.ndarray:
        spec = self.index["episodes"][episode_id]["future_mask"]
        array = self._read(spec)
        return np.array(array[t], dtype=np.uint8)

    def length(self, episode_id: str) -> int:
        return int(self.index["episodes"][episode_id]["length"])

    def _append(self, dtype: str, array: np.ndarray) -> dict:
        flat = np.ascontiguousarray(array).ravel()
        shard_id, memmap, offset = self._reserve(dtype, int(flat.size))
        memmap[offset : offset + flat.size] = flat
        return {"shard": shard_id, "offset": offset, "shape": list(array.shape), "dtype": dtype}

    def _reserve(self, dtype: str, count: int) -> tuple[int, np.memmap, int]:
        shards = self.index["shards"][dtype]
        itemsize = np.dtype(dtype).itemsize
        capacity = int(self.meta.get("shard_capacity", 64 * 1024 * 1024))
        if not shards or shards[-1]["filled"] + count > shards[-1]["capacity"]:
            shard_id = len(shards)
            path = self.root / f"shard_{dtype}_{shard_id:05d}.mmap"
            # A single array larger than the default capacity gets its own shard.
            this_capacity = max(capacity, count)
            memmap = np.memmap(path, mode="w+", dtype=dtype, shape=(this_capacity,))
            shards.append({"path": path.name, "capacity": this_capacity, "filled": 0})
            self._maps[(shard_id, dtype)] = memmap
        else:
            shard_id = len(shards) - 1
            memmap = self._open_shard(shard_id, dtype, writable=True)
        offset = int(shards[shard_id]["filled"])
        shards[shard_id]["filled"] = offset + count
        return shard_id, memmap, offset

    def _open_shard(self, shard_id: int, dtype: str, writable: bool) -> np.memmap:
        key = (shard_id, dtype)
        cached = self._maps.get(key)
        if cached is not None:
            return cached
        spec = self.index["shards"][dtype][shard_id]
        path = self.root / spec["path"]
        mode = "r+" if writable else "r"
        memmap = np.memmap(path, mode=mode, dtype=dtype, shape=(int(spec["capacity"]),))
        self._maps[key] = memmap
        return memmap

    def _read(self, spec: dict) -> np.ndarray:
        memmap = self._open_shard(int(spec["shard"]), spec["dtype"], writable=False)
        offset = int(spec["offset"])
        count = int(np.prod(spec["shape"]))
        flat = memmap[offset : offset + count]
        return np.array(flat).reshape(spec["shape"])
