"""Build a LIBERO manifest from the local MuJoCo demonstration HDF5 files.

The paper trains on the four LIBERO suites. A full V-JEPA feature store for
every frame is several hundred gigabytes, so ``--max-frames`` keeps a
round-robin slice that still covers all four suites. ``manifest_all.json``
always lists every demonstration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def instruction_from_name(name: str) -> str:
    stem = name
    for suffix in ("_demo.hdf5", ".hdf5"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem.replace("_", " ")


def collect_episodes(hdf5_root: Path) -> list[dict]:
    episodes = []
    for suite in SUITES:
        directory = hdf5_root / suite
        files = sorted(directory.glob("*.hdf5"))
        if not files:
            raise FileNotFoundError(f"no HDF5 files in {directory}")
        for path in files:
            with h5py.File(path, "r") as handle:
                demos = sorted(handle["data"].keys(), key=lambda key: int(key.split("_")[-1]))
                instruction = instruction_from_name(path.name)
                for demo in demos:
                    episodes.append(
                        {
                            "id": f"{suite}/{path.stem}/{demo}",
                            "path": str(path),
                            "demo": demo,
                            "instruction": instruction,
                            "num_frames": int(handle["data"][demo]["actions"].shape[0]),
                            "suite": suite,
                        }
                    )
    return episodes


def select_episodes(episodes: list[dict], max_frames: int) -> list[dict]:
    """Round-robin across suites until adding another episode would pass the budget."""
    buckets = {suite: [episode for episode in episodes if episode["suite"] == suite] for suite in SUITES}
    cursors = {suite: 0 for suite in SUITES}
    chosen: list[dict] = []
    frames = 0
    while True:
        added = False
        for suite in SUITES:
            cursor = cursors[suite]
            if cursor >= len(buckets[suite]):
                continue
            episode = buckets[suite][cursor]
            cursors[suite] = cursor + 1
            if frames and frames + episode["num_frames"] > max_frames:
                continue
            chosen.append(episode)
            frames += episode["num_frames"]
            added = True
            if frames >= max_frames:
                return chosen
        if not added:
            return chosen


def write_manifest(path: Path, episodes: list[dict]) -> None:
    payload = {
        "cameras": ["agentview", "wrist"],
        "episodes": episodes,
    }
    path.write_text(json.dumps(payload))


def main() -> None:
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hdf5-root",
        type=Path,
        default=Path(os.environ["LIBERO_DATA"]) if os.environ.get("LIBERO_DATA") else None,
        help="Root with libero_{spatial,object,goal,10} HDF5 folders (or set LIBERO_DATA)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=24000)
    args = parser.parse_args()
    if args.hdf5_root is None:
        parser.error("pass --hdf5-root or set LIBERO_DATA")
    episodes = collect_episodes(args.hdf5_root)
    selected = select_episodes(episodes, args.max_frames)
    args.output.mkdir(parents=True, exist_ok=True)
    write_manifest(args.output / "manifest_all.json", episodes)
    write_manifest(args.output / "manifest.json", selected)
    frames = sum(episode["num_frames"] for episode in selected)
    print(
        f"wrote {args.output / 'manifest.json'} episodes={len(selected)} frames={frames} "
        f"(all episodes={len(episodes)})"
    )


if __name__ == "__main__":
    main()
