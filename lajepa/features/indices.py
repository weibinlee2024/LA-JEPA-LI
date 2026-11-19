"""Frame indices for tube-motion, tube, and static V-JEPA clips.

Tube-motion (the final recipe) feeds one 16-frame clip whose frames sit at
``t+2, t+4, ..., t+32``. Tubelet size 2 then yields 8 temporal grids, and
those grids attend over the whole horizon inside the encoder. The last index
is exactly 32 frames ahead of ``t``; using ``t+1 ... t+31`` is an off-by-one.
"""

from __future__ import annotations

TEMPORAL_ENCODINGS = ("tube_motion", "tube", "static")


def observation_frame_indices(t: int, encoding: str = "tube_motion") -> list[int]:
    """Frames that form the observation clip ending at ``t``.

    Tube / tube-motion: the two raw frames ``(t-1, t)``, aligned with a
    tubelet. At the first frame the clip repeats frame 0. Static encoding
    uses only the current frame.
    """
    if encoding not in TEMPORAL_ENCODINGS:
        raise ValueError(f"Unknown temporal encoding {encoding!r}")
    if t < 0:
        raise ValueError(f"t must be >= 0, got {t}")
    if encoding == "static":
        return [t]
    return [t - 1 if t >= 1 else 0, t]


def future_frame_indices(
    t: int,
    encoding: str = "tube_motion",
    num_frames: int = 16,
    stride: int = 2,
    horizon: int = 32,
) -> list[int]:
    """Absolute frame indices of the future targets for decision time ``t``.

    tube_motion: ``num_frames`` indices (default 16) starting at ``t+stride``
    and stepping by ``stride``, so the last frame is ``t+horizon`` when
    ``stride * num_frames == horizon``.

    tube / static: 8 target times spaced by ``horizon / 8`` (default 4),
    i.e. ``t+4, t+8, ..., t+32``. Tube encoding later expands each target
    into a 2-frame clip; static encoding uses the single target frame.
    """
    if encoding not in TEMPORAL_ENCODINGS:
        raise ValueError(f"Unknown temporal encoding {encoding!r}")
    if t < 0:
        raise ValueError(f"t must be >= 0, got {t}")
    if encoding == "tube_motion":
        if stride * num_frames != horizon:
            raise ValueError(
                f"tube-motion expects stride * num_frames == horizon, "
                f"got {stride} * {num_frames} != {horizon}"
            )
        return [t + stride * (i + 1) for i in range(num_frames)]
    n_grids = 8
    step = horizon // n_grids
    return [t + step * (i + 1) for i in range(n_grids)]


def future_grid_valid(t: int, episode_length: int, encoding: str = "tube_motion") -> list[bool]:
    """Which of the 8 future temporal grids have both source frames inside the episode.

    A grid is supervised only when every raw frame it was encoded from exists.
    Padded frames past ``episode_length`` are masked out of the future loss.
    """
    if episode_length <= 0:
        raise ValueError("episode_length must be positive")
    if encoding == "tube_motion":
        frames = future_frame_indices(t, encoding="tube_motion")
        valid = []
        for grid in range(8):
            pair = frames[2 * grid : 2 * grid + 2]
            valid.append(all(index < episode_length for index in pair))
        return valid
    frames = future_frame_indices(t, encoding=encoding)
    if encoding == "static":
        return [index < episode_length for index in frames]
    # Tube: the clip ending at the target also needs the previous frame,
    # which is inside the episode whenever the target itself is (or is frame 0).
    return [index < episode_length for index in frames]


def action_indices(t: int, horizon: int = 32) -> list[int]:
    return [t + i for i in range(horizon)]


def action_valid(t: int, episode_length: int, horizon: int = 32) -> list[bool]:
    return [index < episode_length for index in action_indices(t, horizon)]


def clip_frames_for_encoding(t: int, encoding: str = "tube_motion") -> list[list[int]]:
    """Return the encoder clips (each a list of absolute frame indices).

    tube_motion yields one 16-frame clip. tube yields eight 2-frame clips.
    static yields eight 1-frame clips. Observation clips are separate; use
    ``observation_frame_indices``.
    """
    if encoding == "tube_motion":
        return [future_frame_indices(t, encoding="tube_motion")]
    targets = future_frame_indices(t, encoding=encoding)
    if encoding == "static":
        return [[index] for index in targets]
    clips = []
    for index in targets:
        prev = index - 1 if index >= 1 else 0
        clips.append([prev, index])
    return clips
