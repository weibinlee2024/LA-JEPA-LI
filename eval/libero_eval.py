"""Closed-loop LIBERO / LIBERO-Plus evaluation.

Headline protocol from Appendix B: MuJoCo 3.3.2, 10 settling steps, and the
upstream per-suite limits (220, 280, 300, 520). Non-language LIBERO-Plus
variants are scored with the corrected base-task instruction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval.libero_plus_instructions import load_instruction_overrides, resolve_instruction
from lajepa.inference import LaJEPAPolicy

SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
SETTLE_STEPS = 10
ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "libero_plus_fixed_instructions"


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    w = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(max(1.0 - w * w, 0.0))
    if den < 1e-8:
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * np.arccos(w) / den).astype(np.float32)


def libero_proprio(obs: dict) -> np.ndarray:
    eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
    eef_aa = quat2axisangle(obs["robot0_eef_quat"])
    gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)
    return np.concatenate([eef_pos, eef_aa, gripper], axis=0)


def _flip(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(image[::-1, ::-1])


def prepare_images(obs: dict, image_size: int = 256):
    import torch
    import torch.nn.functional as F

    views = [
        _flip(obs["agentview_image"]),
        _flip(obs["robot0_eye_in_hand_image"]),
    ]
    tensors = []
    for view in views:
        tensor = torch.from_numpy(view).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        tensor = F.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)
        tensors.append(tensor)
    # [cameras, time=2, 3, H, W]. The previous frame is repeated on the first call.
    frame = torch.cat(tensors, dim=0)
    return frame


class FrameHistory:
    def __init__(self):
        self.prev = None

    def update(self, frame):
        import torch

        if self.prev is None:
            clip = frame.unsqueeze(1).repeat(1, 2, 1, 1, 1)
        else:
            clip = torch.stack([self.prev, frame], dim=1)
        self.prev = frame
        return clip


def evaluate(checkpoint: str, suite: str = "libero_spatial", episodes: int = 50, device: str = "cuda", image_size: int = 256, instruction_dir: str | None = None) -> dict:
    import torch
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    policy = LaJEPAPolicy.from_checkpoint(checkpoint, device=device)
    overrides = load_instruction_overrides(instruction_dir or ASSET_DIR)
    task_suite = benchmark.get_benchmark_dict()[suite]()
    max_steps = SUITE_MAX_STEPS[suite]
    successes = 0
    total = 0
    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env = OffScreenRenderEnv(bddl_file_name=task.bddl_file, camera_heights=256, camera_widths=256)
        base_instruction = task.language if hasattr(task, "language") else task.name.replace("_", " ")
        for episode in range(min(episodes, len(initial_states))):
            env.reset()
            obs = env.set_init_state(initial_states[episode])
            for _ in range(SETTLE_STEPS):
                obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32))
            history = FrameHistory()
            policy.reset()
            done = False
            instruction = resolve_instruction(f"{suite}/{task_id}", base_instruction, category=None, base_instruction=base_instruction, overrides=overrides)
            text, text_mask = _encode_text(policy, instruction)
            for _ in range(max_steps):
                frame = prepare_images(obs, image_size)
                clip = history.update(frame)
                action = policy.act(clip, libero_proprio(obs), text, text_mask)
                obs, _, done, _ = env.step(action.astype(np.float32))
                if done:
                    break
            successes += int(done)
            total += 1
        env.close()
    return {"suite": suite, "successes": successes, "episodes": total, "success_rate": successes / max(total, 1)}


def _encode_text(policy, instruction: str):
    """Bag a cached vector when the checkpoint did not ship a T5 module.

    Training caches FLAN-T5 states. Evaluation without that cache cannot
    invent them; load ``google/flan-t5-large`` when transformers is present.
    """
    import torch
    from lajepa.models.encoders import FrozenT5

    encoder = getattr(policy, "_t5", None)
    if encoder is None:
        encoder = FrozenT5()
        encoder.to(policy.device)
        policy._t5 = encoder
    hidden, mask = encoder([instruction])
    return hidden[0].cpu(), mask[0].cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate La-JEPA on LIBERO")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--instruction-dir", default=None)
    args = parser.parse_args()
    result = evaluate(args.checkpoint, args.suite, args.episodes, args.device, instruction_dir=args.instruction_dir)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
