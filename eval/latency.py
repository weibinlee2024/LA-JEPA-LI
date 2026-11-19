"""Policy-call latency.

Matches Appendix C: batch size 1, 20 warmup calls, median of the next 100.
The timed region includes instruction embedding lookup, the host-to-device
copy, the compiled or eager policy forward, and the copy of the action chunk
back to the host. Simulator stepping is not included.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from lajepa.models.lajepa import LaJEPA


def measure(model: LaJEPA, steps: int = 10, warmup: int = 20, repeats: int = 100, device: str = "cuda", compile_model: bool = False) -> float:
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device_t).eval()
    sample = model.sample_actions
    if compile_model and device_t.type == "cuda":
        sample = torch.compile(sample)
    batch = _synthetic_batch(model, device_t)

    def once():
        if device_t.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        actions = sample(batch, steps=steps)
        actions = actions.float().cpu()
        if device_t.type == "cuda":
            torch.cuda.synchronize()
        _ = actions[0, 0, 0].item()
        return (time.perf_counter() - start) * 1000.0

    for _ in range(warmup):
        once()
    samples = [once() for _ in range(repeats)]
    return float(statistics.median(samples))


def _synthetic_batch(model: LaJEPA, device: torch.device) -> dict:
    cameras = model.world.cam_embed.shape[0]
    # Two frames at the encoder resolution is the timed LIBERO setup when an
    # encoder is attached. Stored-feature timing uses a 16x16 grid, which is
    # the native ViT-L/16 token grid at 256 before 2x pooling.
    spatial = 16
    batch = {
        "obs_features": torch.randn(1, cameras, 1, spatial, spatial, model.feat_norm.obs_mean.numel(), device=device),
        "proprio": torch.randn(1, model.condition.proprio[0].in_features, device=device),
        "text_embeds": torch.randn(1, 16, model.condition.text[0].in_features, device=device),
        "text_mask": torch.ones(1, 16, dtype=torch.bool, device=device),
    }
    return batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Median La-JEPA policy-call latency")
    parser.add_argument("--size", choices=["s", "b", "l"], default="s")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    specs = {
        "s": dict(hidden_dim=384, depth=12, num_heads=6),
        "b": dict(hidden_dim=768, depth=12, num_heads=12),
        "l": dict(hidden_dim=1024, depth=24, num_heads=16),
    }
    model = LaJEPA(**specs[args.size], feature_dim=1024, text_dim=1024, action_dim=7, proprio_dim=8, num_cameras=2)
    median = measure(model, steps=args.steps, device=args.device, compile_model=args.compile)
    print(f"La-JEPA {args.size} {args.steps} steps median {median:.2f} ms")


if __name__ == "__main__":
    main()
