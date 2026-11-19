"""DDP training with bf16, EMA, gradient clipping, and the Appendix B schedule.

Global batch is ``batch_size * world_size * grad_accum`` and must equal the
recipe's global batch (128 on LIBERO, 400 on RoboTwin). Matrix weights use
AdamW decay 0.01; biases, norms, and LoRA use decay 0. LoRA has its own
learning rate.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from lajepa.data.dataset import EpisodeWindowDataset, collate_windows
from lajepa.data.normalize import PercentileNormalizer, load_stats, save_stats
from lajepa.dropouts import apply_condition_dropout
from lajepa.ema import ModelEMA
from lajepa.models.encoders import FrozenT5, ObservationEncoder, encode_instructions
from lajepa.models.lajepa import LaJEPA
from lajepa.schedule import lora_lr_scale, policy_lr_scale


def run_training(cfg: DictConfig) -> None:
    distributed, rank, world, local_rank = _init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    torch.manual_seed(int(cfg.seed) + rank)
    output = Path(cfg.output_dir)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, output / "config.yaml")
    model = build_model(cfg).to(device)
    stats = load_stats(cfg.data.stats_path) if cfg.data.get("stats_path") else None
    if stats and "features" in stats:
        model.feat_norm.load_stats(stats["features"])
    action_norm = PercentileNormalizer.from_state(stats["action"]) if stats else None
    state_norm = PercentileNormalizer.from_state(stats["state"]) if stats else None
    if action_norm is None or state_norm is None:
        raise FileNotFoundError(
            "data.stats_path must point at a stats file with action, state, and feature moments. "
            "Fit it with scripts/compute_norm_stats.py before training."
        )
    dataset = EpisodeWindowDataset(
        manifest=cfg.data.manifest,
        normalizer_action=action_norm,
        normalizer_state=state_norm,
        feature_store=cfg.data.get("feature_store"),
        cameras=list(cfg.data.cameras),
        image_size=int(cfg.data.image_size),
        horizon=int(cfg.model.horizon),
        temporal_encoding=str(cfg.data.temporal_encoding),
        augment=bool(cfg.recipe.augment) and bool(cfg.recipe.online_obs_encoder),
        vertical_fov_deg=float(cfg.data.vertical_fov_deg),
        return_images=bool(cfg.recipe.online_obs_encoder),
    )
    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(cfg.num_workers),
        collate_fn=collate_windows,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(cfg.num_workers) > 0,
        prefetch_factor=4 if int(cfg.num_workers) > 0 else None,
    )
    optimizer = _optimizer(model, cfg)
    ema = ModelEMA(model, decay=float(cfg.recipe.ema_decay))
    text_encoder = None
    raw_model = model
    step = _resume(cfg, raw_model, ema, optimizer, device)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None, find_unused_parameters=not bool(cfg.recipe.predict_future))
    micro = int(cfg.batch_size)
    global_batch = int(cfg.data.global_batch_size)
    if global_batch % (micro * world) != 0:
        raise ValueError(
            f"global batch {global_batch} is not divisible by batch_size {micro} * world {world}"
        )
    grad_accum = global_batch // (micro * world)
    steps = int(cfg.max_steps)
    pending = 0
    optimizer.zero_grad(set_to_none=True)
    while step < steps:
        if sampler is not None:
            sampler.set_epoch(step)
        for batch in loader:
            batch = _batch_to_device(batch, device)
            if "text_embeds" not in batch:
                if text_encoder is None:
                    text_encoder = _frozen_text_encoder(cfg, device)
                encode_instructions(batch, text_encoder)
            batch = apply_condition_dropout(
                batch,
                camera_prob=float(cfg.recipe.camera_dropout),
                text_prob=float(cfg.recipe.text_dropout),
                proprio_prob=float(cfg.recipe.proprio_dropout),
                eligible_cameras=list(cfg.data.eligible_cameras),
            )
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda" and str(cfg.mixed_precision) == "bf16"):
                out = model(batch)
                loss = out["loss"] / grad_accum
            loss.backward()
            pending += 1
            if pending < grad_accum:
                continue
            _set_lrs(optimizer, step, steps)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), float(cfg.max_grad_norm))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update(raw_model)
            pending = 0
            if rank == 0 and step % int(cfg.log_every) == 0:
                print(
                    f"step {step} loss {float(out['loss']):.4f} "
                    f"action {float(out['action_loss']):.4f} future {float(out['future_loss']):.4f}"
                )
            if rank == 0 and step > 0 and step % int(cfg.save_every) == 0:
                _save(output / f"step_{step:07d}.pt", raw_model, ema, optimizer, step, stats, cfg)
            step += 1
            if step >= steps:
                break
    if rank == 0:
        _save(output / "final.pt", raw_model, ema, optimizer, step, stats, cfg)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


def _resume(cfg, model, ema, optimizer, device) -> int:
    path = cfg.get("resume")
    if not path:
        return 0
    payload = torch.load(Path(str(path)), map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    if payload.get("ema"):
        ema.load_state_dict(payload["ema"])
        for key, value in ema.shadow.items():
            ema.shadow[key] = value.to(device)
    optimizer.load_state_dict(payload["optimizer"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)
    done = int(payload["step"])
    print(f"resumed {path} at step {done}", flush=True)
    return done + 1


def build_model(cfg) -> LaJEPA:
    factors = [int(v) for v in cfg.data.reduction_factors]
    reduction = "identity" if bool(cfg.data.reduction_in_store) else str(cfg.data.reduction)
    obs_encoder = None
    if bool(cfg.recipe.online_obs_encoder):
        variant = str(cfg.model.get("encoder_variant", "vit_l"))
        backbone = VJEPAEncoder_from_variant(variant, cfg)
        checkpoint = cfg.model.get("encoder_checkpoint")
        if variant == "vit_l" and not checkpoint:
            raise FileNotFoundError(
                "model.encoder_checkpoint must point at the frozen V-JEPA 2.1 ViT-L weights"
            )
        if checkpoint:
            backbone.load_checkpoint(checkpoint)
        by_size = cfg.data.get("lora_blocks_by_size")
        if by_size is not None and cfg.model.name in by_size:
            lora_blocks = int(by_size[cfg.model.name])
        else:
            lora_blocks = int(cfg.recipe.lora_blocks)
        obs_encoder = ObservationEncoder(
            backbone,
            lora_last_n=lora_blocks if bool(cfg.recipe.lora.enabled) else 0,
            lora_rank=int(cfg.recipe.lora.rank),
            lora_alpha=int(cfg.recipe.lora.alpha),
            lora_dropout=float(cfg.recipe.lora.dropout),
        )
    return LaJEPA(
        hidden_dim=int(cfg.model.hidden_dim),
        depth=int(cfg.model.depth),
        num_heads=int(cfg.model.num_heads),
        mlp_ratio=float(cfg.model.mlp_ratio),
        feature_dim=int(cfg.model.feature_dim),
        text_dim=int(cfg.model.text_dim),
        action_dim=int(cfg.data.action_dim),
        proprio_dim=int(cfg.data.proprio_dim),
        horizon=int(cfg.model.horizon),
        num_cameras=len(list(cfg.data.cameras)),
        reduction=reduction,
        factors=factors,
        predict_future=bool(cfg.recipe.predict_future),
        gradient_checkpointing=bool(cfg.model.gradient_checkpointing),
        obs_encoder=obs_encoder,
    )


def _frozen_text_encoder(cfg, device: torch.device) -> FrozenT5:
    name = str(cfg.model.get("text_encoder", "google/flan-t5-large"))
    encoder = FrozenT5(name).to(device)
    expected = int(cfg.model.text_dim)
    if int(encoder.output_dim) != expected:
        raise ValueError(f"{name} width {encoder.output_dim} does not match model.text_dim {expected}")
    return encoder


def VJEPAEncoder_from_variant(variant: str, cfg):
    from lajepa.features.vjepa_encoder import VJEPAEncoder

    if variant == "vit_l":
        return VJEPAEncoder.vit_l(rope_grid=int(cfg.model.get("rope_grid", 16)))
    if variant == "tiny":
        return VJEPAEncoder.tiny()
    raise ValueError(f"unknown encoder variant {variant}")


def _optimizer(model: LaJEPA, cfg) -> torch.optim.AdamW:
    decay, nodecay, lora = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" in name:
            lora.append(param)
        elif param.ndim >= 2:
            decay.append(param)
        else:
            nodecay.append(param)
    groups = []
    if decay:
        groups.append({"params": decay, "lr": float(cfg.learning_rate), "weight_decay": float(cfg.weight_decay), "kind": "policy"})
    if nodecay:
        groups.append({"params": nodecay, "lr": float(cfg.learning_rate), "weight_decay": 0.0, "kind": "policy"})
    if lora:
        groups.append({"params": lora, "lr": float(cfg.recipe.lora.lr), "weight_decay": 0.0, "kind": "lora"})
    return torch.optim.AdamW(groups, betas=(0.9, 0.95))


def _set_lrs(optimizer, step: int, total: int) -> None:
    for group in optimizer.param_groups:
        base = group["lr"] if "base_lr" not in group else group["base_lr"]
        group.setdefault("base_lr", group["lr"])
        scale = lora_lr_scale(step, total) if group["kind"] == "lora" else policy_lr_scale(step, total)
        group["lr"] = group["base_lr"] * scale


def _save(path, model, ema, optimizer, step, stats, cfg) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "stats": stats,
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        path,
    )


def _batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def _init_distributed():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:
        return False, 0, 1, 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    return True, dist.get_rank(), dist.get_world_size(), local_rank
