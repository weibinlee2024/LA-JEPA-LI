"""Hydra entry point. Launch with torchrun for the 4-GPU DDP setup."""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lajepa.train import run_training


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()
