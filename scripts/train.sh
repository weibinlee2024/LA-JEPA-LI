#!/usr/bin/env bash
# 4 x RTX 4090 DDP. Global batch is data.global_batch_size (128 LIBERO / 400 RoboTwin).
# Example:
#   NPROC=4 BATCH=8 bash scripts/train.sh model=s data=libero recipe=final
set -euo pipefail
cd "$(dirname "$0")/.."
NPROC="${NPROC:-4}"
torchrun --standalone --nproc_per_node="${NPROC}" scripts/train.py "$@"
