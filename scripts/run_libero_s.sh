#!/usr/bin/env bash
# One-GPU LIBERO run of the paper recipe (model=s, recipe=final).
# Global batch stays 128: micro-batch 32 with gradient accumulation 4.
#
# Prerequisites:
#   export VJEPA_CHECKPOINT=/path/to/vjepa2_1_vitl_dist_vitG_384.pt
#   export LIBERO_DATA=/path/to/libero   # directory with the four suite HDF5 folders
set -euo pipefail
cd "$(dirname "$0")/.."

: "${VJEPA_CHECKPOINT:?set VJEPA_CHECKPOINT to the distilled V-JEPA 2.1 ViT-L weights}"
: "${LIBERO_DATA:?set LIBERO_DATA to the LIBERO HDF5 root}"

PY="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-data/libero}"
MAX_FRAMES="${MAX_FRAMES:-24000}"

"$PY" scripts/prepare_libero.py --output "${DATA_ROOT}" --hdf5-root "${LIBERO_DATA}" --max-frames "${MAX_FRAMES}"
"$PY" scripts/precompute_features.py \
  --manifest "${DATA_ROOT}/manifest.json" \
  --output "${DATA_ROOT}/features" \
  --checkpoint "${VJEPA_CHECKPOINT}" \
  --pool-factor 2 \
  --batch-windows 4 \
  --device cuda
"$PY" scripts/compute_norm_stats.py \
  --manifest "${DATA_ROOT}/manifest.json" \
  --feature-store "${DATA_ROOT}/features" \
  --output "${DATA_ROOT}/stats.pt"
NPROC="${NPROC:-1}" torchrun --standalone --nproc_per_node="${NPROC}" \
  scripts/train.py model=s data=libero recipe=final batch_size=32 num_workers=8
