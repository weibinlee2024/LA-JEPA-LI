#!/usr/bin/env bash
# Encode tube-motion future targets into the sharded feature store.
# LIBERO passes --pool-factor 2 so the stored grids are already 2x2-pooled.
# RoboTwin omits pooling; the model patch-groups full-resolution tokens.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/precompute_features.py "$@"
