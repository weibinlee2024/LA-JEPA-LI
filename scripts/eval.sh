#!/usr/bin/env bash
# Usage:
#   bash scripts/eval.sh libero  --checkpoint runs/lajepa/final.pt --suite libero_spatial
#   bash scripts/eval.sh latency --size s --steps 10
set -euo pipefail
cd "$(dirname "$0")/.."
target="${1:-libero}"
shift || true
case "${target}" in
  libero) python eval/libero_eval.py "$@" ;;
  latency) python eval/latency.py "$@" ;;
  *) echo "expected 'libero' or 'latency'" >&2; exit 1 ;;
esac
