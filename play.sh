#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -gt 0 ]; then
  exec uv run tracker-play "$@"
fi

exec uv run tracker-play Tracker-Tracking-Flat-Adam-Pro-29 \
  --checkpoint-file /path/to/checkpoint.pt \
  --motion-file /path/to/motion.npz \
  --device cuda:0 \
  --num-envs 8
