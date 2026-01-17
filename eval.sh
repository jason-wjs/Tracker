#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -gt 0 ]; then
  exec uv run tracker-eval "$@"
fi

exec uv run tracker-eval Tracker-Tracking-Flat-Adam-Pro-29 \
  --checkpoint /path/to/model.pt \
  --motion-pack /path/to/pack_dir \
  --motion-split val \
  --num-episodes 100 \
  --num-envs 128 \
  --device cuda:0
