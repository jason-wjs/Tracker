#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -gt 0 ]; then
  exec uv run tracker-train "$@"
fi

exec uv run tracker-train Tracker-Tracking-Flat-Adam-SP-29 \
  --motion-file /path/to/motion.npz \
  --gpu-ids 0 \
  --agent.max-iterations 5000 \
  --env.scene.num-envs 4096
