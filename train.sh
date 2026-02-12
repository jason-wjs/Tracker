#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -gt 0 ]; then
  exec uv run tracker-train "$@"
fi

# exec uv run tracker-train Tracker-Tracking-Flat-Adam-Pro-29 \
#   --motion-file /path/to/motion.npz \
#   --gpu-ids 0 \
#   --agent.max-iterations 5000 \
#   --env.scene.num-envs 4096

exec uv run tracker-train Tracker-Teleop-Flat-Adam-Pro-29-No-State-Estimation \
    --motion-pack /home/humanoid/Downloads/Data/GMR_test/bvh_test1/pack_adam_pro_29_bvh_test1 \
    --motion-split train \
    --gpu-ids 0 \
    --agent.logger wandb \
    --agent.max-iterations 10000 \
    --env.scene.num-envs 4096