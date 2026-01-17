# tracker

NOTE: This branch is experimental. See `docs/CHANGELOG.md`.

**High-fidelity, whole-body motion tracking for the Adam robot family.**

`tracker` is a task package built on top of [`mjlab`](https://github.com/mujocolab/mjlab.git), designed for high-fidelity motion tracking with offline motion datasets.

## Core Features

Developed with a focus on speed, modularity, and high-fidelity simulation, `tracker` provides:
- **Environment Registration**: Seamless integration with the `mjlab` ecosystem.
- **Robot Assets**: Self-contained MJCF and mesh data for the Adam robot family, resolved dynamically.
- **Workflow Utilities**: Specialized CLIs for model inspection, policy evaluation, and training.
- **Flexible Data Sources**: Native support for offline `.npz` files and W&B artifact resolution.

## Getting Started

### Installation
Ensure you have `uv` installed, then synchronize the environment:
```bash
UV_CACHE_DIR=$PWD/.uv-cache uv sync --group dev --frozen
```

### Technical Notes
- **Hardware**: Set `--gpu-ids 0` for NVIDIA acceleration or `None` for CPU execution.
- **Logging**: Use `WANDB_DIR=$PWD/.wandb` to maintain localized W&B run logs.
- **Caches**: Local directories like `.uv-cache/`, `.venv/`, and `.tracker-cache/` are ignored by version control.

## Usage Guide

### 1. Discover Environments
List all registered task IDs and robot configurations:
```bash
uv run tracker-list-envs
```

### 2. Inspect Robot Assets
Validate robot models and verify joint/actuator configurations:
```bash
# Headless: just load and print model summary.
uv run tracker-view-robot --robot-id adam_pro --variant 29 --viewer none

# Interactive viewer (requires a working GUI/OpenGL setup):
uv run tracker-view-robot --robot-id adam_sp --variant 23 --viewer native
```

### 3. Train (offline motion file)
```bash
uv run tracker-train Tracker-Tracking-Flat-Adam-Pro-29 \
  --motion-file /path/to/motion.npz \
  --gpu-ids 0 \
  --agent.max-iterations 5000 \
  --env.scene.num-envs 4096
```

### 4. Play (rollout)
```bash
uv run tracker-play Tracker-Tracking-Flat-Adam-Pro-29 \
  --checkpoint-file /path/to/checkpoint.pt \
  --motion-file /path/to/motion.npz \
  --device cuda:0 \
  --num-envs 8
```

Key arguments:
- `<task_id>`: environment ID from `tracker-list-envs`.
- `--motion-file` (or `--motion_file`): local `.npz` motion file (offline training).
- `--registry-name`: W&B motion artifact alias (use instead of `--motion-file`).
- `--gpu-ids`: CUDA device indices (e.g. `0`).
- `--agent.logger`: logger backend (e.g. `wandb`).
- `--agent.wandb-project`: W&B project name for logging.
- `--agent.max-iterations`: total training iterations.
- `--env.scene.num-envs`: number of parallel environments.


## Development
Run the test suite or build the distribution package:
```bash
uv run pytest -q
uv build
```
