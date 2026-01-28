# tracker

NOTE: This branch is experimental. See `docs/CHANGELOG.md`.

**High-fidelity, whole-body motion tracking for the Adam robot family.**

`tracker` is a task package built on top of [`mjlab`](https://github.com/mujocolab/mjlab.git), designed for high-fidelity motion tracking with offline motion datasets.

## Status (this branch)

This branch implements **Phase-2 multi-clip training support**:
- Packed motion datasets (`tracker-pack-motions`) with train/val/test splits.
- Multi-clip offline training/eval via `--motion-pack` + `--motion-split`.
- Per-clip **adaptive time-bin sampling** for motion packs via `--motion-pack-sampling-mode adaptive`.

**Planned next (new branch):** global curriculum / difficulty across clips (clip-wise sampling weights).

## Core Features

Developed with a focus on speed, modularity, and high-fidelity simulation, `tracker` provides:
- **Environment Registration**: Seamless integration with the `mjlab` ecosystem.
- **Robot Assets**: Self-contained MJCF and mesh data for the Adam robot family, resolved dynamically.
- **Workflow Utilities**: Specialized CLIs for model inspection, policy evaluation, and training.
- **Flexible Data Sources**: Native support for offline single-clip `.npz` files and packed multi-clip motion datasets.

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

### 3. Pack a multi-clip dataset (optional)
Create a packed dataset directory from a YAML manifest of clips:
```bash
uv run tracker-pack-motions \
  --manifest /path/to/manifest.yaml \
  --out /path/to/pack_dir
```

### 4. Train
Single-clip offline training:
```bash
uv run tracker-train Tracker-Tracking-Flat-Adam-Pro-29 \
  --motion-file /path/to/motion.npz \
  --gpu-ids 0 \
  --agent.max-iterations 5000 \
  --env.scene.num-envs 4096
```

Packed multi-clip offline training:
```bash
uv run tracker-train Tracker-Tracking-Flat-Adam-Pro-29 \
  --motion-pack /path/to/pack_dir \
  --motion-split train \
  --motion-pack-sampling-mode adaptive \
  --gpu-ids 0 \
  --agent.max-iterations 5000 \
  --env.scene.num-envs 4096
```

Motion-pack sampling modes:
- `adaptive`: per-clip adaptive time-bin sampling (recommended for large packs).
- `uniform`: uniform time sampling.
- `start`: always start at `t=0`.

### 5. Evaluate (val/test split)
```bash
uv run tracker-eval Tracker-Tracking-Flat-Adam-Pro-29 \
  --checkpoint /path/to/model.pt \
  --motion-pack /path/to/pack_dir \
  --motion-split val \
  --num-episodes 100 \
  --num-envs 128 \
  --device cuda:0
```

Tip: if you split commands across lines in bash, end each line with `\` (otherwise flags like `--motion-split` may be interpreted as separate shell commands).

Key arguments:
- `<task_id>`: environment ID from `tracker-list-envs`.
- `--motion-file` (or `--motion_file`): local `.npz` motion file (offline training).
- `--registry-name`: W&B motion artifact alias (use instead of `--motion-file`).
- `--motion-pack`: path to a packed multi-clip dataset directory (offline training).
- `--motion-split`: `train|val|test` split name when using `--motion-pack`.
- `--motion-pack-sampling-mode`: `adaptive|uniform|start` when using `--motion-pack`.
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
